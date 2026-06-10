"""
Reflection:
Classifying weather conditions for outdoor running is not the best use of an LLM
because the inputs are structured numbers: temperature and precipitation.
Deterministic code could probably do this faster, cheaper, and more consistently.
A rule-based approach would gain reliability and lower cost, but it would lose
some flexibility if the criteria became more subjective or complex.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# --- Setup ---

ACCOUNT_URL = "https://brandonctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

TODAY = date.today().isoformat()
RAW_BLOB = "raw/2026-06-02/weather.json"
PROCESSED_BLOB = f"processed/{TODAY}/weather_classified.json"
FALLBACK_PATH = Path("assignments/resources/weather_raw.json")
OUTPUT_PATH = Path("outputs/first_10_records.json")

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


# --- Helpers ---

def reshape_hourly_data(weather_json):
    hourly = weather_json["hourly"]

    records = []
    for time, temp, precip in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"],
    ):
        records.append({
            "time": time,
            "temperature_2m": temp,
            "precipitation": precip,
        })

    return records


def classify_record(client, record):
    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in VALID_LABELS:
        return "unknown"

    return label


# --- Step 1: Read ---

def load_raw_weather(container_client):
    try:
        print(f"Trying to download blob: {RAW_BLOB}")
        blob_client = container_client.get_blob_client(RAW_BLOB)
        raw_bytes = blob_client.download_blob().readall()
        weather_json = json.loads(raw_bytes)
        print("Loaded weather data from Blob Storage.")
        return weather_json

    except Exception as e:
        print(f"Could not load today's blob: {e}")
        print(f"Loading fallback file: {FALLBACK_PATH}")

        with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)


# --- Main Script ---

def main():
    load_dotenv()

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )
    container_client = blob_service_client.get_container_client(CONTAINER)

    openai_client = OpenAI()

    weather_json = load_raw_weather(container_client)
    records = reshape_hourly_data(weather_json)

    records_to_process = records[:24]
    enriched_records = []

    # --- Step 2: Transform ---

    for index, record in enumerate(records_to_process, start=1):
        conditions = classify_record(openai_client, record)

        enriched_record = record.copy()
        enriched_record["conditions"] = conditions
        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Processed {index} records...")

    # --- Step 3: Write ---

    processed_json = json.dumps(enriched_records, indent=2)

    processed_blob_client = container_client.get_blob_client(PROCESSED_BLOB)
    processed_blob_client.upload_blob(processed_json, overwrite=True)

    print(f"Uploaded enriched records to: {PROCESSED_BLOB}")

    # --- Step 4: Spot-Check ---

    downloaded_processed = processed_blob_client.download_blob().readall()
    downloaded_records = json.loads(downloaded_processed)

    df = pd.DataFrame(downloaded_records)

    print("\nCondition counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    # --- Step 5: Save Output ---

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_records[:10], f, indent=2)

    print(f"\nSaved first 10 records to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()