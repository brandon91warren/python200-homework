# Video Link - https://www.youtube.com/watch?v=RsEW98S35bg #
import json
from datetime import date

import requests
from openai import OpenAI
from prefect import flow, task
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


ACCOUNT_URL = "https://brandonctd2026sa.blob.core.windows.net"
CONTAINER_NAME = "pipeline-data"

LATITUDE = 35.2271
LONGITUDE = -80.8431
CITY = "Charlotte, NC"


@task(retries=2, retry_delay_seconds=10)
def extract_weather() -> dict:
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,precipitation",
        "forecast_days": 7,
        "timezone": "auto",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    print(f"Extracted 7-day hourly weather data for {CITY}")
    return response.json()


@task
def transform_weather(raw_data: dict) -> list:
    client = OpenAI()

    hourly = raw_data["hourly"]
    times = hourly["time"]
    temperatures = hourly["temperature_2m"]
    precipitation = hourly["precipitation"]

    enriched_records = []

    for i in range(len(times)):
        record = {
            "time": times[i],
            "temperature_2m": temperatures[i],
            "precipitation": precipitation[i],
        }

        if i < 24:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are classifying hourly weather conditions for outdoor running.\n"
                            "Given a temperature in Celsius and a precipitation amount in mm,\n"
                            "classify the conditions as exactly one of: good, marginal, or bad.\n"
                            "Reply with that one word only -- no punctuation, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Temperature: {temperatures[i]} Celsius\n"
                            f"Precipitation: {precipitation[i]} mm"
                        ),
                    },
                ],
            )

            classification = response.choices[0].message.content.strip().lower()

            if classification not in ["good", "marginal", "bad"]:
                classification = "unknown"

            record["running_condition"] = classification
        else:
            record["running_condition"] = None

        enriched_records.append(record)

        if (i + 1) % 6 == 0 and i < 24:
            print(f"Classified {i + 1} records...")

    print(f"Transformed {len(enriched_records)} hourly records")
    return enriched_records


@task
def load_weather(records: list) -> str:
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    json_bytes = json.dumps(records, indent=2).encode("utf-8")

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )

    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(json_bytes, overwrite=True)

    print(f"Uploaded enriched records to {blob_path}")
    print(f"Bytes uploaded: {len(json_bytes)}")

    return blob_path


@flow(log_prints=True)
def weather_etl_pipeline():
    raw_data = extract_weather()
    enriched_records = transform_weather(raw_data)
    final_blob_path = load_weather(enriched_records)

    print(f"ETL pipeline completed successfully. Final blob path: {final_blob_path}")


if __name__ == "__main__":
    weather_etl_pipeline()