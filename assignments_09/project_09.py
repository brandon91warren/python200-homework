import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# --- Setup ---

ACCOUNT_URL = "https://brandonctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

LATITUDE = 35.2271
LONGITUDE = -80.8431


def main():
    # --- Step 1: Extract ---

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    weather_data = response.json()

    # --- Step 2: Serialize ---

    json_bytes = json.dumps(weather_data, indent=2).encode("utf-8")

    # --- Step 3: Load ---

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential
    )

    container_client = blob_service_client.get_container_client(CONTAINER)

    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"

    container_client.upload_blob(
        name=blob_path,
        data=json_bytes,
        overwrite=True
    )

    print(f"Uploaded {blob_path}")
    print(f"Bytes uploaded: {len(json_bytes)}")

    # --- Step 4: Verify ---

    print("\nBlobs in container:")

    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")

    # --- Step 5: Read Back ---

    downloaded_blob = container_client.download_blob(blob_path)
    downloaded_bytes = downloaded_blob.readall()

    downloaded_data = json.loads(downloaded_bytes.decode("utf-8"))

    hourly_df = pd.DataFrame(downloaded_data["hourly"])

    print("\nFirst 5 rows of hourly weather data:")
    print(hourly_df.head())

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    output_path = outputs_dir / "weather_raw.json"

    with open(output_path, "wb") as file:
        file.write(downloaded_bytes)

    print(f"\nSaved downloaded JSON to {output_path}")


if __name__ == "__main__":
    main()