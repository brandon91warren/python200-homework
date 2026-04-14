import numpy as np
import pandas as pd
from prefect import flow, task

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow():
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary

if __name__ == "__main__":
    pipeline_flow()

# This pipeline is very small, so using Prefect adds extra setup and overhead
# compared with just using plain Python functions. For a handful of numbers and
# three simple steps, regular Python is easier to read and faster to run.

# Prefect becomes more useful in larger real-world pipelines, such as scheduled
# ETL jobs, workflows with many dependent steps, pipelines that need retries,
# logging, monitoring, alerts, cloud execution, or pulling data from APIs and databases.
# Even if each step stays simple, Prefect helps manage the full workflow reliably.