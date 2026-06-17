# Prefect Question 1
"""
A @task in Prefect is used for one step of work inside a pipeline.
Tasks are useful when you want Prefect to track, retry, log, or cache that step.

A @flow is the overall workflow that organizes and runs tasks.
The flow controls the order of execution and shows the full pipeline run in the Prefect UI.

If I had a helper function that only converts Celsius to Fahrenheit, I would not decorate it with @task.
Since it is a simple in-memory calculation with no I/O, retries, or separate tracking needed,
it can stay as a normal Python helper function.
"""

# Prefect Question 2
from prefect import task, flow, get_run_logger

@task(retries=3, retry_delay_seconds=30)


# Prefect Question 3
"""
If extract is Completed, transform is Failed, and load never ran, I would look at the failed
flow run in the Prefect UI. Then I would click into the transform task run.

I would expect to find the error message, traceback, task logs, start/end time, and the failed
state details. This would help me understand exactly why transform failed.

The load task never ran because it depended on transform completing successfully.
"""


# =========================
# Production Patterns
# =========================

# Production Question 1
"""
raise_for_status() checks the HTTP response and raises an exception if the API returned
an error status code, such as 400, 404, or 500.

This is better than only writing:

if response.status_code != 200:
    print("error")

because printing an error does not stop the pipeline. The task may still be marked as
successful even though the API call failed.

If the API returns a 500 error and I use raise_for_status(), the task fails clearly,
Prefect records the failure, and downstream tasks do not run.

If I only print("error"), the pipeline may continue to transform or load bad, missing,
or incomplete data.
"""

# Production Question 2
"""
overwrite=True protects me when I re-run the pipeline and upload to the same blob path:

final/{today}/weather_etl.json

If the first run crashed halfway through the transform step, I can fix the bug and re-run
from the beginning. When the pipeline reaches the upload step again, overwrite=True allows
the new corrected file to replace the existing file at that path.

Without overwrite=True, the upload could fail because a blob with the same name already
exists. That would force me to manually delete the old blob or change the file path before
the pipeline could finish successfully.
"""

# Production Question 3
@task
def load_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")