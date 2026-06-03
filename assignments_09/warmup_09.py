# --- Azure Authentication ---

# Q1
"""
When I run a Python script locally that uses DefaultAzureCredential,
it relies on credentials already available on my computer.

Most commonly, it uses the Azure CLI login session.

Before running the script, I must run:

    az login

DefaultAzureCredential checks several credential sources in order.
One of those sources is AzureCliCredential, so if I am already logged in
through the Azure CLI, DefaultAzureCredential can find that login and use it.
"""


# Q2
"""
A deployed pipeline running on an Azure VM or container should not use az login
because az login is meant for an interactive human user on a local machine.

Instead, deployed Azure resources usually use a managed identity or service
principal.

The same Python code works without changes because DefaultAzureCredential
automatically checks the environment it is running in. Locally, it may use
Azure CLI credentials. In Azure, it can use managed identity credentials.
"""


# Q3
"""
If DefaultAzureCredential immediately gets an AuthenticationError, the two
most likely causes are:

1. I am not logged in locally.
   I would diagnose this by running:

       az account show

   If that fails, I need to run:

       az login

2. The logged-in account or deployed identity does not have permission.
   I would diagnose this by checking the error message for authorization
   details and confirming the identity has the correct role assignment,
   such as Storage Blob Data Reader or Storage Blob Data Contributor.
"""


# --- Blob Storage ---

# Q1
"""
Azure Blob Storage has a three-level hierarchy:

1. Storage account
2. Container
3. Blob

A storage account is like a filing cabinet.
A container is like a drawer inside the filing cabinet.
A blob is like an individual file inside that drawer.

For example:
storage account = my filing cabinet
container = a folder/drawer named raw-data
blob = one file named api_response_2026_06_02.json
"""


# Q2
"""
Scenario 1:
I would use Blob Storage because hourly raw JSON responses are files that I may
want to store cheaply and reprocess later.

Scenario 2:
I would use a relational database like Azure SQL because the analytics team
needs to query structured transaction data by date range and customer ID.

Scenario 3:
I would use Blob Storage because NumPy arrays are file-like objects that can be
saved between pipeline runs, such as .npy files.
"""


# Q3
def list_container(container_client):
    """
    Prints the name and size in bytes of every blob in the container.
    """
    blobs = container_client.list_blobs()

    for blob in blobs:
        print(f"{blob.name}: {blob.size} bytes")


# Q4
def upload_text(container_client, blob_name, text):
    """
    Encodes a Python string as UTF-8 and uploads it as a blob.
    Overwrites the blob if it already exists.
    """
    encoded_text = text.encode("utf-8")

    container_client.upload_blob(
        name=blob_name,
        data=encoded_text,
        overwrite=True
    )