Cloud Concepts Question 1

The core economic model of cloud computing is pay-as-you-go pricing, where companies rent computing resources only when needed instead of buying and maintaining physical servers. This differs from owning your own servers because cloud computing avoids large upfront hardware costs and allows resources to scale more easily.

Cloud Concepts Question 2

Vertical scaling means increasing the power of a single machine, such as adding more RAM or a faster CPU. Horizontal scaling means adding more machines to distribute the workload.

You might choose vertical scaling when one application needs more processing power on a single machine, while horizontal scaling is better when tasks can be split across many systems.

The viral web app scenario uses horizontal scaling because more servers can be added to handle the large increase in users.
The model training scenario uses vertical scaling because the data scientist needs a stronger machine with more RAM and GPU power.
The data pipeline scenario uses horizontal scaling because the file processing work can be distributed across multiple machines.
Cloud Concepts Question 3
Classification
Gmail → SaaS because users simply use the software without managing infrastructure.
Azure Virtual Machines → IaaS because developers manage the operating system and applications while Azure provides the hardware.
Azure App Service → PaaS because Azure manages the infrastructure and runtime environment for applications.
AWS S3 → IaaS because it provides cloud storage infrastructure that developers use directly.
GitHub Codespaces → PaaS because it provides a managed development environment without needing to manage servers.
Snowflake → SaaS because it is a fully managed data platform accessed through the cloud.
Definitions
IaaS (Infrastructure as a Service) provides virtualized hardware resources like servers, networking, and storage. An example is Azure Virtual Machines, where the developer manages the operating system, applications, and configurations.
PaaS (Platform as a Service) provides a managed environment for building and deploying applications. An example is Azure App Service, where the developer manages the application code while the cloud provider manages the infrastructure.
SaaS (Software as a Service) provides fully managed software accessed through the internet. An example is Gmail, where the developer or user only uses the application while the provider manages everything else.
Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake provides built-in tools and infrastructure for handling data workloads, analytics, and scaling without requiring developers to manage servers directly. Compared to using Azure directly, you gain simplicity, automation, and faster setup, but you give up some flexibility and low-level control over the infrastructure.

Cloud Concepts Question 5

The cloud is probably not the right choice when an application requires extremely low latency with local hardware, or when strict legal or security requirements prevent data from being stored on third-party cloud systems.

Azure Basics
Azure Basics Question 1

An Azure subscription is the overall billing and account container that owns cloud resources, while a resource group is a logical collection of related resources within the subscription. The subscription belongs to CTD, while the resource groups are shared for course activities.

Azure Basics Question 2

Ephemeral means the Cloud Shell environment resets and temporary files are deleted after the session ends. The course setup uses attached persistent storage so files and configurations are saved between sessions.

Azure Basics Question 3

The SSH private key is secret and stays on your local machine, while the SSH public key is shared with remote systems. The public key gets uploaded to the remote system because it can only verify the matching private key and cannot be used to access the system by itself.

Azure Basics Question 4

{
  "environmentName": "AzureCloud",
  "homeTenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "isDefault": true,
  "managedByTenants": [],
  "name": "Azure Subscription",
  "state": "Enabled",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "user": {
    "name": "example@email.com",
    "type": "user"
  }
}

Adding --output table changes the display from detailed JSON format into a cleaner table format that is easier for humans to read quickly.