Part 1: Portal Walkthrough Summary

In my video, I showed my Azure portal logged into the Code the Dream tenant and navigated to my personal resource group. I pointed out the storage account inside the resource group and demonstrated that my Cloud Shell storage was persistent by showing that the test.txt file still existed in ~/clouddrive. I also showed the SSH key files in ~/.ssh and ran az group list --output table to display the available resource groups in the subscription.

Part 2: Cost Analysis
Scenario A -- Lightweight Compute

Scenario A used a Standard_B1s virtual machine running about 160 hours per month. This setup was relatively inexpensive because the VM only runs during working hours and uses minimal resources. This type of setup would work well for lightweight applications, testing environments, or small ETL jobs.

Scenario B -- Heavy Analytics Workload

Scenario B was much more expensive because it included a GPU-enabled VM running continuously for the entire month, along with Azure SQL Database resources and 1 TB of Blob Storage. GPU compute costs increase very quickly when the machine stays online 24/7. This scenario reflects the type of infrastructure needed for large analytics workloads, machine learning training, or advanced data processing pipelines.

Interesting Findings

One interesting thing I noticed while exploring the Pricing Calculator was how quickly costs increase when adding GPUs, larger databases, or higher storage amounts. Even small configuration changes had a major impact on the monthly estimate. It was also interesting to compare always-on infrastructure versus workloads that only run during business hours because the runtime hours alone created a huge price difference.

Write-Up Summary

The lightweight Scenario A cost only a few dollars per month, while the GPU-powered Scenario B cost several thousand dollars because of the powerful VM and continuous runtime. The difference between the two scenarios was much larger than I expected at first. I also found it interesting how much databases and storage services contributed to the overall cost estimate. When I ran the Python script in Cloud Shell, the calculated monthly costs matched the estimates from the Pricing Calculator closely, which confirmed that the hourly rate calculations were correct.