# --- LLMs as Transform ---

# Q1

# 1. Deterministic code: parsing "Jan 5th, 2024" into "2024-01-05" follows a predictable date-formatting rule.
# 2. LLM: classifying "my card was charged twice" requires understanding meaning, and this should be labeled billing.
# 3. Deterministic code: calculating an average is exact math and does not need an LLM.
# 4. LLM: extracting "Acme Corp" from a messy freeform job title requires interpreting unstructured text.
# 5. Deterministic code: checking whether a review has more than 100 words is a simple word count.

# Q2

"""
Problem:
The prompt "Summarize this product review in a few sentences" creates inconsistent output.
Downstream, this is hard to parse because the model might return different sentence lengths,
formats, tones, or extra commentary. That makes it harder to store reliably in a database.

Better prompt:
You are transforming product reviews for a data pipeline.
Return only valid JSON using this exact schema:

{
  "summary": "one concise summary sentence",
  "sentiment": "positive | negative | neutral",
  "main_issue": "short phrase or null"
}

Do not include markdown, explanations, or extra text.
"""

# Q3

"""
50,000 calls x 1 second each = 50,000 seconds.

50,000 seconds / 60 = 833.33 minutes.
833.33 minutes / 60 = about 13.9 hours.

One practical strategy is to process records concurrently in batches,
while respecting rate limits and adding retry logic for failed calls.
"""


# --- Azure OpenAI ---

# Q1

"""
Two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly:

1. Enterprise governance and compliance:
   Azure OpenAI can fit into an organization's existing Azure security, compliance,
   identity, networking, and monitoring setup.

2. Azure ecosystem integration:
   Companies already using Azure may want OpenAI models connected with Azure services
   like Azure Key Vault, Azure Storage, Azure AI Search, and private networking.
"""

# Q2

"""
Three Azure-specific parameters used when initializing AzureOpenAI are:

1. azure_endpoint:
   The URL for the Azure OpenAI resource, such as:
   https://your-resource-name.openai.azure.com

2. api_version:
   The Azure OpenAI API version your code should call, such as:
   2024-10-21

3. azure_deployment:
   The deployment name for the model in Azure.
   This is the custom name created when the model was deployed in Azure AI Foundry.
"""

# Q3

"""
When using AzureOpenAI, the model parameter does not use the base model name
like "gpt-4o-mini" unless that is also the deployment name.

Instead, it takes the Azure deployment name.

You find the correct value in Azure AI Foundry / Azure OpenAI Studio under Deployments.
Use the deployment name shown there.
"""