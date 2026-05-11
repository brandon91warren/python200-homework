from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- The Chat Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("\nAPI Q1 Response:")
print(response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens used:", response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

print("\nAPI Q2 Responses:")
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(response.choices[0].message.content)

# I noticed that temperature 0 is more consistent and predictable, while higher temperatures are more creative and varied.
# I would use temperature 0 if I needed a consistent, reproducible output.


# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

print("\nAPI Q3 Responses:")
for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}: {choice.message.content}")


# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("\nAPI Q4 Response:")
print(response.choices[0].message.content)

# The response was cut short because max_tokens limited how many tokens the model could generate.
# In a real application, max_tokens helps control response length, cost, and speed.


# --- System Messages and Personas ---

# System Q1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 Tutor Response:")
print(response.choices[0].message.content)

messages = [
    {"role": "system", "content": "You are a sarcastic but helpful senior software engineer. You explain things briefly and directly."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q1 Different Personality Response:")
print(response.choices[0].message.content)

# The explanation changed because the system message changed the model's tone, style, and personality.


# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 Response:")
print(response.choices[0].message.content)

# The model knows Jordan's name because the conversation history was included in the messages list.
# The API is stateless, so it only knows the name because we passed that earlier message again.


# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q1 Zero-Shot Response:")
print(response.choices[0].message.content)


# Prompt Q2 - One-Shot
prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q2 One-Shot Response:")
print(response.choices[0].message.content)

# Adding one example helped show the model the format I wanted and made the output more consistent.


# Prompt Q3 - Few-Shot
prompt = f"""
Classify the sentiment of each review as positive, negative, or mixed.

Examples:
Review: "The app is easy to use and works perfectly."
Sentiment: positive

Review: "The product arrived late and broke after one day."
Sentiment: negative

Review: "The price was good, but customer service was frustrating."
Sentiment: mixed

Reviews:
1. {reviews[0]}
2. {reviews[1]}
3. {reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q3 Few-Shot Response:")
print(response.choices[0].message.content)

# Zero-shot is good for simple tasks when no examples are needed.
# One-shot is useful when I want to show the model the exact format.
# Few-shot is best when I want stronger consistency or the task has more nuance.


# Prompt Q4 - Chain of Thought
prompt = """
Solve the problem step by step, then label the final answer clearly.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q4 Chain of Thought Response:")
print(response.choices[0].message.content)

# Asking the model to reason step by step can improve accuracy because it encourages the model to break the problem into smaller parts.


# Prompt Q5 - Structured Output
review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the review below.
Return only valid JSON with these keys:
sentiment, confidence, reason

The confidence should be a float from 0 to 1.
The reason should be one sentence.

Review: {review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

raw_response = response.choices[0].message.content

print("\nPrompt Q5 Raw JSON Response:")
print(raw_response)

try:
    parsed = json.loads(raw_response)
    print("Sentiment:", parsed["sentiment"])
    print("Confidence:", parsed["confidence"])
    print("Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("The response was not valid JSON.")
    print("Raw response for debugging:")
    print(raw_response)


# Prompt Q6 - Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 Instructions Response:")
print(response.choices[0].message.content)

regular_text = "The sunset over the mountains was beautiful. The sky turned orange and purple as the evening became quiet."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{regular_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("\nPrompt Q6 Non-Instructions Response:")
print(response.choices[0].message.content)

# Delimiters help separate the user's text from the instructions.
# This helps prevent confusion and reduces the chance that user text changes the task accidentally.


# --- Local Models with Ollama ---

# Ollama Q1

"""
Ollama output:
A large language model is an artificial intelligence system trained on huge amounts of text so it can understand and generate human-like language. It can answer questions, summarize information, write content, and help with many language-based tasks.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

print("\nOllama Q1 OpenAI Response:")
print(response.choices[0].message.content)

# The OpenAI response may be more polished and detailed, while the local Ollama response may be shorter or less refined.
# One advantage of running a model locally is that it can work without sending data to an outside API.
# One disadvantage is that local models may be slower or less powerful depending on the computer and model size.