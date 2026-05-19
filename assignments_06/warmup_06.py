# warmup_06.py

from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ============================================================
# --- RAG Concepts ---
# ============================================================

# Concepts Q1
"""
Scenario A: RAG
The legal team should use RAG because the assistant needs to answer questions from a large internal policy library that changes every quarter.
RAG works well because it can retrieve the most current information from the documents instead of relying only on the model's training.

Scenario B: Fine-tuning
The startup should use fine-tuning because they want the model to consistently write in a very specific brand voice.
Since they already have 3,000 writing examples, fine-tuning can help the model learn that style better.

Scenario C: Prompt engineering
The data analyst should use prompt engineering because she only needs help with one short two-page report.
She can paste the report into the prompt and ask questions directly without needing a full RAG system.
"""

# Concepts Q2
"""
A confidently wrong answer is more harmful than an answer that says "I am not sure" because people are more likely to trust it.
For example, if an AI confidently gives the wrong medical dosage, someone could follow the advice and get seriously hurt.

The tone matters because confident language makes the answer sound reliable, even when the content is wrong.
If the model admits uncertainty, the user is more likely to double-check the information.
"""

# Concepts Q3
"""
Correct RAG pipeline order:

1. Extract text from source documents
   The system reads and pulls usable text from PDFs, websites, files, or other documents.

2. Split text into chunks
   Long documents are broken into smaller pieces so the system can search them more easily.

3. Convert text chunks into embeddings
   Each chunk is turned into a vector that represents its meaning.

4. Receive the user's query
   The user asks a question.

5. Embed the user's query
   The question is also converted into a vector so it can be compared to the document chunks.

6. Retrieve the most relevant chunks
   The system finds the chunks that are most similar to the user's query.

7. Inject retrieved chunks into the prompt
   The relevant chunks are added to the prompt as context for the LLM.

8. Generate a response from the LLM
   The model answers the question using the retrieved context.
"""


# ============================================================
# --- Keyword RAG ---
# ============================================================

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }

    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []

    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)

    best = next(((name, content) for score, name, content in scores if score > 0), None)

    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


# Keyword Question 1
print("\n" + "=" * 60)
print("Keyword Question 1")
print("=" * 60)

query = "What are your hours on the weekend?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# Keyword retrieval selected loyalty.txt, even though hours.txt is the correct answer. This happened because the query used "weekend" but the document says "weekends", and keyword retrieval only matches exact words. It also matched the word "your" in loyalty.txt and hiring.txt, which is not a meaningful word for the question.


# Keyword Question 2
print("\n" + "=" * 60)
print("Keyword Question 2")
print("=" * 60)

query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The selected document may be None found or may not retrieve the best answer because the menu mentions coffee drinks but does not use the word caffeine.
# Keyword RAG does not handle this well because it depends on exact word overlap.
# Semantic retrieval would do better because it can understand that espresso, lattes, cappuccinos, and cold brew are related to caffeine.


# Keyword Question 3
print("\n" + "=" * 60)
print("Keyword Question 3")
print("=" * 60)

# Prediction:
# I think loyalty.txt will be selected because "sign up for rewards" is similar to joining a loyalty program.
# However, keyword retrieval may struggle because the exact words "sign", "rewards", or "up" do not appear in the loyalty document.

query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# My prediction may not be correct because keyword matching only looks for exact overlapping words.
# If "rewards" does not appear in loyalty.txt, the function may fail to select the best document even though loyalty.txt is the correct semantic match.


# ============================================================
# --- Semantic RAG Concepts ---
# ============================================================

# Semantic Question 1
"""
A vector embedding is a list of numbers that represents the meaning of text.
It allows a computer to compare ideas instead of only comparing exact words.

A chunk with a cosine similarity score of 0.85 is more relevant than one with a score of 0.30.
The 0.85 score means the chunk is much closer in meaning to the query.

Semantic search can find relevant chunks even when the exact words are different because it compares meaning.
For example, it can connect "rewards" with "loyalty program" even though the words are not the same.
"""

# Semantic Question 2
"""
| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Vector meaning / embedding similarity |
| What is retrieved?         | Full document                     | Most relevant chunks |
| Can it handle synonyms?    | No                                | Yes |
| Storage format             | Plain text dictionary             | Vector store / index |
| Relevance score            | Number of overlapping keywords    | Cosine similarity or another vector similarity score |
"""


# ============================================================
# --- LlamaIndex ---
# ============================================================

print("\n" + "=" * 60)
print("LlamaIndex Section")
print("=" * 60)

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.llms.openai import OpenAI
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    pdf_path = "brightleaf_pdfs"

    print(f"\nLoading documents from: {pdf_path}")
    docs = SimpleDirectoryReader(pdf_path).load_data()

    print(f"Loaded {len(docs)} documents.")

    index = VectorStoreIndex.from_documents(docs)

    # ------------------------------------------------------------
    # LlamaIndex Question 1
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LlamaIndex Question 1")
    print("=" * 60)

    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]

    query_engine = index.as_query_engine(similarity_top_k=3)

    for question in questions:
        print("\nQuestion:", question)
        response = query_engine.query(question)

        print("\nAnswer:")
        print(response)

        print("\nRetrieved source nodes:")
        for i, node in enumerate(response.source_nodes, start=1):
            print(f"\nSource Node {i}")
            print("Similarity score:", node.score)
            print("Chunk preview:", node.node.text[:150].replace("\n", " "))

    # Query 1 comment:
    # The retrieved chunks should look relevant if they mention employee benefits, insurance, time off, or similar HR information.
    # The response may sound confident if the retrieved chunks directly answer the question.
    # If it says "based on the context," that means the model is grounding the answer in the retrieved text.

    # Query 2 comment:
    # The retrieved chunks should look relevant if they mention security, passwords, devices, access control, or data protection.
    # If unrelated HR or company policy chunks appear, that would be unexpected and could weaken the response.


    # ------------------------------------------------------------
    # LlamaIndex Question 2
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LlamaIndex Question 2")
    print("=" * 60)

    question = "What employee benefits does BrightLeaf offer?"

    for top_k in [1, 5]:
        print(f"\nRunning query with similarity_top_k={top_k}")

        query_engine_k = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine_k.query(question)

        print("\nResponse:")
        print(response)

        print("\nSource node scores:")
        for i, node in enumerate(response.source_nodes, start=1):
            print(f"Source Node {i} score:", node.score)
            print("Chunk preview:", node.node.text[:150].replace("\n", " "))

    # More context can help when the extra chunks are relevant, but more retrieved context is not always better.
    # If similarity_top_k=5 includes unrelated chunks, the answer may become less focused or include unnecessary information.


    # ------------------------------------------------------------
    # LlamaIndex Question 3
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LlamaIndex Question 3")
    print("=" * 60)

    struggle_query = "What is BrightLeaf's long-term plan for expanding internationally?"

    print("\nStruggle Query:", struggle_query)

    response = query_engine.query(struggle_query)

    print("\nResponse:")
    print(response)

    print("\nRetrieved chunks:")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nSource Node {i}")
        print("Similarity score:", node.score)
        print("Chunk preview:", node.node.text[:300].replace("\n", " "))

    # I expected the pipeline to struggle because international expansion may not be covered in the BrightLeaf documents.
    # If the response hedges or says the information is not in the context, that is a good behavior.
    # If it gives a confident answer without support, the system may need stronger instructions to only answer from retrieved context.


    # ------------------------------------------------------------
    # LlamaIndex Question 4
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("LlamaIndex Question 4")
    print("=" * 60)

    judge_llm = OpenAI(model="gpt-4o-mini")

    faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

    q1 = "What employee benefits does BrightLeaf offer?"
    response1 = query_engine.query(q1)

    print("\nEvaluation Query 1:", q1)

    faithfulness_result1 = faithfulness_evaluator.evaluate_response(response=response1)
    relevancy_result1 = relevancy_evaluator.evaluate_response(query=q1, response=response1)

    print("Faithfulness passing:", faithfulness_result1.passing)
    print("Faithfulness score:", faithfulness_result1.score)
    print("Faithfulness feedback:", faithfulness_result1.feedback)

    print("Relevancy passing:", relevancy_result1.passing)
    print("Relevancy score:", relevancy_result1.score)
    print("Relevancy feedback:", relevancy_result1.feedback)

    q2 = "What is BrightLeaf's policy on employee housing in Tokyo?"
    response2 = query_engine.query(q2)

    print("\nEvaluation Query 2:", q2)

    faithfulness_result2 = faithfulness_evaluator.evaluate_response(response=response2)
    relevancy_result2 = relevancy_evaluator.evaluate_response(query=q2, response=response2)

    print("Faithfulness passing:", faithfulness_result2.passing)
    print("Faithfulness score:", faithfulness_result2.score)
    print("Faithfulness feedback:", faithfulness_result2.feedback)

    print("Relevancy passing:", relevancy_result2.passing)
    print("Relevancy score:", relevancy_result2.score)
    print("Relevancy feedback:", relevancy_result2.feedback)

    # Evaluation comments:
    # A faithfulness score of 1.0 means the answer is fully supported by the retrieved context.
    # A faithfulness score of 0.0 would mean the answer is not supported and may be hallucinated.
    #
    # Relevancy measures whether the response actually answers the user's question.
    # This is different from faithfulness because an answer can be supported by the context but still not answer the question well.
    #
    # The scores should change between the two queries because the first query is likely answered in the documents,
    # while the second query asks about information that probably does not exist in the BrightLeaf PDFs.
    #
    # LLM-as-a-judge means using another language model to evaluate the quality of a response.
    # It is useful for RAG evaluation because answers are often written in natural language, so there may not be one exact correct string to compare against.

except Exception as e:
    print("\nThe LlamaIndex section could not run.")
    print("Error:", e)
    print("\nCheck that:")
    print("1. Your packages are installed.")
    print("2. Your .env file contains OPENAI_API_KEY.")
    print("3. Your BrightLeaf PDF path is correct.")