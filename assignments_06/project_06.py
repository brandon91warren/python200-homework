
from dotenv import load_dotenv
from pathlib import Path
import os
import string

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# ============================================================
# Step 1: Setup
# ============================================================

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found. Check your .env file."

# Put the groundwork_docs folder inside assignments_06
docs_dir = Path("groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# ============================================================
# Step 2: Load the Documents
# ============================================================

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\n" + "=" * 60)
print("Step 2: Loaded Documents")
print("=" * 60)

print(f"Documents loaded: {len(documents)}")

for doc in documents:
    print("File:", doc.metadata.get("file_name"))


# ============================================================
# Step 3: Build the Index and Query Engine
# ============================================================

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


# ============================================================
# Step 4: Query the Assistant
# ============================================================

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n" + "=" * 60)
print("Step 4: Query the Assistant")
print("=" * 60)

for question in questions:
    print("\nQuestion:", question)

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    top_node = response.source_nodes[0]
    print("\nTop Retrieved Source Node:")
    print("Document name:", top_node.node.metadata.get("file_name"))
    print("Similarity score:", top_node.score)
    print("Chunk preview:", top_node.node.text[:200].replace("\n", " "))

# Reflection:
# The assistant sounded mostly confident and accurate because the questions were directly related to the Groundwork documents.
# The answers were strongest when the retrieved source document clearly matched the question.
# One thing that can be surprising is that semantic RAG may retrieve a related document even when the exact words from the question do not appear.


# ============================================================
# Step 5: Find a Failure
# ============================================================

print("\n" + "=" * 60)
print("Step 5: Find a Failure")
print("=" * 60)

failure_question = "Can I reserve the entire coffee shop for a wedding reception?"

print("\nFailure Question:", failure_question)

failure_response = query_engine.query(failure_question)

print("\nResponse:")
print(failure_response)

print("\nRetrieved Source Nodes:")

for i, node in enumerate(failure_response.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print("Document name:", node.node.metadata.get("file_name"))
    print("Similarity score:", node.score)
    print("Chunk preview:", node.node.text[:200].replace("\n", " "))

# Failure reflection:
# I asked about reserving the entire coffee shop for a wedding reception because that specific information may not be in the documents.
# The system might retrieve catering or events information, but that does not necessarily answer the full question.
# If the model gives a confident answer without the documents clearly supporting it, that shows why AI-generated answers should be checked against sources.
# To improve the system, I would add stricter instructions telling the model to say when information is not available in the documents.


# ============================================================
# Optional Extension A: Side-by-Side Keyword vs Semantic RAG
# ============================================================

print("\n" + "=" * 60)
print("Optional Extension A: Keyword RAG vs Semantic RAG")
print("=" * 60)


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

    scores.sort(reverse=True)

    best = next(((name, content, score) for score, name, content in scores if score > 0), None)

    if best:
        return best

    return ("None found", "No relevant content.", 0)


keyword_documents = {
    f.name: f.read_text()
    for f in docs_dir.glob("*.txt")
}

for question in questions:
    print("\nQuestion:", question)

    keyword_name, keyword_content, keyword_score = simple_keyword_retrieval(
        question,
        keyword_documents,
        verbose=False
    )

    semantic_response = query_engine.query(question)
    semantic_top_node = semantic_response.source_nodes[0]

    print("\nKeyword RAG Result:")
    print("Document:", keyword_name)
    print("Score:", keyword_score)
    print("Preview:", keyword_content[:200].replace("\n", " "))

    print("\nLlamaIndex Semantic RAG Result:")
    print("Answer:", semantic_response)
    print("Top document:", semantic_top_node.node.metadata.get("file_name"))
    print("Similarity score:", semantic_top_node.score)

# Extension A reflection:
# Keyword RAG worked best when the question used the same words that appeared in the documents.
# Semantic RAG usually gave better answers because it understood meaning, not just exact word overlap.
# Keyword RAG can fail when the user uses synonyms, plural words, or a phrase that is related but not exact.


# ============================================================
# Optional Extension C: Add a New Document
# ============================================================

print("\n" + "=" * 60)
print("Optional Extension C: Add a New Document")
print("=" * 60)

new_doc_path = docs_dir / "seasonal_specials.txt"

new_doc_text = """
Groundwork Coffee Co. Seasonal Specials

This fall, Groundwork Coffee Co. is offering a Maple Cinnamon Latte, a Pumpkin Cold Brew, and an Apple Chai Tea.
All seasonal drinks can be made with oat milk, almond milk, or regular dairy milk.

Seasonal pastries include pumpkin muffins, apple turnovers, and cinnamon scones.
The seasonal menu is available from September 1 through November 30.
"""

new_doc_path.write_text(new_doc_text)

print(f"Added new document: {new_doc_path.name}")

updated_documents = SimpleDirectoryReader(str(docs_dir)).load_data()
updated_index = VectorStoreIndex.from_documents(updated_documents)
updated_query_engine = updated_index.as_query_engine(similarity_top_k=3)

new_question = "What seasonal drinks does Groundwork offer in the fall?"

print("\nNew Document Test Question:", new_question)

new_response = updated_query_engine.query(new_question)

print("\nAnswer:")
print(new_response)

print("\nRetrieved Source Nodes:")
for i, node in enumerate(new_response.source_nodes, start=1):
    print(f"\nSource Node {i}")
    print("Document name:", node.node.metadata.get("file_name"))
    print("Similarity score:", node.score)
    print("Chunk preview:", node.node.text[:200].replace("\n", " "))

# Extension C reflection:
# I added a seasonal_specials.txt document with fall drinks, seasonal pastries, milk options, and availability dates.
# I tested it by asking what seasonal drinks Groundwork offers in the fall.
# This demonstrates an advantage of RAG over fine-tuning because I only needed to add a new document and rebuild the index.
# I did not need to retrain or fine-tune the model.


# ============================================================
# Optional Extension D: Persistent Vector Store with pgvector
# ============================================================

# I am skipping the Docker/pgvector stretch extension unless Docker is installed and running.
# The main practical advantage of persisting embeddings is that the system does not need to re-embed every document every time it runs.
# In production, I would definitely want this for a large document collection, a customer support assistant, or any app that needs faster startup and repeated querying.
# The in-memory store is simpler for homework and testing, but a persistent vector database is better for real applications.


# ============================================================
# Step 6: Final Reflection
# ============================================================

# The LlamaIndex implementation only took a few main lines to load documents, build the index, and create a query engine.
# That shows the value of using a framework because it handles chunking, embedding, indexing, and retrieval without writing all of that logic manually.
#
# A different use case would be an HR assistant that answers employee questions from company policies, benefits documents, onboarding guides, and PTO rules.
# This would help employees get answers quickly without needing to search through many files or wait for HR.
#
# One failure mode RAG cannot fully prevent is the model misinterpreting retrieved context.
# Even when retrieval works correctly, the model can still summarize badly, overstate an answer, or sound too confident about incomplete information.