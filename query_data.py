import argparse
from transformers import pipeline

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search(query_text, k=6)

    if not results:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text
    )
import argparse
from transformers import pipeline

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search(query_text, k=6)

    if not results:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text
    )

    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=120
    )

    response = generator(prompt)[0]["generated_text"]

    print("\n=== ANSWER ===\n")
    print(response.split("Answer:")[-1].strip())


if __name__ == "__main__":
    main()
