import argparse
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

    # Same embeddings used during indexing
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load vector database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Retrieve relevant documents
    results = db.similarity_search(query_text, k=6)

    if not results:
        print("No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    sources = [doc.metadata.get("source", "Unknown") for doc in results]

    print("\nRetrieved Context:\n")
    print(context_text[:1000])  # print limited context

    print("\nSources:")
    for src in sources:
        print(src)

    # Format prompt (for LLM step later)
    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text
    )

    print("\nGenerated Prompt:\n")
    print(prompt)


if __name__ == "__main__":
    main()