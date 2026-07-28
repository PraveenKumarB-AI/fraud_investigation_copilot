"""
rag/test_retrieval.py — MODULE 6: Verify the index retrieves sensibly
Picks a real flagged (illicit) transaction and retrieves its most similar
neighbors from the index, to sanity-check that retrieval works before an
agent depends on it.
Run:  python -m rag.test_retrieval
"""

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "rag/chroma_store"


def main():
    merged = pd.read_csv("data/processed/elliptic_merged.csv")
    illicit_sample = merged[merged["class"] == "1"].iloc[0]
    tx_id = str(illicit_sample["txId"])

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("transactions")

    print(f"Querying using illicit transaction {tx_id} as the reference case...\n")
    reference = collection.get(ids=[tx_id], include=["documents"])
    print(f"Reference: {reference['documents'][0]}\n")

    results = collection.query(
        query_texts=[reference["documents"][0]],
        n_results=6,
    )

    print("Top 6 most similar transactions in the index:")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['label']:>7}] {doc[:100]}...")


if __name__ == "__main__":
    main()
