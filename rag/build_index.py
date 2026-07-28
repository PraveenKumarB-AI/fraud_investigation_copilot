"""
rag/build_index.py — MODULE 6: Build the transaction investigation index
For every labeled transaction, creates a plain-English summary of its
feature profile and graph position, embeds it, and stores it in ChromaDB.
This lets an agent later retrieve similar past cases and graph-neighbor
context for any flagged transaction. Synthetic, generated summaries --
the Elliptic dataset has no real account names or case notes.
Run:  python -m rag.build_index
"""

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "data/processed"
CHROMA_DIR = "rag/chroma_store"
BATCH_SIZE = 500


def build_edge_lookup(edges):
    """For each txId, which transactions did it send to / receive from."""
    outgoing, incoming = {}, {}
    for a, b in zip(edges["txId1"], edges["txId2"]):
        outgoing.setdefault(a, []).append(b)
        incoming.setdefault(b, []).append(a)
    return outgoing, incoming


def summarize(row, outgoing, incoming, label_map):
    tx_id = row["txId"]
    label = label_map.get(tx_id, "unknown")
    n_out = len(outgoing.get(tx_id, []))
    n_in = len(incoming.get(tx_id, []))
    return (
        f"Transaction {tx_id} at time step {row['time_step']}. "
        f"Label: {label}. "
        f"RSI-style local features: feat_0={row['feat_0']:.2f}, feat_1={row['feat_1']:.2f}, "
        f"feat_2={row['feat_2']:.2f}. "
        f"Graph position: {n_out} outgoing connections (sent to), "
        f"{n_in} incoming connections (received from). "
        f"Total connectivity: {n_out + n_in} neighboring transactions."
    )


def main():
    print("1. Loading data...")
    merged = pd.read_csv(f"{PROCESSED_DIR}/elliptic_merged.csv")
    edges = pd.read_csv(f"{PROCESSED_DIR}/elliptic_edges.csv")
    outgoing, incoming = build_edge_lookup(edges)

    label_names = {"1": "illicit", "2": "licit", "unknown": "unknown"}
    label_map = dict(zip(merged["txId"], merged["class"].astype(str).map(label_names)))

    print("2. Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("3. Setting up ChromaDB (persistent, on disk)...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("transactions")

    print(f"4. Indexing {len(merged)} transactions in batches of {BATCH_SIZE}...")
    for start in range(0, len(merged), BATCH_SIZE):
        batch = merged.iloc[start:start + BATCH_SIZE]
        summaries = [summarize(row, outgoing, incoming, label_map) for _, row in batch.iterrows()]
        embeddings = embedder.encode(summaries, show_progress_bar=False).tolist()
        ids = [str(tx_id) for tx_id in batch["txId"]]
        metadatas = [
            {"txId": int(row["txId"]), "time_step": int(row["time_step"]),
             "label": label_map.get(row["txId"], "unknown")}
            for _, row in batch.iterrows()
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=summaries, metadatas=metadatas)
        if start % (BATCH_SIZE * 20) == 0:
            print(f"   Indexed {start + len(batch)}/{len(merged)}...")

    print(f"\n5. Done. Collection now has {collection.count()} transactions indexed.")
    print(f"   Stored at {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
