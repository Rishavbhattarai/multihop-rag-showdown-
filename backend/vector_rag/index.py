"""Embed every corpus paragraph and store it in a persistent Chroma collection.

HotpotQA paragraphs are short (max ~325 words), so one paragraph = one chunk.
"""
import json

import chromadb

from backend import config, llm

COLLECTION = "corpus"
BATCH = 32


def get_collection():
    client = chromadb.PersistentClient(path=str(config.ARTIFACTS_DIR / "chroma"))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def main():
    docs = [json.loads(l) for l in open(config.CORPUS_PATH)]
    col = get_collection()
    existing = set(col.get()["ids"])
    todo = [d for d in docs if d["id"] not in existing]

    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        texts = [f"{d['title']}: {d['text']}" for d in batch]
        col.add(
            ids=[d["id"] for d in batch],
            embeddings=llm.embed_documents(texts),
            documents=[d["text"] for d in batch],
            metadatas=[{"title": d["title"]} for d in batch],
        )
    print(f"indexed {len(todo)} new, {col.count()} total")


if __name__ == "__main__":
    main()
