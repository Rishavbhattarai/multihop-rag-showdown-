"""Classic vector RAG: top-k cosine retrieval -> shared answer generation."""
import sys
import time

from backend import answer as gen
from backend import config, llm
from backend.vector_rag.index import get_collection


def retrieve(question, k=config.TOP_K):
    res = get_collection().query(query_embeddings=[llm.embed_query(question)], n_results=k)
    return [
        {"id": i, "title": m["title"], "text": d, "score": round(1 - dist, 4)}
        for i, m, d, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]


def answer(question, k=config.TOP_K):
    t0 = time.perf_counter()
    chunks = retrieve(question, k)
    retrieval_s = time.perf_counter() - t0

    context = "\n\n".join(f"[{c['id']}] {c['title']}: {c['text']}" for c in chunks)
    ans, reasoning, usage = gen.generate(question, context)
    return {
        "pipeline": "vector",
        "answer": ans,
        "reasoning": reasoning,
        "chunks": chunks,
        "retrieved_ids": [c["id"] for c in chunks],
        "usage": {**usage, "retrieval_s": round(retrieval_s, 3),
                  "total_s": round(time.perf_counter() - t0, 3)},
    }


if __name__ == "__main__":
    out = answer(" ".join(sys.argv[1:]))
    for c in out["chunks"]:
        print(f"{c['score']:.3f}  {c['id']}  {c['title']}")
    print(out["reasoning"])
    print("->", out["answer"], out["usage"])
