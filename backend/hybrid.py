"""Hybrid ablation: interleave vector and graph paragraphs (v1, g1, v2, g2, ...) under the same
TOP_K budget, paragraphs only (no triple list)."""
import time

from backend import answer as gen
from backend import config
from backend.graph_rag import query as graph
from backend.vector_rag import query as vector


def answer(question, k=config.TOP_K):
    t0 = time.perf_counter()
    v_chunks = vector.retrieve(question, k)
    g = graph.retrieve(question, k)
    merged, seen = [], set()
    for pair in zip(v_chunks, g["chunks"] + [None] * k):
        for c in pair:
            if c and c["id"] not in seen and len(merged) < k:
                seen.add(c["id"])
                merged.append({**c, "source": "vector" if c in v_chunks else "graph"})
    retrieval_s = time.perf_counter() - t0

    ans, reasoning, usage = gen.generate(question, graph.format_paragraphs(merged))
    return {
        "pipeline": "hybrid",
        "answer": ans,
        "reasoning": reasoning,
        "chunks": merged,
        "retrieved_ids": [c["id"] for c in merged],
        "usage": {
            "prompt_tokens": usage["prompt_tokens"] + g["link_usage"]["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"] + g["link_usage"]["completion_tokens"],
            "latency_s": usage["latency_s"],
            "retrieval_s": round(retrieval_s, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        },
    }
