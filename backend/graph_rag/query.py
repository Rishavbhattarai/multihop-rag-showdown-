"""GraphRAG query: link question entities -> beam traversal -> graph-picked paragraphs -> answer.

Fairness: the answer model gets the same budget as vector RAG (TOP_K paragraphs),
chosen by the graph instead of by embedding similarity, plus the traversed triples.
"""
import json
import pickle
import sys
import time
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz, process

from backend import answer as gen
from backend import config, llm
from backend.graph_rag.build import GRAPH_PATH
from backend.graph_rag.resolve import normalize

FUZZY_LINK = 88
EMBED_LINK = 0.8
BEAM_EDGES = 12   # edges kept per hop
HUB_DEGREE = 80   # don't traverse *through* hubs like "United States"

ENTITY_PROMPT = (
    "List the proper names (specific people, places, organizations, creative works, events) in this question, "
    "one per line, exactly as written, nothing else. Do not list generic words like 'river', 'team', "
    "'region' or 'organization'.\nQuestion: {q}"
)


@lru_cache(maxsize=1)
def load():
    with open(GRAPH_PATH, "rb") as f:
        kg = pickle.load(f)
    kg["norm_names"] = [normalize(n) for n in kg["nodes"]]
    kg["edge_row"] = {e: i for i, e in enumerate(kg["edges"])}
    kg["corpus"] = {d["id"]: d for d in map(json.loads, open(config.CORPUS_PATH))}
    return kg


def link_entities(question, q_emb, kg):
    raw, usage = llm.chat(ENTITY_PROMPT.format(q=question))
    mentions = [m.strip(" -*•\"'") for m in raw.splitlines() if m.strip(" -*•\"'")]
    # proper names carry a capital letter; drops generic words the model lists anyway
    mentions = [m for m in mentions if any(c.isupper() for c in m)]
    seeds = {}
    for m in mentions:
        hit = process.extractOne(normalize(m), kg["norm_names"], scorer=fuzz.WRatio, score_cutoff=FUZZY_LINK)
        if hit:
            seeds[kg["nodes"][hit[2]]] = m
        m_emb = np.asarray(llm.embed_query(m), dtype=np.float32)
        sims = kg["node_emb"] @ (m_emb / np.linalg.norm(m_emb))
        best = int(sims.argmax())
        if sims[best] >= EMBED_LINK:
            seeds.setdefault(kg["nodes"][best], m)
    return mentions, seeds, usage


def traverse(seeds, q_emb, kg):
    g = kg["graph"]
    visited, frontier, kept, seen_facts = set(seeds), set(seeds), {}, set()
    for hop in range(1, config.MAX_HOPS + 1):
        cands = set()
        for n in frontier:
            if n not in seeds and g.degree(n) > HUB_DEGREE:
                continue
            cands.update((u, v, k) for u, v, k in g.out_edges(n, keys=True))
            cands.update((u, v, k) for u, v, k in g.in_edges(n, keys=True))
        cands = [e for e in cands if (e[0], g.edges[e]["rel"], e[1]) not in seen_facts]
        if not cands:
            break
        rows = [kg["edge_row"][e] for e in cands]
        scores = kg["edge_emb"][rows] @ q_emb
        frontier = set()
        taken = 0
        for i in np.argsort(-scores):
            u, v, k = cands[i]
            fact = (u, g.edges[u, v, k]["rel"], v)
            if fact in seen_facts:  # same fact from another paragraph
                continue
            seen_facts.add(fact)
            kept[(u, v, k)] = {"score": float(scores[i]), "hop": hop}
            frontier.update(n for n in (u, v) if n not in visited)
            taken += 1
            if taken == BEAM_EDGES:
                break
        visited |= frontier
    return kept


def answer(question, k=config.TOP_K):
    kg = load()
    g = kg["graph"]
    t0 = time.perf_counter()
    q_emb = np.asarray(llm.embed_query(question), dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb)
    mentions, seeds, link_usage = link_entities(question, q_emb, kg)
    link_s = time.perf_counter() - t0

    kept = traverse(seeds, q_emb, kg)
    # rank source paragraphs by their best edge in the traversed subgraph
    doc_score, doc_hop = {}, {}
    for (u, v, key), meta in kept.items():
        d = g.edges[u, v, key]["doc"]
        if meta["score"] > doc_score.get(d, -1.0):
            doc_score[d], doc_hop[d] = meta["score"], meta["hop"]
    by_score = sorted(doc_score, key=doc_score.get, reverse=True)
    # seed edges always score highest, so reserve a slot for each hop's best paragraph first;
    # otherwise the deeper hops a multi-hop question needs never make the cut
    doc_ids = []
    for hop in sorted(set(doc_hop.values())):
        best = next((d for d in by_score if doc_hop[d] == hop and d not in doc_ids), None)
        if best and len(doc_ids) < k:
            doc_ids.append(best)
    doc_ids += [d for d in by_score if d not in doc_ids][: k - len(doc_ids)]
    traverse_s = time.perf_counter() - t0 - link_s

    ranked = sorted(kept.items(), key=lambda kv: -kv[1]["score"])
    triples = [f"({u})-[{g.edges[u, v, key]['rel']}]->({v})" for (u, v, key), _ in ranked]
    paragraphs = [kg["corpus"][d] for d in doc_ids]
    if triples:
        context = ("Knowledge graph facts:\n" + "\n".join(triples) + "\n\nSource paragraphs:\n"
                   + "\n\n".join(f"[{p['id']}] {p['title']}: {p['text']}" for p in paragraphs))
    else:
        context = "(no graph facts found)"
    ans, reasoning, usage = gen.generate(question, context)

    node_hop = {s: 0 for s in seeds}
    for (u, v, _), meta in kept.items():
        for n in (u, v):
            node_hop[n] = min(node_hop.get(n, meta["hop"]), meta["hop"])
    return {
        "pipeline": "graph",
        "answer": ans,
        "reasoning": reasoning,
        "mentions": mentions,
        "seeds": [{"node": n, "mention": m} for n, m in seeds.items()],
        "triples": triples,
        "subgraph": {
            "nodes": [{"id": n, "hop": h, "seed": n in seeds} for n, h in node_hop.items()],
            "edges": [{"source": u, "target": v, "rel": g.edges[u, v, key]["rel"], "doc": g.edges[u, v, key]["doc"],
                       "score": round(m["score"], 4), "hop": m["hop"]} for (u, v, key), m in ranked],
        },
        "chunks": [{"id": p["id"], "title": p["title"], "text": p["text"], "score": round(doc_score[p["id"]], 4)}
                   for p in paragraphs],
        "retrieved_ids": doc_ids,
        "usage": {
            "prompt_tokens": usage["prompt_tokens"] + link_usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"] + link_usage["completion_tokens"],
            "latency_s": usage["latency_s"],
            "retrieval_s": round(link_s + traverse_s, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        },
    }


if __name__ == "__main__":
    out = answer(" ".join(sys.argv[1:]))
    print("seeds:", out["seeds"])
    print("\n".join(out["triples"][:15]))
    print("docs:", [c["title"] for c in out["chunks"]])
    print(out["reasoning"])
    print("->", out["answer"], out["usage"])
