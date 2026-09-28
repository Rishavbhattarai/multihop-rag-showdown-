"""Build the knowledge graph from cached triples and pre-embed its nodes and edges.

Outputs artifacts/graph.pkl containing the NetworkX MultiDiGraph plus embedding
matrices, so query time only needs one embedding call (the question).
"""
import json
import pickle

import networkx as nx
import numpy as np

from backend import config, llm
from backend.graph_rag.extract import load_cache
from backend.graph_rag.resolve import resolve

GRAPH_PATH = config.ARTIFACTS_DIR / "graph.pkl"
BATCH = 128


def _embed(texts):
    vecs = []
    for i in range(0, len(texts), BATCH):
        vecs.extend(llm.embed_documents(texts[i : i + BATCH]))
    m = np.asarray(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build():
    records = list(load_cache().values())
    alias, log = resolve(records)
    json.dump(log, open(config.ARTIFACTS_DIR / "merges.json", "w"), indent=1)

    g = nx.MultiDiGraph()
    edge_list = []  # (u, v, key) in a stable order matching edge_emb rows
    for r in records:
        for t in r["triples"]:
            u, v = alias[t["s"]], alias[t["o"]]
            if u == v:
                continue
            for n, surface in ((u, t["s"]), (v, t["o"])):
                if n not in g:
                    g.add_node(n, aliases=set(), docs=set())
                g.nodes[n]["aliases"].add(surface)
                g.nodes[n]["docs"].add(r["id"])
            k = g.add_edge(u, v, rel=t["rel"], doc=r["id"])
            edge_list.append((u, v, k))

    nodes = list(g.nodes)
    node_emb = _embed(nodes)
    edge_emb = _embed([f"{u} {g.edges[u, v, k]['rel']} {v}" for u, v, k in edge_list])

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump({"graph": g, "nodes": nodes, "node_emb": node_emb,
                     "edges": edge_list, "edge_emb": edge_emb}, f)

    degrees = sorted((d for _, d in g.degree()), reverse=True)
    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges from {len(records)} paragraphs; "
          f"{len(alias) - len(set(alias.values()))} names merged; top degrees {degrees[:5]}")


if __name__ == "__main__":
    build()
