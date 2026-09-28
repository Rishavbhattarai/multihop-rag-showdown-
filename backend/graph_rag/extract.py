"""LLM triple extraction per paragraph, cached to JSONL so runs are resumable.

Uses a compact "subject | relation | object" line format rather than JSON:
~40% fewer output tokens on a local 7B model, with similar triple quality.

    python -m backend.graph_rag.extract            # full corpus
    python -m backend.graph_rag.extract --limit 20 # quick quality check
"""
import argparse
import json
import time

from backend import config, llm

CACHE_PATH = config.ARTIFACTS_DIR / "extraction_cache.jsonl"

SYSTEM = """Extract a knowledge graph from a Wikipedia paragraph as triples, one per line:
subject | relation | object

Rules:
- Use full, specific names ("Ed Harris", not "he"). Resolve pronouns.
- The paragraph's main subject is given as TITLE; use exactly that name for it.
- relation: short lowercase verb phrase, 1-4 words ("spouse of", "directed by", "located in", "based on", "population").
- object: a single entity, place, date or number, never a clause.
- Extract every factual relation, including dates, numbers and locations. Output only triples.

Example
TITLE: Lostock Dam
TEXT: Lostock Dam is a dam on the Paterson River in New South Wales, Australia. It was completed in 1971.
Lostock Dam | located on | Paterson River
Lostock Dam | located in | New South Wales
Paterson River | located in | New South Wales
Lostock Dam | completed in | 1971"""


def parse_triples(raw):
    triples, seen = [], set()
    for line in raw.splitlines():
        parts = [p.strip().strip("*-•` ") for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            s, rel, o = parts
            t = (s, rel.lower().replace("_", " "), o)
            if t not in seen:  # a looping generation repeats the same lines
                seen.add(t)
                triples.append(dict(zip(("s", "rel", "o"), t)))
    return triples


def extract(doc):
    raw, usage = llm.chat(f"TITLE: {doc['title']}\nTEXT: {doc['text']}", system=SYSTEM)
    return {"id": doc["id"], "title": doc["title"], "triples": parse_triples(raw), "usage": usage}


def load_cache():
    if not CACHE_PATH.exists():
        return {}
    out = {}
    for line in open(CACHE_PATH):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # half-written last line while extraction is running
            continue
        out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    docs = [json.loads(l) for l in open(config.CORPUS_PATH)][: args.limit]
    done = load_cache()
    todo = [d for d in docs if d["id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to extract", flush=True)

    t0 = time.perf_counter()
    with open(CACHE_PATH, "a") as f:
        for n, doc in enumerate(todo, 1):
            try:
                rec = extract(doc)
            except Exception as e:  # transient Ollama errors; skipped docs retry on next run
                print(f"  {doc['id']} failed: {e}", flush=True)
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                rate = (time.perf_counter() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.1f}s/doc  eta {rate * (len(todo) - n) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
