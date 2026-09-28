"""Run both pipelines on every question and write artifacts/results.json.

Per-question runs are cached in artifacts/eval_runs.jsonl, so an interrupted
eval resumes where it stopped. Use --fresh after rebuilding the graph or index.

    python -m backend.eval.run_eval [--fresh] [--limit N]
"""
import argparse
import json
import statistics
import time

from backend import config
from backend.eval import metrics
from backend import hybrid
from backend.graph_rag.query import answer as graph_answer
from backend.vector_rag.query import answer as vector_answer

RUNS_PATH = config.ARTIFACTS_DIR / "eval_runs.jsonl"
RESULTS_PATH = config.ARTIFACTS_DIR / "results.json"
PIPELINES = {
    "vector": vector_answer,
    "graph": graph_answer,
    # ablations: same 5-paragraph budget, no triple list
    "graph_paras": lambda q: graph_answer(q, with_triples=False),
    "hybrid": hybrid.answer,
}


def score(q, out):
    golds = [q["answer"], *q["answer_aliases"]]
    em = metrics.exact_match(out["answer"], golds)
    return {
        "qid": q["id"],
        "pipeline": out["pipeline"],
        "type": q["type"],
        "split": q["split"],
        "question": q["question"],
        "gold": q["answer"],
        "pred": out["answer"],
        "em": em,
        "f1": round(metrics.f1(out["answer"], golds), 4),
        "judge": bool(em) or metrics.llm_judge(q["question"], out["answer"], golds),
        "support_recall": round(metrics.support_recall(out["retrieved_ids"], q["supporting_ids"]), 4),
        "retrieved_ids": out["retrieved_ids"],
        "total_s": out["usage"]["total_s"],
        "prompt_tokens": out["usage"]["prompt_tokens"],
        "completion_tokens": out["usage"]["completion_tokens"],
    }


def load_runs():
    if not RUNS_PATH.exists():
        return {}
    return {(r["pipeline"], r["qid"]): r for r in map(json.loads, open(RUNS_PATH))}


def aggregate(rows):
    lat = sorted(r["total_s"] for r in rows)
    return {
        "n": len(rows),
        "em": round(statistics.mean(r["em"] for r in rows), 3),
        "f1": round(statistics.mean(r["f1"] for r in rows), 3),
        "judge_acc": round(statistics.mean(r["judge"] for r in rows), 3),
        "support_recall": round(statistics.mean(r["support_recall"] for r in rows), 3),
        "full_support": round(statistics.mean(r["support_recall"] == 1 for r in rows), 3),
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
        "tokens_per_query": round(statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows)),
    }


def summarize(runs, questions):
    eval_ids = {q["id"] for q in questions if q["split"] == "eval"}
    summary = {}
    for p in PIPELINES:
        rows = [r for (pipe, qid), r in runs.items() if pipe == p and qid in eval_ids]
        if not rows:
            continue
        summary[p] = {"all": aggregate(rows)}
        for t in sorted({r["type"] for r in rows}):
            summary[p][t] = aggregate([r for r in rows if r["type"] == t])

    # head-to-head on the judge metric: where does each pipeline win outright?
    wins = {"vector_only": [], "graph_only": [], "both": [], "neither": []}
    for qid in sorted(eval_ids):
        v, g = runs.get(("vector", qid)), runs.get(("graph", qid))
        if not (v and g):
            continue
        key = {(True, False): "vector_only", (False, True): "graph_only",
               (True, True): "both", (False, False): "neither"}[(v["judge"], g["judge"])]
        wins[key].append(qid)
    return summary, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.fresh and RUNS_PATH.exists():
        RUNS_PATH.unlink()

    questions = [json.loads(l) for l in open(config.QUESTIONS_PATH)][: args.limit]
    runs = load_runs()
    t0 = time.perf_counter()
    with open(RUNS_PATH, "a") as f:
        for n, q in enumerate(questions, 1):
            for name, fn in PIPELINES.items():
                if (name, q["id"]) in runs:
                    continue
                row = score(q, fn(q["question"]))
                runs[(name, q["id"])] = row
                f.write(json.dumps(row) + "\n")
                f.flush()
            marks = " ".join(f"{p}={'✓' if runs[(p, q['id'])]['judge'] else '✗'}" for p in PIPELINES)
            print(f"{n:2d}/{len(questions)} [{q['type']}] {marks}  gold={q['answer']!r}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    summary, wins = summarize(runs, questions)
    RESULTS_PATH.write_text(json.dumps({
        "config": {"llm": config.LLM_MODEL, "embed": config.EMBED_MODEL, "top_k": config.TOP_K,
                   "max_hops": config.MAX_HOPS, "dataset": "MuSiQue (answerable, validation)",
                   "n_eval": sum(q["split"] == "eval" for q in questions)},
        "summary": summary,
        "head_to_head": {k: len(v) for k, v in wins.items()},
        "wins": wins,
        "runs": sorted(runs.values(), key=lambda r: (r["qid"], r["pipeline"])),
    }, indent=1))

    for p, s in summary.items():
        print(f"\n{p.upper()}")
        for t, m in s.items():
            print(f"  {t:5s} n={m['n']:2d}  EM {m['em']:.2f}  F1 {m['f1']:.2f}  judge {m['judge_acc']:.2f}  "
                  f"recall {m['support_recall']:.2f}  full-support {m['full_support']:.2f}  "
                  f"p50 {m['latency_p50_s']}s  tok {m['tokens_per_query']}")
    print("\nhead-to-head (judge):", {k: len(v) for k, v in wins.items()})


if __name__ == "__main__":
    main()
