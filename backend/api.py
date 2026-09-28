"""FastAPI backend for the side-by-side UI.

    uvicorn backend.api:app --reload --port 8000
"""
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend import config
from backend.eval.run_eval import RESULTS_PATH
from backend.graph_rag.query import answer as graph_answer
from backend.vector_rag.query import answer as vector_answer

app = FastAPI(title="GraphRAG vs Vector RAG")


def _questions():
    return {q["question"]: q for q in map(json.loads, open(config.QUESTIONS_PATH))}


def _results():
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None


class Query(BaseModel):
    question: str


@app.get("/api/demo-questions")
def demo_questions():
    return [
        {"question": q["question"], "type": q["type"], "answer": q["answer"]}
        for q in _questions().values() if q["split"] == "demo"
    ]


@app.get("/api/eval")
def eval_summary():
    res = _results()
    if res is None:
        raise HTTPException(404, "No eval results yet: run python -m backend.eval.run_eval")
    return res


@app.post("/api/query")
def query(body: Query):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    # Ollama serves one generation at a time on this machine, so run the two pipelines sequentially
    out = {"vector": vector_answer(question), "graph": graph_answer(question)}

    # Known dataset question: attach gold answer + supporting paragraphs so the UI can show
    # correctness from the metrics code, never a hardcoded badge
    q = _questions().get(question)
    if q:
        from backend.eval import metrics

        golds = [q["answer"], *q["answer_aliases"]]
        out["gold"] = {"answer": q["answer"], "aliases": q["answer_aliases"],
                       "supporting_ids": q["supporting_ids"], "type": q["type"]}
        for p in ("vector", "graph"):
            pred = out[p]["answer"]
            em = metrics.exact_match(pred, golds)
            out[p]["grade"] = {
                "em": em,
                "f1": round(metrics.f1(pred, golds), 3),
                "judge": bool(em) or metrics.llm_judge(question, pred, golds),
                "support_recall": round(metrics.support_recall(out[p]["retrieved_ids"], q["supporting_ids"]), 3),
            }
    return out
