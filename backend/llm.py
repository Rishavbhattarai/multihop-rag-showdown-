"""Thin Ollama wrapper: chat (optionally JSON mode) and embeddings, with token/latency logging."""
import json
import time

import ollama

from backend import config

_client = ollama.Client()


def chat(prompt, system=None, json_mode=False, temperature=0.0, max_tokens=768):
    # max_tokens caps runaway generations (7B models sometimes loop on list-like paragraphs)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    t0 = time.perf_counter()
    r = _client.chat(
        model=config.LLM_MODEL,
        messages=messages,
        format="json" if json_mode else None,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    content = r["message"]["content"]
    usage = {
        "prompt_tokens": r.get("prompt_eval_count", 0),
        "completion_tokens": r.get("eval_count", 0),
        "latency_s": round(time.perf_counter() - t0, 3),
    }
    return (json.loads(content) if json_mode else content.strip()), usage


# nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
def embed_documents(texts):
    return _client.embed(model=config.EMBED_MODEL, input=[f"search_document: {t}" for t in texts])["embeddings"]


def embed_query(text):
    return _client.embed(model=config.EMBED_MODEL, input=f"search_query: {text}")["embeddings"][0]
