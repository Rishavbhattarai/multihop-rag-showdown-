"""Answer generation shared by both pipelines, so they differ only in retrieval."""
import re

from backend import llm

SYSTEM = (
    "You answer multi-hop questions using only the provided context. "
    "First reason briefly (at most 3 short sentences), chaining facts from the context. "
    "Then end with a final line of the form 'Answer: <answer>' where <answer> is as short as possible "
    "(a name, number, date, or place), with no explanation. "
    "If the context does not contain the answer, end with 'Answer: unknown'."
)

_ANSWER_RE = re.compile(r"answer:\s*(.+)", re.IGNORECASE)


def generate(question, context):
    raw, usage = llm.chat(f"Context:\n{context}\n\nQuestion: {question}", system=SYSTEM)
    matches = _ANSWER_RE.findall(raw)
    ans = matches[-1].strip().rstrip(".") if matches else raw.strip().splitlines()[-1]
    return ans, raw, usage
