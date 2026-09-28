"""Answer and retrieval metrics.

EM/F1 follow the official SQuAD/HotpotQA normalization, taking the max over the
gold answer and its aliases. The LLM judge catches correct paraphrases EM misses
("Sully" vs "Chesley 'Sully' Sullenberger"); it only runs when EM fails.
"""
import re
import string
from collections import Counter

from backend import llm


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred, golds):
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1(pred, golds):
    return max(_f1(pred, g) for g in golds)


def support_recall(retrieved_ids, gold_ids):
    gold = set(gold_ids)
    return len(gold & set(retrieved_ids)) / len(gold)


JUDGE_SYSTEM = (
    "You grade answers to trivia questions. A prediction is correct only if it refers to the same "
    "entity, value or fact as one of the gold answers (paraphrases, abbreviations and extra harmless "
    "detail are fine). Wrong entity, partial answers to a different hop, or 'unknown' are incorrect. "
    'Reply with JSON: {"correct": true} or {"correct": false}.'
)


def llm_judge(question, pred, golds):
    if normalize_answer(pred) in ("", "unknown"):
        return False
    out, _ = llm.chat(
        f"Question: {question}\nGold answers: {golds}\nPrediction: {pred}",
        system=JUDGE_SYSTEM, json_mode=True, max_tokens=20,
    )
    return bool(out.get("correct"))
