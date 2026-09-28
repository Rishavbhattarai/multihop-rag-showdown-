"""Entity resolution: map every surface name from the triples to one canonical node.

Stages (all merges are logged to artifacts/merges.json for auditing):
  1. normalized exact match (case, punctuation, leading article, whitespace)
  2. fuzzy match: token_sort_ratio >= FUZZY_THRESHOLD and every word pair matches (_same_word), text names only
     (dates/numbers never fuzzy-merge: "1917" vs "1971" must stay apart)
Canonical name per cluster: a paragraph title if one is in the cluster, else the most frequent form.
"""
import json
import re
import string
from collections import Counter

import numpy as np
from rapidfuzz import fuzz, process

FUZZY_THRESHOLD = 92
TOKEN_THRESHOLD = 95
SUFFIXES = {"", "s", "es", "al"}
_PUNCT = str.maketrans("", "", string.punctuation.replace("(", "").replace(")", ""))


def normalize(name):
    s = name.lower().translate(_PUNCT)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return " ".join(s.split())


def _same_word(x, y):
    # Plural/possessive/adjectival suffix ("team"/"teams", "historic"/"historical"), or a near-identical
    # spelling that starts with the same letter. Rejects "i"/"ii" (World War I vs II), "claus"/"clause"
    # (Santa Claus vs The Santa Clause), "sejm"/"sejmik", "russia"/"prussia", "chopper"/"copper".
    short, long_ = sorted((x, y), key=len)
    if len(short) >= 3 and long_.startswith(short) and long_[len(short):] in SUFFIXES:
        return True
    return x[:1] == y[:1] and fuzz.ratio(x, y) >= TOKEN_THRESHOLD


def _is_texty(norm):
    return sum(c.isalpha() for c in norm) >= 4


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def resolve(records):
    """records: extraction cache rows. Returns (alias -> canonical name, merge log)."""
    counts = Counter()
    titles = set()
    for r in records:
        titles.add(r["title"])
        for t in r["triples"]:
            counts[t["s"]] += 1
            counts[t["o"]] += 1

    # stage 1: group surface forms by normalized key
    by_norm = {}
    for name in counts:
        by_norm.setdefault(normalize(name), []).append(name)

    uf = UnionFind()
    log = []
    for norm, names in by_norm.items():
        for n in names[1:]:
            uf.union(n, names[0])
        if len(names) > 1:
            log.append({"stage": "exact", "names": names})

    # stage 2: fuzzy over normalized keys (one representative per key)
    keys = [k for k in by_norm if _is_texty(k)]
    if keys:
        scores = process.cdist(keys, keys, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD, workers=-1)
        for i, j in zip(*np.nonzero(scores)):
            if i < j:
                a, b = keys[i], keys[j]
                # different numbers inside otherwise-similar names = different things ("Super Bowl 50" vs "51")
                if re.findall(r"\d+", a) != re.findall(r"\d+", b):
                    continue
                # whole-string similarity hides one-word swaps ("Cleveland Browns" vs "Cleveland Barons"),
                # so every word must also match its counterpart
                ta, tb = sorted(a.split()), sorted(b.split())
                if len(ta) != len(tb) or not all(_same_word(x, y) for x, y in zip(ta, tb)):
                    continue
                uf.union(by_norm[a][0], by_norm[b][0])
                log.append({"stage": "fuzzy", "score": float(scores[i, j]), "names": [by_norm[a][0], by_norm[b][0]]})

    clusters = {}
    for name in counts:
        clusters.setdefault(uf.find(name), []).append(name)

    alias = {}
    for members in clusters.values():
        in_titles = [m for m in members if m in titles]
        canonical = in_titles[0] if in_titles else max(members, key=lambda m: counts[m])
        for m in members:
            alias[m] = canonical
    return alias, log


if __name__ == "__main__":
    from backend import config
    from backend.graph_rag.extract import load_cache

    alias, log = resolve(list(load_cache().values()))
    json.dump(log, open(config.ARTIFACTS_DIR / "merges.json", "w"), indent=1)
    print(f"{len(alias)} surface names -> {len(set(alias.values()))} nodes, "
          f"{sum(l['stage'] == 'exact' for l in log)} exact / {sum(l['stage'] == 'fuzzy' for l in log)} fuzzy merges")
