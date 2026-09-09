#!/usr/bin/env python3
"""E4: attacker attribution / identity-rotation linkage.

Question: can the gateway's API-observable fingerprints (routing scores from
the frozen-backbone embedding head + stylometric features) re-link rotated
attacker identities?

Simulation: 20 attacker "campaigns", each = a base attack seed + K=5 rotated
variants where the attacker rotates (a) surface text (adversarial transform),
(b) API-visible identity (api_key id, user-agent). Each request carries a
campaign id (ground truth) unknown to the linker.

Linkage method:
  1. embedding fingerprint: 1024-d gate embedding from the NSFA-0.8B backbone
     (via /gate max_risk per-domain probability vector + text stylometry),
  2. k-away linkage: cluster requests whose fingerprint distance < tau;
     a campaign is "re-linked" if >=2 of its rotated variants land in one
     cluster with precision >= threshold.

We measure linkage precision @ k-away (k = # identities rotated).
Output: /datagroup/guard/eval/honeyroute/e4_attribution.json
"""
import json, os, random, math, urllib.request
import numpy as np

random.seed(42)
import os as _os
OUT = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
GATE = "http://127.0.0.1:8002/gate"

# import transforms from the existing harness
import sys
import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
from pipeline.attack_seeds import SEEDS
from pipeline.redteam_harness import (t_upper, t_leetspeak, t_space_pad,
                                      t_wordless, t_translate_hint, t_roleplay,
                                      t_academic)
TRANSFORMS = [t_upper, t_leetspeak, t_space_pad, t_wordless,
              t_translate_hint, t_roleplay, t_academic]

def score(text):
    body = json.dumps({"text": text, "task": "query"}).encode()
    req = urllib.request.Request(GATE, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def stylometry(text):
    s = text
    words = s.split()
    return [len(s), len(words), len(set(w.lower() for w in words)),
            s.count("?"), s.count("!"), sum(c.isupper() for c in s) / max(len(s), 1),
            sum(c.isdigit() for c in s) / max(len(s), 1),
            len(s) / max(len(words), 1)]

def fingerprint(text):
    out = score(text)
    probs = out["probabilities"]
    v = [probs.get(d, 0.0) for d in sorted(probs)]
    return np.array(v + stylometry(text) + [out["max_risk"]]), out

def main():
    # 20 campaigns, one base seed each, 5 rotated variants each
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    campaigns = all_seeds[:20]
    fps, labels, rotate_k = [], [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed]
        for k in range(1, 6):
            tf = TRANSFORMS[(ci + k) % len(TRANSFORMS)]
            texts.append(tf(seed))
        for k, t in enumerate(texts):
            f, _ = fingerprint(t)
            fps.append(f)
            labels.append(ci)
            rotate_k.append(k)
    X = np.array(fps)
    # z-normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # cosine distance matrix
    Nn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = 1 - Nn @ Nn.T
    # sweep tau; for each request link to nearest neighbor with different
    # rotation index; campaign re-linked if >=2 variants (incl. base) grouped
    results = {}
    for tau in [0.05, 0.1, 0.15, 0.2, 0.3]:
        # union-find over pairs with D < tau and different k
        parent = list(range(len(X)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if D[i, j] < tau:
                    ra, rb = find(i), find(j)
                    if ra != rb: parent[ra] = rb
        groups = {}
        for i in range(len(X)):
            groups.setdefault(find(i), []).append(i)
        # precision: fraction of groups that are campaign-pure (single label)
        pure = sum(1 for g in groups.values()
                   if len(set(labels[i] for i in g)) == 1)
        prec = pure / len(groups)
        # recall: fraction of campaigns fully re-linked into one group
        by_c = {}
        for i in range(len(X)):
            by_c.setdefault(labels[i], []).append(find(i))
        relink = sum(1 for c, rs in by_c.items() if len(set(rs)) == 1)
        rec = relink / len(by_c)
        results[f"tau={tau}"] = {"precision": round(prec, 4),
                                 "recall": round(rec, 4)}
    out = {"experiment": "E4_attribution",
           "n_campaigns": len(campaigns), "k_rotations": 5,
           "fingerprint": "7-domain risk probs + stylometry + max_risk (API-observable only)",
           "results": results,
           "n_requests": len(X)}
    json.dump(out, open(os.path.join(OUT, "e4_attribution.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("saved e4_attribution.json")

if __name__ == "__main__":
    main()
