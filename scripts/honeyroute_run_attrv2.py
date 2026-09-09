#!/usr/bin/env python3
"""Attribution v2: multi-view fingerprint fusion for identity-rotation linkage.

Views (all API-observable):
  V1: frozen-backbone 1024-d embedding (router feature)
  V2: stylometry (14 dims: length, charset stats, punctuation, structure)
  V3: per-domain risk probabilities from the 5 query heads (5 dims)
      — "what kind of attack" semantics

Fusion: concat z-normalized views, unit-norm; same calibrated-threshold
protocol as run_e4c (20 calib campaigns, 30 test campaigns, k=1..5).
Reports pairwise P/R/F1 and per-k recall; compares against V1-only.
Output: /datagroup/guard/eval/honeyroute/attribution_fusion_v2.json
"""
import json, os, random, sys
import numpy as np
import torch

import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
random.seed(42); torch.manual_seed(42)
import os as _os
OUT = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")

from pipeline.attack_seeds import SEEDS
from pipeline.redteam_harness import (t_upper, t_leetspeak, t_space_pad,
                                      t_wordless, t_translate_hint, t_roleplay,
                                      t_academic)
from train_head import preprocess_sample
from transformers import AutoTokenizer, AutoModel

TRANSFORMS = [t_upper, t_leetspeak, t_space_pad, t_wordless,
              t_translate_hint, t_roleplay, t_academic]
BB = os.environ.get("BACKBONE_DIR", "/datagroup/guard/guard-backbone-0.8B")
tok = AutoTokenizer.from_pretrained(BB, trust_remote_code=True)
model = AutoModel.from_pretrained(BB, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to("cuda:0").eval()
MAXT = 512

@torch.no_grad()
def embed(text):
    f = preprocess_sample(tok, text, "query", "", MAXT)
    toks = tok([f], max_length=MAXT, truncation=True, return_tensors="pt").to("cuda:0")
    out = model(**toks, output_hidden_states=True)
    sl = toks["attention_mask"].sum(dim=1) - 1
    e = out["last_hidden_state"][0, sl[0]].float().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)

def stylometry(text):
    words = text.split()
    L = max(len(text), 1)
    nw = max(len(words), 1)
    return np.array([
        len(text), nw, len(set(w.lower() for w in words)),
        text.count("?"), text.count("!"), text.count("."),
        text.count(","), text.count("\n"),
        sum(c.isupper() for c in text) / L,
        sum(c.isdigit() for c in text) / L,
        sum(not c.isalnum() and not c.isspace() for c in text) / L,
        L / nw,
        len(set(text)) / L,
        float(any('\u4e00' <= ch <= '\u9fff' for ch in text)),
    ])

def risks(text):
    import urllib.request
    body = json.dumps({"text": text, "task": "query"}).encode()
    req = urllib.request.Request("http://127.0.0.1:8002/gate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        g = json.loads(r.read())
    return np.array([g["probabilities"].get(d, 0.0)
                     for d in sorted(g["probabilities"])] + [g["max_risk"]])

def views(text):
    return embed(text), stylometry(text), risks(text)

def norm_matrix(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def build(campaigns):
    V1, V2, V3, labels, ks = [], [], [], [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed] + [TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed)
                          for k in range(1, 6)]
        for k, t in enumerate(texts):
            v1, v2, v3 = views(t)
            V1.append(v1); V2.append(v2); V3.append(v3)
            labels.append(ci); ks.append(k)
    return np.array(V1), np.array(V2), np.array(V3), labels, ks

def eval_tau(D, labels, tau):
    n = len(labels)
    same = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] == labels[j]]
    diff = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] != labels[j]]
    tp = sum(1 for i, j in same if D[i, j] < tau)
    fp = sum(1 for i, j in diff if D[i, j] < tau)
    P = tp / max(tp + fp, 1); R = tp / max(len(same), 1)
    return {"P": round(P, 4), "R": round(R, 4),
            "F1": round(2*P*R/max(P+R, 1e-9), 4)}

def linkage(V1c, V2c, V3c, y, mode):
    if mode == "embed":
        X = norm_matrix(V1c)
    elif mode == "fusion":
        X = np.hstack([norm_matrix(V1c),
                       norm_matrix(V2c),
                       norm_matrix(V3c)])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return 1 - X @ X.T

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    calib, test = all_seeds[:20], all_seeds[20:50]
    print("building calib views ...", flush=True)
    V1c, V2c, V3c, yc, _ = build(calib)
    print("building test views ...", flush=True)
    V1t, V2t, V3t, yt, kt = build(test)

    results = {}
    for mode in ("embed", "fusion"):
        D_cal = linkage(V1c, V2c, V3c, yc, mode)
        taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        best = max(taus, key=lambda t: eval_tau(D_cal, yc, t)["F1"])
        D_test = linkage(V1t, V2t, V3t, yt, mode)
        res = eval_tau(D_test, yt, best)
        # per-k recall (k,0) pairs
        by_k = {}
        for k in range(1, 6):
            pairs = [(i, j) for i in range(len(yt)) for j in range(len(yt))
                     if yt[i] == yt[j] and {kt[i], kt[j]} == {0, k}]
            if pairs:
                hit = sum(1 for i, j in pairs if D_test[i, j] < best)
                by_k[f"k={k}"] = round(hit / len(pairs), 4)
        results[mode] = {"tau": best, **res, "recall_by_k": by_k}
        print(mode, results[mode], flush=True)
    out = {"experiment": "attribution_fusion_v2",
           "views": {"embed": "frozen 1024-d", "fusion": "embed+stylometry(14)+riskprobs(6)"},
           "n_calib": 20, "n_test": 30, "results": results}
    json.dump(out, open(os.path.join(OUT, "attribution_fusion_v2.json"), "w"), indent=2)
    print("saved")

if __name__ == "__main__":
    main()
