#!/usr/bin/env python3
"""Attribution v4: retrieval-based campaign attribution (operationally framed).

Pairwise same/different precision is capped by the corpus (same-domain
campaigns are near-paraphrases), so we report the analyst task: given an
observed transformed request, retrieve its source campaign from a
database of 30 anchor campaigns (chance top-1 = 1/30 = 3.3%).
Views: V1 embed (1024-d), V2 stylometry (14-d), V3 risk probs (6-d).
Linkers: cosine on V1, cosine on fused views, supervised LR (trained on
20 calib campaigns) on per-view cosines.
Output: /datagroup/guard/eval/honeyroute/attribution_retrieval_v4.json
"""
import json, os, random, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

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

def norm_matrix(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def build(campaigns):
    V1, V2, V3, labels, ks = [], [], [], [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed] + [TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed)
                          for k in range(1, 6)]
        for k, t in enumerate(texts):
            v1, v2, v3 = embed(t), stylometry(t), risks(t)
            V1.append(v1); V2.append(v2); V3.append(v3)
            labels.append(ci); ks.append(k)
    return norm_matrix(np.array(V1)), norm_matrix(np.array(V2)), \
        norm_matrix(np.array(V3)), labels, ks

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    calib, test = all_seeds[:20], all_seeds[20:50]
    print("building calib views ...", flush=True)
    V1c, V2c, V3c, yc, kc = build(calib)
    print("building test views ...", flush=True)
    V1t, V2t, V3t, yt, kt = build(test)

    def fusion(V1, V2, V3):
        X = np.hstack([V1, V2, V3])
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # supervised linker on calib pairs
    n = len(yc)
    same = [(i, j) for i in range(n) for j in range(i+1, n) if yc[i] == yc[j]]
    rng = np.random.RandomState(0)
    diff = set()
    while len(diff) < len(same):
        i, j = rng.randint(0, n, 2)
        if i != j and yc[i] != yc[j]:
            diff.add((min(i, j), max(i, j)))
    Xc_list = [V1c, V2c, V3c]
    F, y = [], []
    for (i, j), lab in zip(same + sorted(diff), [1]*len(same) + [0]*len(diff)):
        F.append([float(X[i] @ X[j]) for X in Xc_list]); y.append(lab)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(np.array(F), np.array(y))

    # retrieval evaluation on test
    anchors = [i for i in range(len(yt)) if kt[i] == 0]
    variants = [i for i in range(len(yt)) if kt[i] > 0]
    Xt_list = [V1t, V2t, V3t]
    Ft = fusion(V1t, V2t, V3t)

    def score_linker(name, i, a):
        if name == "embed_cos":
            return float(V1t[i] @ V1t[a])
        if name == "fusion_cos":
            return float(Ft[i] @ Ft[a])
        if name == "supervised":
            return lr.predict_proba([[float(X[i] @ X[a]) for X in Xt_list]])[0, 1]
        raise ValueError(name)

    results = {}
    for name in ("embed_cos", "fusion_cos", "supervised"):
        top1 = top5 = 0
        by_k = {}
        for i in variants:
            sims = [(score_linker(name, i, a), yt[a]) for a in anchors]
            sims.sort(reverse=True)
            ks_hit = [1 if lab == yt[i] else 0 for _, lab in sims]
            top1 += ks_hit[0]
            top5 += sum(ks_hit[:5]) > 0
            by_k.setdefault(kt[i], []).append(ks_hit[0])
        results[name] = {
            "top1": round(top1 / len(variants), 4),
            "top5": round(top5 / len(variants), 4),
            "chance_top1": round(1 / len(anchors), 4),
            "top1_by_k": {f"k={k}": round(sum(v)/len(v), 4)
                          for k, v in sorted(by_k.items())},
        }
        print(name, results[name], flush=True)

    out = {"experiment": "attribution_retrieval_v4",
           "task": "given a transformed variant (k=1..5 rotations), retrieve "
                   "its source campaign among 30 anchors",
           "n_calib": 20, "n_test": 30, "n_variants": len(variants),
           "lr_coefs": [round(float(v), 4) for v in lr.coef_[0]],
           "results": results}
    json.dump(out, open(os.path.join(OUT, "attribution_retrieval_v4.json"), "w"),
              indent=2)
    print("saved")

if __name__ == "__main__":
    main()
