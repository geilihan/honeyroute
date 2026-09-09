#!/usr/bin/env python3
"""Attribution v3: supervised multi-view linkage (learned on calib campaigns).

Pairwise protocol identical to v2 (20 calib / 30 test campaigns, k=1..5
rotations), but instead of a single cosine threshold we train logistic
regression on calib pairs over per-view cosine similarities
[V1 embed, V2 stylometry, V3 risk-probs], select the decision threshold
by F1 on calib, and evaluate on test. Also reports top-1 anchor
retrieval: for each transformed variant, is its nearest k=0 anchor the
same campaign?
Output: /datagroup/guard/eval/honeyroute/attribution_fusion_v3.json
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
    return norm_matrix(np.array(V1)), norm_matrix(np.array(V2)), \
        norm_matrix(np.array(V3)), labels, ks

def pair_features(Xc_list, labels):
    """Dense sampling: all same-campaign pairs + equal-count random diff pairs."""
    n = len(labels)
    same = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] == labels[j]]
    rng = np.random.RandomState(0)
    diff = set()
    while len(diff) < len(same):
        i, j = rng.randint(0, n, 2)
        if i != j and labels[i] != labels[j]:
            diff.add((min(i, j), max(i, j)))
    F, y = [], []
    for (i, j), lab in zip(same + sorted(diff), [1]*len(same) + [0]*len(diff)):
        F.append([float(X[i] @ X[j]) for X in Xc_list]); y.append(lab)
    return np.array(F), np.array(y)

def full_pairs(X_list, labels):
    n = len(labels)
    idx = [(i, j) for i in range(n) for j in range(i+1, n)]
    F = np.array([[float(X[i] @ X[j]) for X in X_list] for i, j in idx])
    same_mask = np.array([labels[i] == labels[j] for i, j in idx])
    return F, same_mask, idx

def prf(y_true, y_pred):
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    P = tp / max(tp + fp, 1); R = tp / max(int(y_true.sum()), 1)
    return {"P": round(P, 4), "R": round(R, 4),
            "F1": round(2*P*R/max(P+R, 1e-9), 4)}

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    calib, test = all_seeds[:20], all_seeds[20:50]
    print("building calib views ...", flush=True)
    Vc = build(calib)
    print("building test views ...", flush=True)
    Vt = build(test)
    Xc_list = [Vc[0], Vc[1], Vc[2]]
    Xt_list = [Vt[0], Vt[1], Vt[2]]
    yc, yt, kt = Vc[3], Vt[3], Vt[4]

    # train on calib
    F_cal_dense, y_cal = pair_features(Xc_list, yc)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(F_cal_dense, y_cal)
    print("view weights (LR coefs):", lr.coef_[0].round(3), flush=True)
    # threshold by F1 on calib (full pair sweep)
    F_cal_all, same_cal, _ = full_pairs(Xc_list, yc)
    s_cal = lr.predict_proba(F_cal_all)[:, 1]
    taus = np.quantile(s_cal, np.linspace(0.05, 0.95, 37))
    best_tau, best_f1 = 0.5, -1
    for t in np.unique(np.round(taus, 4)):
        m = prf(same_cal, s_cal >= t)
        if m["F1"] > best_f1:
            best_f1, best_tau = m["F1"], float(t)

    # evaluate on test
    F_test, same_test, idx = full_pairs(Xt_list, yt)
    s_test = lr.predict_proba(F_test)[:, 1]
    pred = s_test >= best_tau
    res = prf(same_test, pred)

    # per-k recall
    lab_of = {i: (yt[i], kt[i]) for i in range(len(yt))}
    by_k = {}
    for k in range(1, 6):
        pairs = [(pi, (i, j)) for pi, (i, j) in enumerate(idx)
                 if yt[i] == yt[j] and {kt[i], kt[j]} == {0, k}]
        if pairs:
            hit = sum(1 for pi, _ in pairs if pred[pi])
            by_k[f"k={k}"] = round(hit / len(pairs), 4)

    # top-1 anchor retrieval: for each variant (k>=1), nearest k=0 anchor
    anchors = [i for i in range(len(yt)) if kt[i] == 0]
    correct = tot = 0
    for i in range(len(yt)):
        if kt[i] == 0:
            continue
        sims = [lr.predict_proba([[float(X[i] @ X[a]) for X in Xt_list]])[0, 1]
            for a in anchors]
        best_a = anchors[int(np.argmax(sims))]
        tot += 1; correct += int(yt[best_a] == yt[i])
    retrieval = round(correct / max(tot, 1), 4)

    out = {"experiment": "attribution_fusion_v3_supervised",
           "method": "logistic regression over per-view cosines "
                     "[embed, stylometry, riskprobs], trained on 20 calib "
                     "campaigns, threshold by calib F1",
           "n_calib": 20, "n_test": 30,
           "lr_coefs": [round(float(v), 4) for v in lr.coef_[0]],
           "tau": round(best_tau, 4),
           "pairwise": res, "recall_by_k": by_k,
           "anchor_retrieval_top1": retrieval}
    print(json.dumps(out, indent=2), flush=True)
    json.dump(out, open(os.path.join(OUT, "attribution_fusion_v3.json"), "w"),
              indent=2)
    print("saved")

if __name__ == "__main__":
    main()
