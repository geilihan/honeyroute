#!/usr/bin/env python3
"""E5 ablation, exact recompute: pairwise linkage F1 for BOTH fingerprint
representations on the SAME campaigns:
  (a) frozen-backbone 1024-d embedding
  (b) head-visible risk-probability vector + stylometric features
Threshold calibrated on a disjoint campaign split for each, pairwise P/R/F1
reported on the test split. Mirrors run_e4c exactly."""
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
GATE = "http://127.0.0.1:8002/gate"

@torch.no_grad()
def embed(text):
    f = preprocess_sample(tok, text, "query", "", MAXT)
    toks = tok([f], max_length=MAXT, truncation=True, return_tensors="pt").to("cuda:0")
    out = model(**toks, output_hidden_states=True)
    sl = toks["attention_mask"].sum(dim=1) - 1
    e = out["last_hidden_state"][0, sl[0]].float().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)

def gate(text):
    import urllib.request
    body = json.dumps({"text": text, "task": "query"}).encode()
    req = urllib.request.Request(GATE, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def stylometry(text):
    words = text.split()
    L = max(len(text), 1)
    return [len(text), len(words), len(set(w.lower() for w in words)),
            text.count("?"), text.count("!"),
            sum(c.isupper() for c in text) / L,
            sum(c.isdigit() for c in text) / L,
            len(text) / max(len(words), 1)]

def eval_tau(D, labels, tau):
    n = len(labels)
    same = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] == labels[j]]
    diff = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] != labels[j]]
    tp = sum(1 for i, j in same if D[i, j] < tau)
    fp = sum(1 for i, j in diff if D[i, j] < tau)
    prec = tp / max(tp + fp, 1); rec = tp / max(len(same), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"P": round(prec, 4), "R": round(rec, 4), "F1": round(f1, 4)}

def build(campaigns, mode):
    feats, labels = [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed] + [TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed)
                          for k in range(1, 6)]
        for t in texts:
            if mode == "embed":
                feats.append(embed(t))
            else:
                g = gate(t)
                v = [g["probabilities"].get(d, 0.0)
                     for d in sorted(g["probabilities"])]
                v = v + stylometry(t) + [g["max_risk"]]
                arr = np.array(v)
                arr = (arr - arr.mean()) / (arr.std() + 1e-9)  # z-norm per set later
                feats.append(arr)
        labels += [ci] * 6
    X = np.array(feats)
    if mode != "embed":  # z-normalize across the whole matrix
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return 1 - Xn @ Xn.T, labels

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    calib, test = all_seeds[:20], all_seeds[20:50]
    taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    out = {}
    for mode in ("embed", "surface"):
        D_c, y_c = build(calib, mode)
        best = max(taus, key=lambda t: eval_tau(D_c, y_c, t)["F1"])
        D_t, y_t = build(test, mode)
        out[mode] = {"tau_calibrated": best, "test": eval_tau(D_t, y_t, best)}
        print(mode, out[mode], flush=True)
    json.dump({"experiment": "E5_ablation_fingerprint", **out},
              open(os.path.join(OUT, "e5_ablation_fingerprint.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
