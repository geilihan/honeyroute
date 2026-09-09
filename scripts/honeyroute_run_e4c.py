#!/usr/bin/env python3
"""E4 (v3): attribution via frozen-backbone embedding fingerprints, ONLINE
threshold selection: tau chosen on a disjoint calibration set of campaigns.
Also reports per-k recall (k=1..5 identity rotations)."""
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

def eval_tau(D, labels, tau):
    n = len(labels)
    same = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] == labels[j]]
    diff = [(i, j) for i in range(n) for j in range(i+1, n) if labels[i] != labels[j]]
    tp = sum(1 for i, j in same if D[i, j] < tau)
    fp_ = sum(1 for i, j in diff if D[i, j] < tau)
    prec = tp / max(tp + fp_, 1); rec = tp / max(len(same), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"pairwise_precision": round(prec, 4), "pairwise_recall": round(rec, 4),
            "pairwise_f1": round(f1, 4)}

def build(campaigns):
    fps, labels, ks = [], [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed] + [TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed)
                          for k in range(1, 6)]
        for k, t in enumerate(texts):
            fps.append(embed(t)); labels.append(ci); ks.append(k)
    X = np.array(fps)
    return 1 - X @ X.T, labels, ks

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    calib, test = all_seeds[:20], all_seeds[20:50]
    D_c, y_c, _ = build(calib)
    best = max([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
               key=lambda t: eval_tau(D_c, y_c, t)["pairwise_f1"])
    D_t, y_t, k_t = build(test)
    res = eval_tau(D_t, y_t, best)
    # per-k recall: for each k, fraction of (k,0) campaign pairs linked
    by_k = {}
    for k in range(1, 6):
        pairs = [(i, j) for i in range(len(y_t)) for j in range(len(y_t))
                 if y_t[i] == y_t[j] and {k_t[i], k_t[j]} == {0, k}]
        if pairs:
            hit = sum(1 for i, j in pairs if D_t[i, j] < best)
            by_k[f"k={k}"] = round(hit / len(pairs), 4)
    out = {"experiment": "E4_attribution_embed_calibrated",
           "n_calib_campaigns": 20, "n_test_campaigns": 30, "k_rotations": 5,
           "dim": 1024,
           "tau_selected_on_calib": best,
           "test_metrics": res, "recall_by_rotation_k": by_k}
    json.dump(out, open(os.path.join(OUT, "e4_attribution_final.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
