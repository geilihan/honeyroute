#!/usr/bin/env python3
"""E4 supplement: pooled held-out F1 for each router generation v0/v1/v2.

For each daily head version (runs/heads/2026-09-{01,02,03}), score the
SAME pooled held-out set (all seven domains' seed_val.jsonl, n=103) and
report pooled P/R/F1. Heads are MLPs over the frozen 0.8B embedding; we
extract embeddings locally (cuda:0) and run each generation's heads on top.
"""
import json, os, sys, glob
import numpy as np
import torch

import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
import torch.nn as nn

import os as _os
OUT = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
SEEDS = os.environ.get("SEEDS_DIR", "/datagroup/guard/data/seeds")
HEADS = "/datagroup/guard/runs/heads"
MAXT = 512

from train_head import preprocess_sample
from transformers import AutoTokenizer, AutoModel

BB = os.environ.get("BACKBONE_DIR", "/datagroup/guard/guard-backbone-0.8B")
tok = AutoTokenizer.from_pretrained(BB, trust_remote_code=True)
model = AutoModel.from_pretrained(BB, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to("cuda:0").eval()

@torch.no_grad()
def embed(text, task):
    f = preprocess_sample(tok, text, task, "", MAXT)
    toks = tok([f], max_length=MAXT, truncation=True, return_tensors="pt").to("cuda:0")
    out = model(**toks, output_hidden_states=True)
    sl = toks["attention_mask"].sum(dim=1) - 1
    return out["last_hidden_state"][0, sl[0]].float().cpu()

class Head(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers = []
        in_d = cfg["input_size"]
        for h in cfg["hidden_dims"]:
            layers.append(nn.Linear(in_d, h))
            if cfg.get("use_layer_norm"):
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.get("dropout_rate", 0.3)))
            in_d = h
        self.layers = nn.ModuleList([nn.Sequential(*layers)])
        self.output_layer = nn.Linear(in_d, cfg["num_classes"])
    def forward(self, x):
        return self.output_layer(self.layers[0](x))

def build_head(head_cfg):
    return Head(head_cfg)

def prf(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(2*P*R/max(P+R, 1e-9), 4)}

def main():
    # pooled held-out set
    rows = []
    for dom in sorted(os.listdir(SEEDS)):
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val):
            continue
        task = "response" if dom in ("Hazardous_Action_Generation",
                                     "Sensitive_Information_Leakage") else "query"
        for line in open(val, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"text": r["text"], "label": int(r["label"]),
                         "task": task, "domain": dom})
    print("n =", len(rows))
    # embed once
    feats, labels, tasks, doms = [], [], [], []
    for i, r in enumerate(rows):
        feats.append(embed(r["text"], r["task"]))
        labels.append(r["label"]); tasks.append(r["task"]); doms.append(r["domain"])
        if (i+1) % 25 == 0: print("embedded", i+1, flush=True)
    X = torch.stack(feats)

    results = {}
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        hdir = os.path.join(HEADS, day)
        preds = []
        for x, task, dom in zip(X, tasks, doms):
            # find head matching this domain & task; fall back to max over
            # query heads when this exact domain head is absent in this gen
            cand = glob.glob(os.path.join(hdir, f"NSFA-{dom}_head.pth"))
            if not cand:
                # use ensemble max over available heads of same task
                hs = [h for h in glob.glob(os.path.join(hdir, "*_head.pth"))]
                best = 0.0
                for hp in hs:
                    d = torch.load(hp, map_location="cpu", weights_only=False)
                    if d.get("task") != task:
                        continue
                    net = build_head(d["head_config"])
                    net.load_state_dict(d["head_state_dict"]); net.eval()
                    p = torch.softmax(net(x.unsqueeze(0)), -1)[0, 1].item()
                    thr = d.get("calibrated_threshold", 0.5)
                    best = max(best, p / max(thr, 1e-6))
                preds.append(1 if best >= 1.0 else 0)
                continue
            d = torch.load(cand[0], map_location="cpu", weights_only=False)
            net = build_head(d["head_config"])
            net.load_state_dict(d["head_state_dict"]); net.eval()
            p = torch.softmax(net(x.unsqueeze(0)), -1)[0, 1].item()
            thr = d.get("calibrated_threshold", 0.5)
            preds.append(1 if p / max(thr, 1e-6) >= 1.0 else 0)
        results[f"v_{day}"] = prf(preds, labels)
        print(day, results[f"v_{day}"], flush=True)
    json.dump({"experiment": "E4_pooled_per_generation",
               "n_heldout": len(rows), "results": results},
              open(os.path.join(OUT, "e4_pooled_generations.json"), "w"), indent=2)
    print("saved")

if __name__ == "__main__":
    main()
