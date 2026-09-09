#!/usr/bin/env python3
"""E4 (v2): attribution via frozen-backbone embedding fingerprints.

Uses the NSFA-0.8B backbone's last-token hidden state (the same feature the
router heads consume) as the attacker fingerprint. 20 campaigns x 6 rotated
variants; k-away linkage precision/recall over cosine threshold sweep.
Runs in guard-engine on cuda:0 (free GPU).
"""
import json, os, random, sys
import numpy as np
import torch

import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
random.seed(42)
torch.manual_seed(42)
import os as _os
OUT = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")

from pipeline.attack_seeds import SEEDS
from pipeline.redteam_harness import (t_upper, t_leetspeak, t_space_pad,
                                      t_wordless, t_translate_hint, t_roleplay,
                                      t_academic)
TRANSFORMS = [t_upper, t_leetspeak, t_space_pad, t_wordless,
              t_translate_hint, t_roleplay, t_academic]

from transformers import AutoTokenizer, AutoModel
from train_head import preprocess_sample  # code dir already on sys.path

BB = os.environ.get("BACKBONE_DIR", "/datagroup/guard/guard-backbone-0.8B")
tok = AutoTokenizer.from_pretrained(BB, trust_remote_code=True)
model = AutoModel.from_pretrained(BB, trust_remote_code=True,
                                  torch_dtype=torch.bfloat16).to("cuda:0").eval()

SYSTEM_PROMPT = ""
MAX_TOKENS = 512

@torch.no_grad()
def embed(text, task="query"):
    formatted = preprocess_sample(tok, text, task, SYSTEM_PROMPT, MAX_TOKENS)
    toks = tok([formatted], max_length=MAX_TOKENS, truncation=True,
               return_tensors="pt").to("cuda:0")
    out = model(**toks, output_hidden_states=True)
    seq_len = toks["attention_mask"].sum(dim=1) - 1
    emb = out["last_hidden_state"][0, seq_len[0]].float().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-9)

def main():
    all_seeds = [(d, s) for d, atk in SEEDS.items() for s in atk]
    random.shuffle(all_seeds)
    campaigns = all_seeds[:20]
    fps, labels = [], []
    for ci, (dom, seed) in enumerate(campaigns):
        texts = [seed] + [TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed) for k in range(1, 6)]
        for t in texts:
            fps.append(embed(t)); labels.append(ci)
    X = np.array(fps)  # (120, 1024), unit-norm
    D = 1 - X @ X.T
    results = {}
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
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
        pure = sum(1 for g in groups.values()
                   if len(set(labels[i] for i in g)) == 1)
        prec = pure / len(groups)
        by_c = {}
        for i in range(len(X)):
            by_c.setdefault(labels[i], []).append(find(i))
        relink = sum(1 for c, rs in by_c.items() if len(set(rs)) == 1)
        rec = relink / len(by_c)
        # pairwise metrics too: fraction of same-campaign pairs linked
        same = [(i, j) for i in range(len(X)) for j in range(i+1, len(X)) if labels[i] == labels[j]]
        diff = [(i, j) for i in range(len(X)) for j in range(i+1, len(X)) if labels[i] != labels[j]]
        tp = sum(1 for i, j in same if D[i, j] < tau)
        fp_ = sum(1 for i, j in diff if D[i, j] < tau)
        results[f"tau={tau}"] = {
            "group_precision": round(prec, 4), "full_campaign_recall": round(rec, 4),
            "pairwise_precision": round(tp / max(tp + fp_, 1), 4),
            "pairwise_recall": round(tp / max(len(same), 1), 4)}
    out = {"experiment": "E4_attribution_embed",
           "n_campaigns": 20, "k_rotations": 5, "dim": int(X.shape[1]),
           "fingerprint": "frozen NSFA-0.8B last-token embedding (API-observable router feature)",
           "results": results}
    json.dump(out, open(os.path.join(OUT, "e4_attribution_embed.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("saved")

if __name__ == "__main__":
    main()
