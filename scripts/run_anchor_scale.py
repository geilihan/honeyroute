#!/usr/bin/env python3
"""E6b: anchor-library scaling curve for retrieval attribution.

Question: does three-view-fusion retrieval hold up as the campaign
database grows from 30 to 100+ anchors?

Design: anchors = all attack-seed campaigns (7 domains x N seeds) plus
JailbreakBench confirmed-jailbreak prompts as pseudo-campaigns, capped
at 120. Queries = k=1..5 rotated variants of held-out campaigns
(disjoint from the anchors). For anchor-set sizes
{30, 50, 80, 120} (random subsets, 3 draws each), report fusion-cosine
top-1/top-5 retrieval accuracy and the multiple-of-chance.
Output: /datagroup/guard/eval/honeyroute/anchor_scaling.json
"""
import json, os, random, sys
import numpy as np
import torch

import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
random.seed(2024); torch.manual_seed(2024)
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
    L = max(len(text), 1); nw = max(len(words), 1)
    return np.array([
        len(text), nw, len(set(w.lower() for w in words)),
        text.count("?"), text.count("!"), text.count("."),
        text.count(","), text.count("\n"),
        sum(c.isupper() for c in text) / L,
        sum(c.isdigit() for c in text) / L,
        sum(not c.isalnum() and not c.isspace() for c in text) / L,
        L / nw, len(set(text)) / L,
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

def views_of(texts):
    V1, V2, V3 = [], [], []
    for t in texts:
        v1, v2, v3 = embed(t), stylometry(t), risks(t)
        V1.append(v1); V2.append(v2); V3.append(v3)
    return norm_matrix(np.array(V1)), norm_matrix(np.array(V2)), \
        norm_matrix(np.array(V3))

def fusion(V1, V2, V3):
    X = np.hstack([V1, V2, V3])
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def main():
    # candidate campaigns: seed corpus + JBB prompts (limit text len)
    seed_camps = [(d, s) for d, atk in SEEDS.items() for s in atk]
    jbb = json.load(open(os.path.join(OUT, "jbb_attacks_uniq.json"),
                         encoding="utf-8"))
    jbb_prompts = [p["prompt"] if isinstance(p, dict) else p for p in jbb]
    jbb_prompts = [p[:1500] for p in jbb_prompts if len(p) > 50][:200]
    jbb_camps = [("jbb", p) for p in jbb_prompts]

    all_camps = seed_camps + jbb_camps
    random.shuffle(all_camps)
    n_query = 30          # query campaigns (held out)
    query_camps = all_camps[:n_query]
    pool = all_camps[n_query:]
    max_anchor = min(120, len(pool))
    print(f"pool={len(pool)} query={n_query}", flush=True)

    # build views: anchors (pool[:max_anchor]) + query anchor text + variants
    print("embedding pool anchors ...", flush=True)
    pool_texts = [s for _, s in pool[:max_anchor]]
    P1, P2, P3 = views_of(pool_texts)
    print("embedding query campaigns ...", flush=True)
    q_anchor_texts, q_variant_texts, q_ids = [], [], []
    for ci, (dom, seed) in enumerate(query_camps):
        q_anchor_texts.append(seed)
        for k in range(1, 6):
            q_variant_texts.append(TRANSFORMS[(ci + k) % len(TRANSFORMS)](seed))
            q_ids.append(ci)
    QA1, QA2, QA3 = views_of(q_anchor_texts)
    QV1, QV2, QV3 = views_of(q_variant_texts)
    print("views done", flush=True)
    json.dump({"status": "views_built"}, open(os.path.join(OUT, "anchor_partial.json"), "w"))

    sizes = [30, 50, 80, max_anchor]
    curve = {}
    rng = random.Random(7)
    for size in sizes:
        if size > max_anchor:
            continue
        trials = []
        for draw in range(3):
            idx = sorted(rng.sample(range(max_anchor), size))
            F_pool = fusion(P1[idx], P2[idx], P3[idx])
            F_qa = fusion(QA1, QA2, QA3)
            F_qv = fusion(QV1, QV2, QV3)
            top1 = top5 = 0
            # each variant retrieves among: its own campaign anchor + the
            # size-1 pool subset  (database = pool subset + 30 query anchors?
            # no: database = pool subset only; own anchor is NOT in db for
            # the pure scaling question. But then chance = 0... Instead:
            # database = pool subset + the 30 query anchors (realistic
            # forensic db: past campaigns incl. the source). chance=1/(size+30).)
            DB = np.vstack([F_pool, F_qa])
            db_owner = list(range(size)) + [max_anchor + i for i in range(n_query)]
            sims = F_qv @ DB.T
            for qi in range(len(q_ids)):
                order = np.argsort(-sims[qi])
                ranks = [db_owner[j] for j in order]
                own = max_anchor + q_ids[qi]
                hit_rank = ranks.index(own) if own in ranks else 999
                top1 += hit_rank == 0
                top5 += hit_rank < 5
            nq = len(q_ids)
            trials.append({"top1": round(top1 / nq, 4),
                           "top5": round(top5 / nq, 4),
                           "chance": round(1 / (size + n_query), 4)})
        curve[f"anchors={size}"] = {
            "db_size": size + n_query,
            "mean_top1": round(sum(t["top1"] for t in trials) / len(trials), 4),
            "mean_top5": round(sum(t["top5"] for t in trials) / len(trials), 4),
            "chance_top1": round(1 / (size + n_query), 4),
            "trials": trials}
        print(f"anchors={size}:", curve[f"anchors={size}"], flush=True)
        json.dump(curve, open(os.path.join(OUT, "anchor_partial.json"), "w"))

    out = {"experiment": "anchor_scaling_curve",
           "design": "query = 30 held-out campaigns x 5 rotated variants; "
                     "database = N pool anchors + the 30 query-campaign "
                     "anchors; fusion-cosine retrieval",
           "n_pool_available": max_anchor, "curve": curve}
    json.dump(out, open(os.path.join(OUT, "anchor_scaling.json"), "w"), indent=2)
    print("saved")

if __name__ == "__main__":
    main()
