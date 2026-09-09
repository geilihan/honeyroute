import os
import os as _os
#!/usr/bin/env python3
"""E1: router detection quality on held-out seed splits.

For each domain, scores seed_val.jsonl (held out from training) via the
guard-engine /gate endpoint, computes per-domain and pooled (micro) metrics:
ROC-AUC, precision/recall @ threshold 0.5, FPR@95TPR, plus latency stats.
Writes /datagroup/guard/eval/honeyroute/e1_detection.json
"""
import json, os, time, glob
import urllib.request

GATE = "http://127.0.0.1:8002/gate"
SEEDS = os.environ.get("SEEDS_DIR", "/datagroup/guard/data/seeds")
OUT_DIR = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
os.makedirs(OUT_DIR, exist_ok=True)

def score(text, task="query"):
    body = json.dumps({"text": text, "task": task}).encode()
    req = urllib.request.Request(GATE, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def metrics(scores, labels):
    import numpy as np
    s, y = np.array(scores), np.array(labels)
    pos, neg = s[y == 1], s[y == 0]
    # ROC-AUC (rank based, handles ties)
    n = len(s); ranks = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1
    auc = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg) + 1e-12)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # FPR@95TPR
    t = np.sort(pos)[max(0, int(np.ceil(0.95 * len(pos))) - 1)] if len(pos) else 1.0
    fpr95 = float((neg >= t).mean()) if len(neg) else float('nan')
    return {"n": n, "auc": round(float(auc), 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "fpr@95tpr": round(fpr95, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def main():
    all_scores, all_labels = [], []
    per_domain = {}
    lat = []
    domains = sorted(os.listdir(SEEDS))
    for dom in domains:
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val):
            continue
        rows = [json.loads(l) for l in open(val, encoding="utf-8")]
        task = "response" if dom in ("Hazardous_Action_Generation",
                                     "Sensitive_Information_Leakage") else "query"
        ss, yy = [], []
        for r in rows:
            t0 = time.time()
            out = score(r["text"], task)
            lat.append(out["latency_ms"])
            s = out["max_risk"]
            ss.append(s); yy.append(int(r["label"]))
            all_scores.append(s); all_labels.append(int(r["label"]))
            time.sleep(0.01)
        m = metrics(ss, yy)
        m["task"] = task
        per_domain[dom] = m
        print(dom, m, flush=True)
    pooled = metrics(all_scores, all_labels)
    import statistics
    result = {
        "experiment": "E1_detection_quality",
        "gate_version": json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8002/status", timeout=10).read())["version"],
        "split": "seed_val (held out)",
        "threshold": 0.5,
        "per_domain": per_domain,
        "pooled": pooled,
        "latency_ms": {"mean": round(statistics.mean(lat), 1),
                       "p50": round(statistics.median(lat), 1),
                       "max": round(max(lat), 1), "n": len(lat)},
    }
    out = os.path.join(OUT_DIR, "e1_detection.json")
    json.dump(result, open(out, "w"), indent=2)
    print("POOLed:", pooled)
    print("saved", out)

if __name__ == "__main__":
    main()
