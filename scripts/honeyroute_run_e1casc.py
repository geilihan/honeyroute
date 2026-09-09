#!/usr/bin/env python3
"""E1 supplement: L1+L2 cascade baseline vs HoneyRoute router.

Cascade baseline (the standard two-tier guard): L1 gate (0.8B, 38ms)
screens; requests below threshold go to production; requests in the
medium-risk band go to L2 (8b guard-LLM moderation, ~14.8s) for a final
verdict. HoneyRoute: L1 alone decides (divert vs production), no L2 hop.

Measures on the pooled held-out set (n=103):
  - decision P/R/F1 (cascade: L1 pass + L2 verdict; ours: L1 threshold)
  - p50/p99 added latency per request (benign should avoid the 14.8s hop)
  - L2 invocation rate (cost driver)
Also on the red-team transformed set (n=100 variants): evasion rate.
Output: /datagroup/guard/eval/honeyroute/e1_cascade.json
"""
import json, os, random, statistics, sys, time, urllib.request

import os as _os
sys.path.insert(0, _os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
random.seed(42)
import os as _os
OUT = _os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
SEEDS = os.environ.get("SEEDS_DIR", "/datagroup/guard/data/seeds")

def gate(text, task="query"):
    body = json.dumps({"text": text, "task": task}).encode()
    req = urllib.request.Request("http://127.0.0.1:8002/gate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def l2_moderate(text):
    global _HEAD
    prompt = _head() + text + "<|im_end|>\n<|im_start|>assistant\n"
    body = json.dumps({"model": "production-8B", "prompt": prompt,
                       "max_tokens": 512, "temperature": 0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8011/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())["choices"][0]["text"]
    lat = (time.time() - t0) * 1000
    verdict = "unsafe" if out.strip().lower().startswith("unsafe") else "safe"
    return (1 if verdict == "unsafe" else 0), lat

_HEAD_CACHE = None
def _head():
    global _HEAD_CACHE
    if _HEAD_CACHE is None:
        import jinja2
        with open("/datagroup/guard/production-8B/chat_template.jinja",
                  encoding="utf-8") as f:
            tpl = f.read()
        t = jinja2.Environment().from_string(tpl)
        rendered = t.render(messages=[{"role": "user", "content": "\x00P\x00"}],
                            add_generation_prompt=True)
        _HEAD_CACHE = rendered.split("\x00P\x00")[0]
    return _HEAD_CACHE

def prf(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"P": round(P, 4), "R": round(R, 4),
            "F1": round(2 * P * R / max(P + R, 1e-9), 4)}

def main():
    rows = []
    for dom in sorted(os.listdir(SEEDS)):
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val):
            continue
        task = "response" if dom in ("Hazardous_Action_Generation",
                                     "Sensitive_Information_Leakage") else "query"
        for line in open(val, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"text": r["text"], "label": int(r["label"]), "task": task})
    labels = [r["label"] for r in rows]
    print("n =", len(rows), flush=True)

    # ---- cascade: margin bands. L1 margin < 1.0 -> pass; >= HC (high conf) -> block;
    # in-between -> L2 verdict. HC factor mirrors the deployed engine (hc*2 in margin terms).
    preds_cascade, lat_cascade, l2_calls = [], [], 0
    for r in rows:
        g = gate(r["text"], r["task"])
        margin = g["risk_margin"]
        t0 = time.time()
        lat = g["latency_ms"]
        if margin >= 1.7:          # high-confidence band (0.85*2 margin terms)
            pred = 1
        elif margin >= 1.0:        # medium band -> L2
            l2_calls += 1
            p2, lat2 = l2_moderate(r["text"])
            pred = p2
            lat += lat2
        else:
            pred = 0
        lat_cascade.append(time.time() - t0 + g["latency_ms"] / 1000)
        preds_cascade.append(pred)
        if len(preds_cascade) % 20 == 0:
            print(len(preds_cascade), flush=True)
    m_cascade = prf(preds_cascade, labels)
    m_cascade.update({
        "l2_invocation_rate": round(l2_calls / len(rows), 4),
        "p50_total_latency_s": round(statistics.median(lat_cascade), 2),
        "p99_total_latency_s": round(sorted(lat_cascade)[int(len(lat_cascade)*0.99)-1], 2)})

    # ---- ours: L1 alone at threshold margin 1.0 (uniform 0.5 max_risk equivalent)
    preds_ours, lat_ours = [], []
    for r in rows:
        g = gate(r["text"], r["task"])
        preds_ours.append(1 if g["max_risk"] >= 0.5 else 0)
        lat_ours.append(g["latency_ms"] / 1000)
    m_ours = prf(preds_ours, labels)
    m_ours.update({"p50_latency_s": round(statistics.median(lat_ours), 3)})

    out = {"experiment": "E1_cascade_baseline", "n": len(rows),
           "cascade_L1L2": m_cascade, "ours_L1_only": m_ours}
    json.dump(out, open(os.path.join(OUT, "e1_cascade.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
