#!/usr/bin/env python3
"""B1: LlamaGuard-4-12B baseline on the E1 detection test set.

Reads the SAME seed_val.jsonl files as honeyroute_run_e1.py (identical
test set), classifies each text with LG4 (vLLM :8013, GPU1), and
computes pooled + per-domain + per-language metrics and latency.
'unsafe' output = positive (attack) prediction.

Outputs /datagroup/guard/eval/honeyroute/e1_llamaguard4_baseline.json
"""
import json, os, re, sys, time, glob, urllib.request

SEEDS = os.environ.get("SEEDS_DIR", "/datagroup/guard/data/seeds")
OUT = os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
LG4_URL = "http://127.0.0.1:8013/v1/chat/completions"

def is_zh(t):
    return any('\u4e00' <= ch <= '\u9fff' for ch in t[:50])

def lg4_classify(text, max_retry=3):
    body = json.dumps({
        "model": "Llama-Guard-4-12B",
        "messages": [{"role": "user", "content":
                      [{"type": "text", "text": text}]}],
        "max_tokens": 40, "temperature": 0.0,
    }).encode()
    last = None
    for _ in range(max_retry):
        try:
            t0 = time.time()
            req = urllib.request.Request(LG4_URL, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            lat = time.time() - t0
            raw = (d["choices"][0]["message"]["content"] or "").strip()
            pred = 1 if re.search(r"\bunsafe\b", raw.lower()) else 0
            return pred, lat, raw[:200]
        except Exception as e:
            last = e; time.sleep(2)
    raise RuntimeError(f"LG4 query failed: {last}")

def main():
    recs = []
    doms = sorted(os.listdir(SEEDS))
    for dom in doms:
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val):
            continue
        rows = [json.loads(l) for l in open(val, encoding="utf-8")]
        for r in rows:
            recs.append({"text": r["text"], "label": int(r.get("label", 1)),
                         "domain": dom})
    print(f"test set: {len(recs)} samples, "
          f"{sum(r['label'] for r in recs)} positive", flush=True)

    out_recs = []
    for i, s in enumerate(recs):
        pred, lat, raw = lg4_classify(s["text"])
        out_recs.append({"i": i, "domain": s["domain"], "label": s["label"],
                         "pred": pred, "lat_s": round(lat, 3), "raw": raw,
                         "lang": "zh" if is_zh(s["text"]) else "en"})
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(recs)}", flush=True)
            json.dump(out_recs, open(os.path.join(OUT, "lg4_partial.json"), "w"))

    def block(rs):
        tp = sum(1 for r in rs if r["label"] == 1 and r["pred"] == 1)
        fp = sum(1 for r in rs if r["label"] == 0 and r["pred"] == 1)
        fn = sum(1 for r in rs if r["label"] == 1 and r["pred"] == 0)
        tn = sum(1 for r in rs if r["label"] == 0 and r["pred"] == 0)
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        return {"n": len(rs), "P": round(P, 4), "R": round(R, 4),
                "F1": round(2*P*R/max(P+R, 1e-9), 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    lats = sorted(r["lat_s"] for r in out_recs)
    pct = lambda p: lats[min(int(len(lats)*p), len(lats)-1)]
    per_domain = {}
    for dom in set(r["domain"] for r in out_recs):
        per_domain[dom] = block([r for r in out_recs if r["domain"] == dom])
    by_lang = {}
    for lang in set(r["lang"] for r in out_recs):
        by_lang[lang] = block([r for r in out_recs if r["lang"] == lang])

    out = {"experiment": "e1_llamaguard4_baseline",
           "model": "Llama-Guard-4-12B (vLLM :8013, GPU1)",
           "n": len(out_recs), "pooled": block(out_recs),
           "per_domain": per_domain, "by_language": by_lang,
           "latency_s": {"p50": round(pct(0.5), 3), "p90": round(pct(0.9), 3),
                         "p99": round(pct(0.99), 3),
                         "mean": round(sum(lats)/len(lats), 3)},
           "records": out_recs}
    json.dump(out, open(os.path.join(OUT, "e1_llamaguard4_baseline.json"), "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("records", "per_domain")}, indent=2))

if __name__ == "__main__":
    main()
