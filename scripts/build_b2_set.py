#!/usr/bin/env python3
"""B2 v2: build the expanded E1 set from REAL sources (no teacher generation
of attacks — that failed quality control).

Attack side (~500):
  - seed_train.jsonl positives across 7 domains (held out from... no wait,
    seed_train was USED for head training — including them biases L1 upward.
    Instead: apply the 13 red-team transformations to seed_val positives
    (different transform than training-time augmentations where possible)
  - JailbreakBench 496 real prompts (goal text, label=1, domain=Prompt_Injection)
Benign side (~500):
  - GLM-5.3 one-shot per-sample benign generation (direct request style)
  - seed_val benign rows reused
Outputs e1_expanded_set_v2.json
"""
import json, os, random, re, sys, time, urllib.request

sys.path.insert(0, os.environ.get("GUARD_CODE_DIR", "/datagroup/guard/code"))
random.seed(123)
OUT = os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
SEEDS = os.environ.get("SEEDS_DIR", "/datagroup/guard/data/seeds")
TEACHER = os.environ.get("TEACHER_GATEWAY_URL")
KEY = os.environ.get("TEACHER_API_KEY", "")

from pipeline.redteam_harness import (t_upper, t_leetspeak, t_space_pad,
                                      t_wordless, t_translate_hint, t_roleplay,
                                      t_academic)
TRANSFORMS = [t_upper, t_leetspeak, t_space_pad, t_wordless,
              t_translate_hint, t_roleplay, t_academic]

def gen_one_benign(topic_hint):
    p = (f"Write a single realistic {topic_hint} user request to an AI "
         "assistant. Output ONLY the request text itself — no quotes, no "
         "explanation, no list. 1-3 sentences.")
    body = json.dumps({"model": "GLM-5.3",
                       "messages": [{"role": "user", "content": p}],
                       "max_tokens": 120, "temperature": 0.9}).encode()
    req = urllib.request.Request(TEACHER, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    t = (m.get("content") or "").strip().strip('"').strip()
    # quality gate: must look like a request, not meta
    if re.search(r"(here is|as requested|sure|the user|request to an ai|"
                 r"^```|^\")", t, re.I):
        return None
    if len(t) < 15 or len(t) > 600:
        return None
    return t

def main():
    is_zh = lambda t: any('\u4e00' <= c <= '\u9fff' for c in t[:50])

    attacks = []
    # 1) transformed seed_val positives (61 base -> ~350 with 6 transforms, capped)
    base = []
    for dom in sorted(os.listdir(SEEDS)):
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val): continue
        for l in open(val, encoding="utf-8"):
            r = json.loads(l)
            if r.get("label", 1) == 1:
                base.append((dom, r["text"]))
    print(f"base positives: {len(base)}", flush=True)
    for i, (dom, text) in enumerate(base):
        attacks.append({"text": text, "label": 1, "domain": dom, "src": "seed_val"})
        for k, tf in enumerate(TRANSFORMS[:5]):
            try:
                t2 = tf(text)
            except Exception:
                continue
            if t2 and t2 != text:
                attacks.append({"text": t2, "label": 1, "domain": dom,
                                "src": f"transform{k}"})
    # 2) JailbreakBench real prompts
    jbb = json.load(open(os.path.join(OUT, "jbb_attacks_uniq.json")))
    for j in jbb:
        attacks.append({"text": j["prompt"], "label": 1,
                        "domain": "Prompt_Injection_and_Jailbreak", "src": "jbb"})
    print(f"attacks assembled: {len(attacks)}", flush=True)

    # dedup + cap at 600
    seen = set(); uniq = []
    for a in attacks:
        if a["text"] not in seen:
            seen.add(a["text"]); uniq.append(a)
    attacks = uniq[:600]

    # benign: seed_val benign + GLM single requests
    benign = []
    for dom in sorted(os.listdir(SEEDS)):
        val = os.path.join(SEEDS, dom, "seed_val.jsonl")
        if not os.path.exists(val): continue
        for l in open(val, encoding="utf-8"):
            r = json.loads(l)
            if r.get("label", 1) == 0:
                benign.append({"text": r["text"], "label": 0,
                               "domain": "benign", "src": "seed_val"})
    print(f"seed_val benign: {len(benign)}", flush=True)
    topics = ["cooking", "programming", "travel", "math homework", "writing",
              "science", "history", "health", "business email", "language learning",
              "sports", "music", "home repair", "career", "shopping",
              "中文烹饪问题", "中文编程入门", "中文健康建议", "中文旅游规划", "中文育儿教育"]
    tries = 0
    while len(benign) < 520 and tries < 700:
        tries += 1
        topic = random.choice(topics)
        try:
            t = gen_one_benign(topic)
        except Exception:
            time.sleep(1); continue
        if t and t not in seen:
            seen.add(t)
            benign.append({"text": t, "label": 0, "domain": "benign",
                           "src": "glm",
                           "lang": "zh" if is_zh(t) else "en"})
        if tries % 50 == 0:
            print(f"benign {len(benign)}/{520} (tries {tries})", flush=True)
            json.dump(benign, open(os.path.join(OUT, "e1_exp_benign_v2_partial.json"), "w"),
                      ensure_ascii=False)
    print(f"benign assembled: {len(benign)}", flush=True)

    for a in attacks:
        a["lang"] = "zh" if is_zh(a["text"]) else "en"
    for b in benign:
        if "lang" not in b: b["lang"] = "zh" if is_zh(b["text"]) else "en"
    out = {"experiment": "e1_expanded_set_v2",
           "note": "attacks from seed_val+transforms+JBB (real, no teacher "
                   "generation); benign from seed_val + GLM one-shot requests "
                   "with quality gate",
           "n_attack": len(attacks), "n_benign": len(benign),
           "attacks": attacks, "benign": benign}
    json.dump(out, open(os.path.join(OUT, "e1_expanded_set_v2.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"DONE atk={len(attacks)} ben={len(benign)}")

if __name__ == "__main__":
    main()
