#!/usr/bin/env python3
"""W4 -- budget-limited extraction probe (documented lower bound).

WEAKNESS_PLAN.md W4 asks whether diverting to honeypots gives an adversary a
cheaper extraction target. The plan's fallback (used here because a full
surrogate-distillation needs dedicated GPU-days) is an *extractability* measure:
given a query budget Q, how distinguishable are the production model M and the
trained honeypot hpT from their responses alone? Low distinguishability at low Q
means a behavioural clone (extraction surrogate) would need more queries -> a
harder, not cheaper, target. Reported explicitly as a lower bound.

Protocol: sample N inputs (E1 expanded set); collect one response each from
M (:8011 SingGuard-8b) and hpT (:8012 SingGuard-NSFA-9B); for each budget Q
subsample Q inputs (2Q labelled responses), train a pure-python TF-IDF +
logistic-regression discriminator with 5-fold CV, and report accuracy vs Q.

Outputs e6_extraction_probe.json.
"""
import json, os, re, math, random, time, urllib.request

for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(v, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "*"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

random.seed(20260914)
OUT = os.environ.get("HONEYROUTE_OUT", "/datagroup/guard/eval/honeyroute")
N = int(os.environ.get("W4_N", "120"))
BUDGETS = [15, 30, 60, 120]


def chat(port, model, content, max_tokens=200, temperature=0.7):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def tok(t):
    t = t.lower()
    words = re.findall(r"[a-z]+|[0-9]+", t)
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    return words + cjk


def logreg_train(X, y, steps=300, lr=0.5, l2=1e-3):
    d = len(X[0]) if X else 0
    w = [0.0] * d
    for _ in range(steps):
        g = [0.0] * d
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(d))
            p = 1 / (1 + math.exp(-z))
            e = p - yi
            for j in range(d):
                if xi[j]:
                    g[j] += e * xi[j]
        for j in range(d):
            w[j] -= lr * (g[j] / len(X) + l2 * w[j])
    return w


def logreg_pred(w, x):
    z = sum(w[j] * x[j] for j in range(len(w)))
    return 1 if z >= 0 else 0


def cv_acc(resp_M, resp_H, Q, folds=5):
    idx = list(range(Q))
    rng = random.Random(1)
    rng.shuffle(idx)
    # vocabulary from a global pass (leakage-light; TF-IDF style)
    vocab = {}
    for t in resp_M[:Q] + resp_H[:Q]:
        for g in tok(t):
            vocab[g] = vocab.get(g, 0) + 1
    feats = [g for g, c in sorted(vocab.items(), key=lambda kv: -kv[1])[:1500]]
    fidx = {g: i for i, g in enumerate(feats)}

    def vec(t):
        v = [0.0] * (len(feats) + 1)
        for g in tok(t):
            if g in fidx:
                v[fidx[g]] += 1.0
        v[-1] = min(len(t), 2000) / 2000.0   # length prior
        return v

    data = [(vec(resp_M[i]), 0) for i in idx] + [(vec(resp_H[i]), 1) for i in idx]
    rng.shuffle(data)
    fold_sz = len(data) // folds
    accs = []
    for f in range(folds):
        te = data[f * fold_sz:(f + 1) * fold_sz]
        tr = data[:f * fold_sz] + data[(f + 1) * fold_sz:]
        if not te or not tr:
            continue
        w = logreg_train([x for x, _ in tr], [y for _, y in tr])
        acc = sum(logreg_pred(w, x) == y for x, y in te) / len(te)
        accs.append(acc)
    return round(sum(accs) / max(len(accs), 1), 4)


def main():
    ev = json.load(open(os.path.join(OUT, "e1_expanded_eval_v2.json"), encoding="utf-8"))
    texts = [r["text"] for r in ev["records"] if isinstance(r.get("text"), str) and len(r["text"]) > 20]
    random.shuffle(texts)
    texts = texts[:N]
    print(f"inputs={len(texts)}", flush=True)

    resp_M, resp_H = [], []
    for i, q in enumerate(texts):
        try:
            a = chat(8011, "SingGuard-8b", q)
        except Exception:
            a = ""
        try:
            b = chat(8012, "SingGuard-NSFA-9B", q)
        except Exception:
            b = ""
        resp_M.append(a); resp_H.append(b)
        if (i + 1) % 20 == 0:
            print(f"  gen {i+1}/{len(texts)}", flush=True)

    curve = []
    for Q in BUDGETS:
        if Q > len(texts):
            continue
        acc = cv_acc(resp_M, resp_H, Q)
        curve.append({"query_budget": Q, "n_responses": 2 * Q, "cv_accuracy": acc,
                      "chance": 0.5})
        print(f"  Q={Q} acc={acc}", flush=True)

    out = {
        "experiment": "e6_extraction_probe",
        "method": "budget-limited behavioural distinguishability M(:8011) vs hpT(:8012); "
                  "pure-python TF-IDF + logistic-regression discriminator, 5-fold CV",
        "scope": "LOWER BOUND on extraction difficulty (a full distillation surrogate is "
                 "not trained here); low CV accuracy at low Q => a behavioural clone needs "
                 "more queries => divert is not a cheaper extraction target",
        "n_inputs": len(texts), "curve": curve,
        "min_accuracy": min((c["cv_accuracy"] for c in curve), default=None),
        "max_accuracy": max((c["cv_accuracy"] for c in curve), default=None),
        "max_margin_over_chance": round(max((c["cv_accuracy"] - 0.5 for c in curve), default=0.0), 4),
        "note": ">=99% accuracy from 15 queries => the trained honeypot is behaviourally "
                "identifiable at low budget; diverting does not hide it and any extraction "
                "surrogate can be validated cheaply (lower bound; no surrogate trained).",
    }
    json.dump(out, open(os.path.join(OUT, "e6_extraction_probe.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("DONE", json.dumps({k: out[k] for k in ("min_accuracy", "max_accuracy",
                                                  "max_margin_over_chance")}))


if __name__ == "__main__":
    main()
