#!/usr/bin/env python3
"""W1 -- language-conditional Chinese-benign threshold calibration (fully offline).

Reads the continuous L1 detector score `l1_risk` from `e1_expanded_eval_v2.json`
(decision rule in the paper: `max_risk >= 0.5`) and searches a Chinese-only
threshold `tau_zh` (English keeps `tau=0.5`).

Discipline (WEAKNESS_PLAN.md W1 / N1):
  * population = the deduped eval records (600 attacks + 503 benign; zh slice
    20 attacks + 126 benign) -- this is exactly the set the paper reports on;
  * tau_zh is searched on a DISJOINT calibration half; the held-out TEST half
    is what gets reported as the calibrated number;
  * the full-slice number is reported separately as the "deployment expectation";
  * repeated 2-fold splits give a split-noise interval; the primary split is
    seeded (PRIMARY_SEED) for a reproducible headline artifact.

Never overwrites existing numbers: writes `e1_zh_calibration.json` and adds a
`zh_calibrated` branch to `e1_expanded_eval_v2.json`.

Usage:  python run_zh_calibration.py [--results DIR]
"""
import argparse, json, math, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_EN = 0.5
PRIMARY_SEED = 20260914
N_SPLITS = 200


def find_results_dir(cli=None):
    for cand in (cli,
                 os.environ.get("HONEYROUTE_OUT"),
                 os.path.join(HERE, "_results"),
                 os.path.join(HERE, "..", "_results"),
                 os.path.join(HERE, "..", "results"),
                 os.path.join(HERE, "results")):
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    raise SystemExit("results dir not found (pass --results)")


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - hw), 4), round(min(1.0, c + hw), 4)]


def metrics(pairs):
    """pairs: iterable of (label, pred)."""
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    tn = sum(1 for y, p in pairs if y == 0 and p == 0)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    return {"n": tp + fp + fn + tn, "P": round(P, 4), "R": round(R, 4),
            "F1": round(F1, 4), "P_ci": wilson(tp, tp + fp),
            "R_ci": wilson(tp, tp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def choose_tau_zh(neg_scores, pos_scores, grid):
    """Smallest FP subject to FN=0 on the calibration half.

    FP is non-increasing in tau, so the optimum is the largest tau that still
    detects every attack; we scan a grid to stay robust to ties.
    """
    best = None
    for t in grid:
        fn = sum(1 for s in pos_scores if s < t)
        if fn != 0:
            continue
        fp = sum(1 for s in neg_scores if s >= t)
        if best is None or fp < best[1] or (fp == best[1] and t > best[0]):
            best = (t, fp)
    if best is not None:
        return best[0]
    # fallback: cannot hold FN=0 (should not happen); minimise FN then FP
    m = min(pos_scores)
    return max(0.0, m - 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    res_dir = find_results_dir(args.results)
    eval_path = os.path.join(res_dir, "e1_expanded_eval_v2.json")
    ev = json.load(open(eval_path, encoding="utf-8"))
    R = ev["records"]

    zh_neg = [r for r in R if r["lang"] == "zh" and r["label"] == 0]
    zh_pos = [r for r in R if r["lang"] == "zh" and r["label"] == 1]
    en = [r for r in R if r["lang"] == "en"]
    print(f"population: {len(R)} (zh benign {len(zh_neg)}, zh attack {len(zh_pos)}, "
          f"en {len(en)})")

    grid = sorted({r["l1_risk"] for r in zh_neg + zh_pos})

    # ---- repeated 2-fold splits: split-noise interval on the held-out half ----
    held_fp, held_fn, held_f1, calib_tau = [], [], [], []
    for seed in range(1, N_SPLITS + 1):
        rng = random.Random(seed)
        neg = zh_neg[:]; pos = zh_pos[:]
        rng.shuffle(neg); rng.shuffle(pos)
        nb, pa = len(neg) // 2, len(pos) // 2
        c_neg, t_neg = neg[:nb], neg[nb:]
        c_pos, t_pos = pos[:pa], pos[pa:]
        t = choose_tau_zh([r["l1_risk"] for r in c_neg],
                          [r["l1_risk"] for r in c_pos], grid)
        tp = sum(1 for r in t_pos if r["l1_risk"] >= t)
        fn = len(t_pos) - tp
        fp = sum(1 for r in t_neg if r["l1_risk"] >= t)
        tn = len(t_neg) - fp
        P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1)
        held_fp.append(fp); held_fn.append(fn)
        held_f1.append(2 * P * Rc / max(P + Rc, 1e-9)); calib_tau.append(t)

    def stat(xs):
        return {"mean": round(sum(xs) / len(xs), 3),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    # ---- primary split (reproducible headline artifact) ----
    rng = random.Random(PRIMARY_SEED)
    neg = zh_neg[:]; pos = zh_pos[:]
    rng.shuffle(neg); rng.shuffle(pos)
    nb, pa = len(neg) // 2, len(pos) // 2
    c_neg, t_neg = neg[:nb], neg[nb:]
    c_pos, t_pos = pos[:pa], pos[pa:]
    tau_primary = choose_tau_zh([r["l1_risk"] for r in c_neg],
                                [r["l1_risk"] for r in c_pos], grid)
    tau_full = min(r["l1_risk"] for r in zh_pos)  # FN=0 over the whole slice

    def eval_zh(tau, negs, poss):
        m = metrics([(0, int(r["l1_risk"] >= tau)) for r in negs] +
                    [(1, int(r["l1_risk"] >= tau)) for r in poss])
        return m

    held = eval_zh(tau_primary, t_neg, t_pos)
    full_before = eval_zh(TAU_EN, zh_neg, zh_pos)
    full_after = eval_zh(tau_full, zh_neg, zh_pos)
    # English unchanged by tau_zh
    en_m = metrics([(r["label"], int(r["l1_risk"] >= TAU_EN)) for r in en])
    pooled_before = metrics([(r["label"], int(r["l1_risk"] >= TAU_EN)) for r in R])
    pooled_after = metrics(
        [(r["label"], int(r["l1_risk"] >= TAU_EN)) for r in en] +
        [(0, int(r["l1_risk"] >= tau_full)) for r in zh_neg] +
        [(1, int(r["l1_risk"] >= tau_full)) for r in zh_pos])

    out = {
        "experiment": "e1_zh_calibration",
        "method": "language-conditional threshold tau_zh on L1 max_risk; "
                  "tau_en=0.5 unchanged; tau chosen to hold zh FN=0 and minimise zh FP",
        "source": os.path.basename(eval_path),
        "population": {"total": len(R), "zh_benign": len(zh_neg),
                       "zh_attack": len(zh_pos), "en": len(en)},
        "tau_en": TAU_EN,
        "primary_split": {"seed": PRIMARY_SEED,
                          "calib": {"zh_benign": len(c_neg), "zh_attack": len(c_pos)},
                          "test": {"zh_benign": len(t_neg), "zh_attack": len(t_pos)},
                          "tau_zh_chosen": round(tau_primary, 4),
                          "held_out_zh": held},
        "repeated_splits": {"n_splits": N_SPLITS, "fold": "2 (disjoint halves)",
                            "held_out_zh_fp": stat(held_fp),
                            "held_out_zh_fn": stat(held_fn),
                            "held_out_zh_f1": stat(held_f1),
                            "tau_zh_chosen": stat(calib_tau)},
        "deployment_expectation_full_slice": {"tau_zh": round(tau_full, 4),
                                              "zh_before": full_before,
                                              "zh_after": full_after},
        "english_unchanged": en_m,
        "pooled_before": pooled_before,
        "pooled_after": pooled_after,
        "note": "held_out_zh_* are the reported calibrated numbers (disjoint test "
                "half); deployment_expectation_full_slice is the in-sample-tinged "
                "full-slice value (half of it tuned) -- report with that caveat.",
    }
    json.dump(out, open(os.path.join(res_dir, "e1_zh_calibration.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    # additively attach a zh_calibrated branch (never overwrite originals)
    ev.setdefault("zh_calibrated", {})
    ev["zh_calibrated"] = {
        "tau_zh": round(tau_full, 4), "tau_en": TAU_EN,
        "held_out_primary": held,
        "held_out_split_mean": {"fp": stat(held_fp)["mean"], "fn": stat(held_fn)["mean"]},
        "full_slice_zh": full_after, "pooled": pooled_after,
        "source_script": "scripts/run_zh_calibration.py",
    }
    json.dump(ev, open(eval_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps({k: out[k] for k in
                      ("tau_en", "primary_split", "repeated_splits",
                       "deployment_expectation_full_slice", "pooled_before",
                       "pooled_after")}, ensure_ascii=False, indent=2))
    print(f"\nwrote {os.path.join(res_dir, 'e1_zh_calibration.json')}")


if __name__ == "__main__":
    main()
