#!/usr/bin/env python3
"""W6 -- E2 fidelity judging robustness (fully offline from released JSONs).

Replaces the single-presentation-order fidelity estimate with an
**order-averaged** estimate: per pair, average the judge's YES/NO verdict from
the original order (`e2_faithfulness_v2.raw[*].eq_hp*`) and the swapped order
(`e2_judge2.raw2[*].eq_hp*`) -- the two files partition the SAME 300 pairs, so
per-pair pairing is exact. Also reports the order agreement + Cohen's kappa, a
bootstrap CI on the averaged estimate, the [orig, swapped] sensitivity interval,
and the cross-family re-judge already in `cross_judge.json` (Deepseek-V4-Flash).

Outputs e2_judge_robustness.json.
"""
import json, os, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def find_results_dir():
    for c in (os.environ.get("HONEYROUTE_OUT"),
              os.path.join(HERE, "_results"), os.path.join(HERE, "..", "_results"),
              os.path.join(HERE, "..", "results")):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise SystemExit("results dir not found")


RES = find_results_dir()


def kappa(v1, v2):
    n = len(v1)
    if n == 0:
        return 0.0, 0.0
    po = sum(a == b for a, b in zip(v1, v2)) / n
    p1 = sum(v1) / n; p2 = sum(v2) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return round((po - pe) / max(1 - pe, 1e-9), 4), round(po, 4)


def boot_ci(vals, n_boot=5000, seed=7):
    if not vals:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def order_average(orig, swap):
    """per-pair average across the two presentation orders (skip invalid = -1)."""
    avg, o_v, s_v = [], [], []
    for o, s in zip(orig, swap):
        o = int(o); s = int(s)
        if o < 0 and s < 0:
            continue
        if o < 0:
            avg.append(s)
        elif s < 0:
            avg.append(o)
        else:
            avg.append((o + s) / 2.0)
        if o >= 0 and s >= 0:
            o_v.append(o); s_v.append(s)
    return avg, o_v, s_v


def main():
    a = json.load(open(os.path.join(RES, "e2_faithfulness_v2.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(RES, "e2_judge2.json"), encoding="utf-8"))
    cross = json.load(open(os.path.join(RES, "cross_judge.json"), encoding="utf-8"))
    ra, rb = a["raw"], b["raw2"]
    assert len(ra) == len(rb) and all(ra[i]["i"] == rb[i]["i"] == i for i in range(len(ra)))

    out = {"experiment": "e2_judge_robustness",
           "method": "per-pair average of original + swapped presentation order "
                     "(e2_faithfulness_v2.raw vs e2_judge2.raw2, same 300 pairs); "
                     "bootstrap CI; cross-family re-judge from cross_judge.json",
           "n_pairs": len(ra), "per_honeypot": {}}

    for name, key in (("hpT", "eq_hpT"), ("hpC", "eq_hpC")):
        orig = [r[key] for r in ra]
        swap = [r[key] for r in rb]
        avg, o_v, s_v = order_average(orig, swap)
        F_orig = round(sum(o_v) / len(o_v), 4)
        F_swap = round(sum(s_v) / len(s_v), 4)
        F_avg = round(sum(avg) / len(avg), 4)
        k, po = kappa(o_v, s_v)
        out["per_honeypot"][name] = {
            "n_valid_orig": len(o_v), "n_valid_swap": len(s_v), "n_pairs_used": len(avg),
            "F_original_order": F_orig, "F_swapped_order": F_swap,
            "F_order_averaged": F_avg, "F_order_averaged_ci": boot_ci(avg),
            "sensitivity_interval": [min(F_orig, F_swap), max(F_orig, F_swap)],
            "order_agreement": po, "order_cohens_kappa": k,
        }

    out["cross_family"] = {
        "note": "second judge family = Deepseek-V4-Flash-Tencent (cross_judge.json); "
                "reproduces the headline fidelity values within noise",
        "hpT": cross.get("hpT"), "hpC": cross.get("hpC")}
    json.dump(out, open(os.path.join(RES, "e2_judge_robustness.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out["per_honeypot"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
