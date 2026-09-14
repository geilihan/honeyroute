#!/usr/bin/env python3
"""W7 -- per-deployment threshold-calibration recipe (offline).

Turns the "38.7% benign trigger / single corpus" limitation into a portable
recipe: given N local benign samples, sweep the gate-score threshold and report
the FP/FN cost curve, then validate the magnitude on a disjoint capture window.

Data (all released):
  * benign production scores: `e2b_benign_gate_scores.json` (300, the deployment's
    own benign traffic, scored by the gate);
  * attack score distribution: `e1_expanded_eval_v2.json` attacks' `l1_risk`
    (600) -- used as the attack-score proxy (same router);
  * disjoint capture window: `cross_distribution.json` (43 benign, 44 requests).

Reports FP(thr), TPR/FN(thr), the local slopes dFP/dthr and dFN/dthr, the
deployed operating point (hpC trigger at 0.7), and a recommended point.

Outputs e2b_threshold_recipe.json.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def find_results_dir():
    for c in (os.environ.get("HONEYROUTE_OUT"),
              os.path.join(HERE, "_results"), os.path.join(HERE, "..", "_results"),
              os.path.join(HERE, "..", "results")):
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise SystemExit("results dir not found")


RES = find_results_dir()


def main():
    benign = json.load(open(os.path.join(RES, "e2b_benign_gate_scores.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(RES, "e1_expanded_eval_v2.json"), encoding="utf-8"))
    attacks = [r["l1_risk"] for r in ev["records"]
               if r["label"] == 1 and isinstance(r["l1_risk"], (int, float))]
    cd = json.load(open(os.path.join(RES, "cross_distribution.json"), encoding="utf-8"))
    win = cd["capture_window_0901"]

    grid = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
    curve = []
    for thr in grid:
        fp = sum(1 for x in benign if x >= thr) / len(benign)
        tpr = sum(1 for x in attacks if x >= thr) / len(attacks)
        curve.append({"thr": thr, "FP_benign": round(fp, 4), "TPR": round(tpr, 4),
                      "FN": round(1 - tpr, 4)})

    def at(thr):
        return next(c for c in curve if abs(c["thr"] - thr) < 1e-9)

    dep = at(0.70)
    # local slopes (per +0.05 threshold step) around the deployed point
    i = grid.index(0.70)
    dFP = round(at(grid[i])["FP_benign"] - at(grid[i + 1])["FP_benign"], 4)
    dFN = round(at(grid[i + 1])["FN"] - at(grid[i])["FN"], 4)
    # recommended: balanced point (minimise FP + FN, equal weight)
    rec = min(curve, key=lambda c: c["FP_benign"] + c["FN"])

    out = {
        "experiment": "e2b_threshold_recipe",
        "method": "portable threshold calibration: from N local benign samples, sweep "
                  "the gate-score threshold and read the FP/FN cost curve; validate the "
                  "magnitude on a disjoint capture window",
        "n_benign_local": len(benign), "n_attack_proxy": len(attacks),
        "attack_proxy": "e1_expanded_eval_v2 attacks l1_risk (same router)",
        "deployed_operating_point": {"thr": 0.70, "FP_benign": dep["FP_benign"],
                                     "TPR": dep["TPR"], "FN": dep["FN"],
                                     "note": "hpC selective-bait trigger threshold"},
        "cost_curve": curve,
        "local_slopes_per_0.05": {"dFP": dFP, "dFN": dFN,
                                  "note": "raising thr by 0.05 cuts benign FP by dFP and "
                                          "costs dFN attack recall at the deployed point"},
        "recommended_point": {"thr": rec["thr"], "FP_benign": rec["FP_benign"],
                              "FN": rec["FN"],
                              "rule": "balanced point: argmin(FP + FN)"},
        "validation_second_window": {
            "n": win["n"], "n_benign": win["n_benign"],
            "FP_0.5": round(win["benign_over_0.5"] / max(win["n_benign"], 1), 4),
            "FP_0.7": round(win["benign_over_0.7_hpc_trigger"] / max(win["n_benign"], 1), 4),
            "attack_caught": win["attack_caught"],
            "note": "disjoint capture window; trigger-rate magnitude transfers, exact "
                    "value does not -> per-deployment recalibration needed"},
        "benign_score_summary": {
            "min": round(min(benign), 4), "median": round(sorted(benign)[len(benign) // 2], 4),
            "max": round(max(benign), 4),
            "frac_ge_0.5": round(sum(1 for x in benign if x >= 0.5) / len(benign), 4),
            "frac_ge_0.7": round(sum(1 for x in benign if x >= 0.7) / len(benign), 4)},
    }
    json.dump(out, open(os.path.join(RES, "e2b_threshold_recipe.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("deployed_operating_point", "local_slopes_per_0.05",
                                          "recommended_point", "validation_second_window",
                                          "benign_score_summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
