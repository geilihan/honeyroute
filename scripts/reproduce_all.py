#!/usr/bin/env python3
"""One-command offline reproduction of every headline number in the paper.

Reads ONLY the released result JSONs under ``_results/`` (no gate, no GPU, no
network) and re-derives each quantitative claim, printing PASS/FAIL/SKIP per
check.  Exits non-zero if any check FAILs, so it doubles as an artifact verifier.

Usage:
  python phase1/reproduce_all.py            # verify (exit 1 on any FAIL)
  python phase1/reproduce_all.py --json     # also write _results/reproduce_report.json

This is the "analysis half" of the artifact: given the released JSONs, every
number in the paper is recomputable here.  (The experiment half additionally
needs the gateway / models.)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "_results")
_cache = {}


def load(name):
    if name not in _cache:
        p = os.path.join(RES, name)
        _cache[name] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    return _cache[name]


def get(name, path, default=None):
    d = load(name)
    if d is None:
        return default
    cur = d
    for k in path.split("/"):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
            cur = cur[int(k)]
        else:
            return default
    return cur


def close(a, b, tol=5e-3):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


CHECKS = []  # (label, actual, expected, tol)


def chk(label, actual, expected, tol=5e-3):
    CHECKS.append((label, actual, expected, tol))


def main():
    # ---------- E1: detection quality ----------
    chk("E1 small-set F1=.911", get("e1_cascade.json", "ours_L1_only/F1"), 0.9106, 1e-3)
    chk("E1 small-set P=.903", get("e1_cascade.json", "ours_L1_only/P"), 0.9032, 1e-3)
    chk("E1 small-set R=.918", get("e1_cascade.json", "ours_L1_only/R"), 0.918, 1e-3)
    chk("E1 L1 p50=38ms", get("e1_cascade.json", "ours_L1_only/p50_latency_s"), 0.038, 1e-3)
    chk("E1 cascade F1=.948", get("e1_cascade.json", "cascade_L1L2/F1"), 0.9483, 1e-3)
    chk("E1 cascade p50=14.6s", get("e1_cascade.json", "cascade_L1L2/p50_total_latency_s"), 14.63, 0.02)
    chk("E1 expanded benign n=503", get("e1_expanded_eval_v2.json", "n") is not None and
        sum(1 for r in load("e1_expanded_eval_v2.json")["records"] if r["label"] == 0), 503, 0)
    chk("E1 expanded attack n=600",
        sum(1 for r in load("e1_expanded_eval_v2.json")["records"] if r["label"] == 1), 600, 0)
    chk("E1 expanded JBB n=250",
        sum(1 for r in load("e1_expanded_eval_v2.json")["records"] if r["src"] == "jbb"), 250, 0)
    chk("E1 LG4 small-set F1=.939", get("e1_llamaguard4_baseline.json", "pooled/F1"), 0.9391, 1e-3)
    chk("E1 ShieldGemma pooled F1=.711", get("e1_shieldgemma_baseline.json", "pooled/F1"), 0.7111, 1e-3)
    chk("E1 LG4 expanded F1=.944", get("e1_expanded_eval_v2.json", "results/LG4/pooled/F1"), 0.9437, 1e-3)
    chk("E1 cascade L2 invocation=.56", get("e1_cascade.json", "cascade_L1L2/l2_invocation_rate"), 0.5631, 1e-3)
    _recs = [r for r in (load("e1_expanded_eval_v2.json") or {}).get("records", []) if r.get("label") == 1]
    _l1m = sum(1 for r in _recs if int(r.get("l1_pred", 1)) == 0)
    _lg4m = sum(1 for r in _recs if int(r.get("lg4_pred", 1)) == 0)
    _ov = sum(1 for r in _recs if int(r.get("l1_pred", 1)) == 0 and int(r.get("lg4_pred", 1)) == 0)
    chk("E1 router misses=12", _l1m, 12, 0)
    chk("E1 LG4 misses=47", _lg4m, 47, 0)
    chk("E1 miss overlap=8", _ov, 8, 0)

    # ---------- W1: Chinese calibration ----------
    chk("W1 tau_zh=0.5779", get("e1_zh_calibration.json", "primary_split/tau_zh_chosen"), 0.5779, 1e-4)
    chk("W1 full-slice zh FP=14", get("e1_zh_calibration.json", "deployment_expectation_full_slice/zh_after/fp"), 14, 0)
    chk("W1 pooled F1=.913", get("e1_zh_calibration.json", "pooled_after/F1"), 0.913, 1e-3)
    chk("W1 200-split FP mean=3.475", get("e1_zh_calibration.json", "repeated_splits/held_out_zh_fp/mean"), 3.475, 1e-3)
    chk("W1 200-split FN mean=0.86", get("e1_zh_calibration.json", "repeated_splits/held_out_zh_fn/mean"), 0.86, 1e-2)

    # ---------- E2: fidelity / judge reliability (W6) ----------
    chk("E2 hpT order-avg F=.931", get("e2_judge_robustness.json", "per_honeypot/hpT/F_order_averaged"), 0.931, 1e-3)
    chk("E2 hpC order-avg F=.263", get("e2_judge_robustness.json", "per_honeypot/hpC/F_order_averaged"), 0.263, 1e-3)
    chk("E2 hpC kappa=.021", get("e2_judge_robustness.json", "per_honeypot/hpC/order_cohens_kappa"), 0.021, 1e-3)
    chk("E2 hpC cross-family kappa=.24", get("e2_judge_robustness.json", "cross_family/hpC/cohens_kappa"), 0.2356, 5e-3)

    chk("E2 hpT raw F=.930 (264/284)", get("e2_faithfulness_v2.json", "F_hpT"), 0.9296, 1e-3)
    chk("E2 hpT n_valid=284", get("e2_faithfulness_v2.json", "n_valid_hpT"), 284, 0)
    chk("E2 hpC raw F=.076 (19/249)", get("e2_faithfulness_v2.json", "F_hpC"), 0.0763, 1e-3)
    chk("E2 hpC n_valid=249", get("e2_faithfulness_v2.json", "n_valid_hpC"), 249, 0)
    chk("E2b selective n_valid=252", get("e2b_selective_hpc.json", "n_valid"), 252, 0)
    # ---------- E2b: selective bait / threshold recipe (W7) ----------
    chk("E2b selective F_hpC=.889", get("e2b_selective_hpc.json", "F_hpC_v2"), 0.8889, 1e-3)
    chk("E2b benign trigger=.387", get("e2b_selective_hpc.json", "trigger_rate_benign"), 0.3867, 1e-3)
    cc = {round(c["thr"], 2): c for c in (get("e2b_threshold_recipe.json", "cost_curve") or [])}
    chk("E2b thr.85 FP=.0033", (cc.get(0.85) or {}).get("FP_benign"), 0.0033, 1e-3)
    chk("E2b thr.85 FN=.3767", (cc.get(0.85) or {}).get("FN"), 0.3767, 1e-3)
    chk("E2b thr.6 FP=.75", (cc.get(0.6) or {}).get("FP_benign"), 0.75, 1e-3)
    chk("E2b thr.6 FN=.0567", (cc.get(0.6) or {}).get("FN"), 0.0567, 1e-3)

    # ---------- E3 / E6: cost + flood absorption ----------
    dt = get("e6_stress_v3_real.json", "A_direct/prod_tokens")
    hr = get("e6_stress_v3_real.json", "B_honeyroute/prod_tokens")
    chk("E6 direct prod tokens=708077", dt, 708077, 0)
    chk("E6 honeyroute prod tokens=15351", hr, 15351, 0)
    chk("E6 token reduction=97.8%", (1 - hr / dt) if (dt and hr) else None, 0.9783, 1e-3)
    chk("E6 token saving=46x", (dt / hr) if (dt and hr) else None, 46.1, 0.5)
    chk("E6 flood diversion=1.0", get("e6_stress_v3_real.json", "B_honeyroute/diversion_by_family/gcg_flood"), 1.0, 0)
    chk("E6 benign diverted=93", get("e6_stress_v3_real.json", "B_honeyroute/benign_diverted"), 93, 0)

    # ---------- E4: continuous-analysis loop ----------
    chk("E4 legit FPR .30->.033", get("e4_positive_loop_v3.json", "v1_legit_fpr"), 0.0333, 1e-3)
    chk("E4 pooled F1 .911->.933", get("e4_positive_loop_v3.json", "pooled_after/F1"), 0.9333, 1e-3)

    # ---------- E7: multi-turn escalation + evasion ----------
    chk("E7 held-out detection=1.0", get("e7_expanded_final.json", "held_out_test/attack_detection"), 1.0, 0)
    chk("E7 held-out fire turn=2.8", get("e7_expanded_final.json", "held_out_test/attack_mean_fire_turn"), 2.8, 0.05)
    chk("E7 held-out benign FP=.08", get("e7_expanded_final.json", "held_out_test/benign_false_flag"), 0.08, 1e-3)
    rob = load("e7_evasion_robustness.json") or {}
    det = sum(int(round(v["rule_detection"] * v["n"])) for v in rob.values())
    n = sum(v["n"] for v in rob.values())
    chk("E7 scripted evasion 42/43=.977", (det / n) if n else None, 0.9767, 1e-3)
    chk("W2 adaptive evasion 1/36=.0278", get("e7_evasion_adaptive.json", "aggregate/adaptive_evasion_rate"), 0.0278, 1e-3)
    chk("O2 v2 valid detection=1.0", get("e7_evasion_adaptive_v2.json", "aggregate/single_adaptive_detection"), 1.0, 0)
    chk("O2 v2 rotation evasion=0", get("e7_evasion_adaptive_v2.json", "aggregate/session_rotation_evasion/2"), 0.0, 0)

    # ---------- W3a / W3b (threshold estimation / white-box) ----------
    chk("W3a intent-preserving evasion=0/40", get("e1_threshold_estimation.json", "n_intent_preserving_evasions"), 0, 0)
    chk("W3a queries to localise~9", get("e1_threshold_estimation.json", "mean_queries_to_localise"), 9.0, 0.1)
    chk("W3b eps.2 breaks 100%", get("e1_whitebox_probe.json", "success_rate_by_eps/0.2"), 1.0, 0)
    chk("W3b baseline max_risk=.85", get("e1_whitebox_probe.json", "baseline_mean_max_risk"), 0.85, 5e-3)

    # ---------- W4: extraction probe ----------
    chk("W4 extraction min acc=99.17%", get("e6_extraction_probe.json", "min_accuracy"), 0.9917, 1e-3)

    # ---------- O1 / O2 / O7: attacker-cost frontier & external ----------
    chk("O1 frontier levels=7", len(get("attacker_cost_frontier.json", "levels") or []), 7, 0)
    chk("O1 max API-only evasion=.0278", get("attacker_cost_frontier.json", "summary/max_evasion_api_only"), 0.0278, 1e-3)
    chk("O2 published baseline detection=.996", get("e1_strong_baseline.json", "detection_rate"), 0.996, 1e-3)
    chk("O7 external attack detection=.996", get("o7_external_generalization.json", "attack_detection_rate"), 0.996, 1e-3)
    chk("O7 external benign flagged@.5=.73", get("o7_external_generalization.json", "benign_flagged_at_0.5"), 0.73, 5e-3)
    chk("O7 in-distribution flagged@.5=.2565", get("o7_external_generalization.json", "in_distribution_benign_flagged_at_0.5"), 0.2565, 1e-3)
    chk("O7 max_risk AUC=0.994", get("o7_operating_point.json", "auc_max_risk"), 0.9942, 2e-3)
    cur = {round(c["thr"], 2): c for c in (get("o7_operating_point.json", "curve") or [])}
    chk("O7 thr.7 ext FP=.053", (cur.get(0.7) or {}).get("benign_FP"), 0.053, 5e-3)
    chk("O7 thr.7 ext recall=.970", (cur.get(0.7) or {}).get("attack_recall"), 0.970, 5e-3)

    # ---------- E5 / E6: attribution linkage & retrieval ----------
    chk("E6 linkage P=.102", get("e4_attribution_final.json", "test_metrics/pairwise_precision"), 0.1022, 1e-3)
    chk("E6 linkage R=.289", get("e4_attribution_final.json", "test_metrics/pairwise_recall"), 0.2889, 1e-3)
    chk("E5/E6 linkage F1=.151", get("e4_attribution_final.json", "test_metrics/pairwise_f1"), 0.151, 1e-3)
    chk("E6 embed retrieval top-1=.487", get("attribution_retrieval_v4.json", "results/embed_cos/top1"), 0.4867, 1e-3)
    chk("E6 fusion top-1=.567", get("attribution_retrieval_v4.json", "results/fusion_cos/top1"), 0.5667, 1e-3)
    chk("E6 fusion top-5=.807", get("attribution_retrieval_v4.json", "results/fusion_cos/top5"), 0.8067, 1e-3)
    chk("E6 supervised linker top-1=.493", get("attribution_retrieval_v4.json", "results/supervised/top1"), 0.4933, 1e-3)
    chk("E6 anchor 60 top-1=.35", get("anchor_scaling.json", "curve/anchors=30/mean_top1"), 0.3533, 1e-3)
    chk("E6 anchor 150 top-1=.25", get("anchor_scaling.json", "curve/anchors=120/mean_top1"), 0.2533, 1e-3)
    # ---------- E5: fingerprint ablation (embedding vs surface features) ----------
    chk("E5 embed linkage F1=.151", get("e5_ablation_fingerprint.json", "embed/test/F1"), 0.151, 1e-3)
    chk("E5 surface linkage F1=.057", get("e5_ablation_fingerprint.json", "surface/test/F1"), 0.0573, 5e-4)
    chk("E5 surface P=.031", get("e5_ablation_fingerprint.json", "surface/test/P"), 0.0314, 5e-4)
    chk("E5 surface R=.329", get("e5_ablation_fingerprint.json", "surface/test/R"), 0.3289, 5e-4)

    # ---------- E2b: second capture window ----------
    _w = (load("cross_distribution.json") or {}).get("capture_window_0901", {})
    chk("E2b 2nd-window trigger=25.6%",
        (_w.get("benign_over_0.7_hpc_trigger", 0) / _w.get("n_benign", 1)) if _w else None, 0.2558, 2e-3)

    # ---------- report ----------
    npass = nfail = nskip = 0
    rows = []
    for label, actual, expected, tol in CHECKS:
        if actual is None:
            status, nskip = "SKIP", nskip + 1
        elif close(actual, expected, tol):
            status, npass = "PASS", npass + 1
        else:
            status, nfail = "FAIL", nfail + 1
        rows.append({"check": label, "actual": actual, "expected": expected, "status": status})
        print(f"{status:4s} {label:38s} actual={actual}  expected={expected}")

    print(f"\nTOTAL {len(CHECKS)}  PASS {npass}  FAIL {nfail}  SKIP {nskip}")
    if "--json" in sys.argv:
        json.dump({"n": len(CHECKS), "pass": npass, "fail": nfail, "skip": nskip, "checks": rows},
                  open(os.path.join(RES, "reproduce_report.json"), "w"), indent=2)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
