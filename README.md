# HoneyRoute — Experiment Code and Results

Code and result data for **"HoneyRoute: Honeypot-Model Routing for
Adversarial LLM Serving"** ([arXiv:2609.08306](https://arxiv.org/abs/2609.08306)).

HoneyRoute is an inference-serving security layer that routes a frozen
risk-router's flagged requests away from the production LLM to a
honeypot model, preserving analyst-visible traceability while cutting
production token consumption under attack.

## Layout

- `scripts/` — all experiment scripts (E1–E7 + attribution, playbooks,
  judge-replication, adversarial-evasion runs), runnable as-is against a
  deployed HoneyRoute gate
- `results/` — the result JSONs reported in the paper (verbatim copies
  of the experiment outputs)
- `figures/` — the paper's data figures (F–T frontier, E7 trend,
  anchor-scaling) as PDFs

## Experiment → script → result map

| Paper block | Script(s) | Result JSON |
|---|---|---|
| E1 detection + baselines | `honeyroute_run_e1.py`, `honeyroute_run_e1b.py` | `e1_detection.json`, `e1_baselines.json` |
| E1 L1+L2 cascade | `honeyroute_run_e1casc.py` | `e1_cascade.json` |
| E2 fidelity (teacher judge) | `honeyroute_run_e2.py`, `honeyroute_run_e2v2.py` | `e2_faithfulness.json`, `e2_faithfulness_v2.json` |
| E2 fidelity, second judge | `run_e2_judge2.py`, `run_e2_judge2_par.py`, `run_cross_judge.py` | `e2_cross_judge*.json` |
| E2b selective hpC | `honeyroute_run_e2b.py`, `run_e2b_judge2.py` | `e2b_selective_hpc.json` |
| E3 cost/latency | (within E1/E6 runs) | `e1_detection.json`, `e6_stress_v3_real.json` |
| E4 attribution (pairwise) | `honeyroute_run_e4.py`, `honeyroute_run_e4b.py`, `honeyroute_run_e4c.py` | `e4_attribution*.json` |
| E4 positive loop | `honeyroute_run_e4loop.py`, `honeyroute_run_e4loop2.py`, `honeyroute_run_e4loop3.py`, `honeyroute_run_e4p.py`, `run_loop_gen0.py`, `run_loop_gen1.py` | `e4_positive_loop*.json`, `e4_pooled_generations.json`, `loop_*.json` |
| E5 ablation / absorption / stress | `honeyroute_run_e5.py`, `honeyroute_run_e5fix.py`, `honeyroute_run_e5stress.py` | `e5_*.json` |
| E6 real-payload floods | `honeyroute_run_e6.py`, `honeyroute_run_e6v2.py` | `e6_stress_v3_real.json`, `e6_ft_frontier.json` |
| E6 attribution retrieval + fusion | `honeyroute_run_attrv2.py`, `honeyroute_run_attrv3.py`, `honeyroute_run_attrv4.py`, `run_anchor_scale.py` | `attribution_*_v*.json`, `anchor_scaling.json` |
| E7 multi-turn soft escalation | `honeyroute_run_mt.py` | `multiturn_soft_escalation.json` |
| E7 human playbooks (Crescendo etc.) | `run_crescendo.py`, `run_playbooks_v2.py` | `crescendo_replay.json`, `playbooks_v2_replay.json` |
| E7 online trend rule | (analysis of E7 margins) | `online_trend_rule.json` |
| E7 adversarial evasion | `run_e7_adv.py`, `run_gate_evade.py` | `e7_adv_*.json` |
| Distribution shift | `run_dist_shift.py` | `cross_distribution.json` |
| E1 LG4-12B / ShieldGemma-2b baselines | `run_lg4_baseline.py`, `run_sg_baseline.py` | `e1_llamaguard4_baseline.json`, `e1_shieldgemma_baseline.json` |
| E1 expanded set + eval (1,103 samples) | `gen_e1_expanded.py`, `build_b2_set.py`, `run_e1_expanded_eval.py` | `e1_expanded_set_v2.json`, `e1_expanded_eval_v2.json` |
| E7 expanded (held-out splits) | `gen_e7_scripts.py`, `run_e7_expanded.py` | `e7_expanded_scripts.json`, `e7_expanded_eval.json`, `e7_expanded_final.json` |
| E7 rule-aware evasion | `run_b4_evasion.py` | `e7_evasion_robustness.json`, `b4_evasion_summary.json` |
| E1 zh benign calibration (W1) | `run_zh_calibration.py`, `dedup_e1_set.py` | `e1_zh_calibration.json` (calibrated branch also in `e1_expanded_eval_v2.json`) |
| E7 adaptive API-black-box evasion (W2) | `run_b4_adaptive.py` | `e7_evasion_adaptive.json` |
| E8 text-realisable white-box attack | `run_e8_whitebox_text.py` (+ `replay_e8.py`) | `e8_whitebox_text.json`, `e8_whitebox_text_gate.json` |
| E8b public-backbone-only transfer | `run_e8_whitebox_text.py` (E8_TARGET=surrogate) | `e8b_transfer.json`, `e8b_transfer_gate.json` |
| Public benign corpus for the surrogate (E8b) | (input data) | `dolly_benign.json` (databricks-dolly-15k sample) |
| E9 input-smoothing defense (D1) | `run_e9_l3_defense.py` | `e9_l3_defense.json` |
| Attacker-cost frontier (O1) | `agg_attacker_cost.py`, `plot_attacker_cost.py` | `attacker_cost_frontier.json` |
| E7 adaptive v2: multi-teacher + session/account rotation (O2) | `run_b4_adaptive_v2.py` | `e7_evasion_adaptive_v2.json` |
| Published strong-evasion baseline (O2) | `run_e1_strong_baseline.py` | `e1_strong_baseline.json` |
| External-corpus generalization: dolly-15k benign + JBB attacks (O7) | `run_o7_external_generalization.py` | `o7_external_generalization.json` |
| External-corpus recalibration curve (O7 ext) | `run_o7_external_calibration.py` | `o7_external_generalization.json` (recalibration_curve) |
| External operating point / AUC (O7 ext) | `run_o7_operating_point.py` | `o7_operating_point.json` |
| One-command offline reproduction (all paper numbers) | `reproduce_all.py` | `reproduce_report.json` |
| E2 judge-order robustness (W6) | `run_e2_judge_robustness.py` | `e2_judge_robustness.json` |
| E2b per-deployment threshold recipe (W7) | `run_e2b_threshold_recipe.py` | `e2b_threshold_recipe.json`, `e2b_benign_gate_scores.json` |
| E1 black-box threshold estimation (W3a) | `run_w3_threshold_estimation.py` | `e1_threshold_estimation.json` |
| E6 budget-limited extraction probe (W4) | `run_e6_extraction_probe.py` | `e6_extraction_probe.json` |
| E1 white-box gradient probe (W3b) | `run_e1_whitebox_probe.py` | `e1_whitebox_probe.json` |

Utilities shipped alongside the reproduction path (not paper experiments): `remote_run.py` + `_ssh.py` (upload/execute a script inside the guard container and pull its result JSON) and `sg_debug.py` / `sg_debug2.py` (ad-hoc ShieldGemma probe scratch).

`results/` additionally retains intermediate / partial JSONs from the experiment runs (`*_partial*.json`, attribution and loop intermediates, multi-teacher partials) for transparency; the paper's numbers come from the files listed in the table above.



Attack corpora: `results/jbb_attacks_uniq.json` (496 unique
JailbreakBench prompts), `results/crescendo_playbooks.json` (verbatim
Crescendo turns from PyRIT), `results/e7_adv_scripts.json` and
`results/loop_*` (generated adversarial/loop-training scripts).
Files named `*_partial.json` are incremental saves from long-running
experiments (kept for completeness; the corresponding final JSON is
authoritative). Additional per-experiment outputs (`e4_attribution*.json`,
`e2_judge2*.json`, `e2b_judge2*.json`, `e5_*.json`, `cross_judge.json`,
`legit_sec_research_fpr.json`, `mt_trend_analysis.json`,
`gate_evasion*.json`, `e7_adversarial.json`) are intermediate or
replication runs referenced in the paper's appendix.

## Reproduction

The scripts expect a deployed HoneyRoute stack (see paper §4 Setup):

- **Gate** at `http://127.0.0.1:8002` — `/gate`, `/gate/full`
  (with `session_id` for the session-risk tracker)
- **Production model** and **replica** served via vLLM
- **Teacher judge** — an OpenAI-compatible chat endpoint

Configure endpoints and paths via environment variables:

```bash
export GUARD_CODE_DIR=/path/to/gate/code        # pipeline/ package
export HONEYROUTE_OUT=/path/to/output/dir       # result JSONs
export BACKBONE_DIR=/path/to/frozen/backbone    # 0.8B router backbone
export SEEDS_DIR=/path/to/seed/corpora         # attack-seed data
export TEACHER_GATEWAY_URL=https://host/v1/chat/completions
export TEACHER_API_KEY=...
```

Then run any experiment with `python scripts/<script>.py`.

## Notes

- Internal deployment identifiers, trace URLs, model names, and API
  endpoints have been anonymized/redacted.
- The gate's frozen backbone and trained heads are described in the
  paper; the router training pipeline is not part of this release.
- Results are raw experiment outputs; percentages in the paper are
  derived from these files.

## License

MIT (code and figures). Result JSONs derived from JailbreakBench
artifacts follow their respective licenses.
