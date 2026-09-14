#!/usr/bin/env python3
"""Render the attacker-cost frontier (O1) as a vector PDF for the paper.

Reads ``_results/attacker_cost_frontier.json`` and writes
``figures/attacker_cost.pdf`` into the given output directories (default: both
submission dirs).  Deterministic, no title (the LaTeX caption carries it).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "_results")


def main(outdirs):
    data = json.load(open(os.path.join(RES, "attacker_cost_frontier.json"), encoding="utf-8"))
    lv = data["levels"]
    labels = [x["level"] for x in lv]
    ev = [100.0 * x["evasion"] for x in lv]
    colors = ["#4C72B0" if x["level"] != "L3" else "#C44E52" for x in lv]

    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    bars = ax.bar(range(len(lv)), ev, color=colors, width=0.62)
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("evasion rate (%)", fontsize=9)
    ax.set_ylim(0, 108)
    ax.tick_params(axis="y", labelsize=8)

    caps = {
        "L1": "decision-only",
        "L2a": "0/1 feedback",
        "L2b": "score, 1-turn",
        "L2c": "score, multi-turn",
        "L2d": "score, +rotation",
        "PUB": "published JBB",
        "L3": "white-box",
    }
    for i, (x, b) in enumerate(zip(lv, bars)):
        h = b.get_height()
        ax.text(i, h + 2, f"{h:.1f}", ha="center", fontsize=8)
        ax.text(i, -13, caps.get(x["level"], ""), ha="center", fontsize=6.5, color="#444444")
        if x.get("cost_queries"):
            ax.text(i, h / 2, f"~{x['cost_queries']:g}q", ha="center", va="center",
                    fontsize=6.5, color="white")
    ax.text(6, 100.5, "not text-realisable", ha="center", fontsize=7, color="#C44E52")
    ax.annotate("", xy=(5.6, 99), xytext=(6.4, 99),
                arrowprops=dict(arrowstyle="-", color="#C44E52", lw=0.8))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for d in outdirs:
        dst = os.path.join(d, "figures", "attacker_cost.pdf")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fig.savefig(dst)
        print("wrote", dst)


if __name__ == "__main__":
    outs = sys.argv[1:] or ["sp_submit", "arxiv_submit"]
    main([os.path.join(HERE, "..", o) for o in outs])
