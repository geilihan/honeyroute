#!/usr/bin/env python3
"""W1 helper -- flag the 17 duplicate rows (12 en + 5 zh) in the released
`e1_expanded_set_v2.json` benign list, keyed by (text, lang).

Non-destructive: keeps every row and marks the 2nd+ occurrences with
`"dup": true` and an ordinal `"dup_of"` index, so the released set stays
traceable while the release no longer double-counts duplicates.

Usage: python dedup_e1_set.py [--results DIR] [--apply]
  (without --apply it only reports; with --apply it rewrites the file)
"""
import argparse, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def find_results_dir(cli=None):
    for cand in (cli, os.environ.get("HONEYROUTE_OUT"),
                 os.path.join(HERE, "..", "_results"),
                 os.path.join(HERE, "..", "results")):
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    raise SystemExit("results dir not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    path = os.path.join(find_results_dir(a.results), "e1_expanded_set_v2.json")
    d = json.load(open(path, encoding="utf-8"))
    seen = defaultdict(list)
    for i, r in enumerate(d["benign"]):
        seen[(r["text"], r.get("lang"))].append(i)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    n_dup_rows = sum(len(v) - 1 for v in dups.values())
    by_lang = defaultdict(int)
    for k, v in dups.items():
        by_lang[k[1]] += len(v) - 1
    print(f"benign={len(d['benign'])} unique(text,lang)={len(seen)} "
          f"dup_keys={len(dups)} dup_rows={n_dup_rows} by_lang={dict(by_lang)}")
    if a.apply:
        for k, idxs in dups.items():
            for j in idxs[1:]:
                d["benign"][j]["dup"] = True
                d["benign"][j]["dup_of"] = idxs[0]
        d["dedup_note"] = (f"(text,lang) duplicates flagged with dup:true "
                           f"({n_dup_rows} rows: {dict(by_lang)}); "
                           f"unique benign = {len(seen)}")
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"applied -> {path}")


if __name__ == "__main__":
    main()
