#!/usr/bin/env python3
"""Summarize results from all 4 autotune modes."""

import json
import os
import re
from glob import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "runtime/libtorch/configs")

def parse_eval(path):
    if not os.path.exists(path):
        return None
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                try:
                    data[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return data

def parse_pareto(path):
    if not os.path.exists(path):
        return None
    trials = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))
    return trials

def main():
    modes = [
        ("激进", "aggressive", "recall↑ + CER↓", "187 hotwords"),
        ("平衡", "balanced", "F1↑ + CER↓", "187 hotwords"),
        ("保守", "conservative", "F1↑ + CER↓", "349 hotwords (+distractors)"),
        ("最保守", "ultra", "F1↑ + CER↓ + Precision↑", "349 hotwords (+distractors)"),
    ]

    print("=" * 100)
    print(f"{'模式':>8} {'Tune目标':>25} {'热词表':>30} {'Tune R/F1':>10} {'Tune CER':>10} {'Hold R':>10} {'Hold CER':>10} {'Hold P':>10} {'Hold F1':>10}")
    print("-" * 100)

    for name_cn, key, obj, hw_desc in modes:
        eval_path = os.path.join(CONFIG_DIR, f"{key}.eval.txt")
        pareto_path = os.path.join(CONFIG_DIR, f"{key}.pareto.jsonl")
        tuned_yaml = os.path.join(CONFIG_DIR, f"{key}.tuned.yaml")

        eval_data = parse_eval(eval_path)
        trials = parse_pareto(pareto_path)

        if not eval_data or not trials:
            print(f"{name_cn:>8} {obj:>25} {hw_desc:>30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        # Find the selected trial (the one referenced in eval.txt header)
        best = trials[0] if trials else {}
        for t in trials:
            if "holdout" in t and t.get("values"):
                best = t
                break

        tune_vals = best.get("values", [None])
        hold = best.get("holdout", {})

        if len(tune_vals) == 3 and all(v is not None for v in tune_vals):  # ultra mode
            tune_f1, tune_cer, tune_prec = tune_vals
            tune_hdr = f"{tune_f1:.1f}%"
        elif len(tune_vals) >= 2 and all(v is not None for v in tune_vals[:2]):
            tune_primary, tune_cer = tune_vals[:2]
            tune_hdr = f"{tune_primary:.1f}%"
        else:
            tune_hdr = "N/A"
            tune_cer = 0

        print(f"{name_cn:>8} {obj:>25} {hw_desc:>30} "
              f"{tune_hdr:>10} {tune_cer:>9.1f}% "
              f"{hold.get('recall', 0):>9.1f}% {hold.get('cer', 0):>9.1f}% "
              f"{hold.get('precision', 0):>9.1f}% {hold.get('f1', 0):>9.1f}%")

    print("=" * 100)
    print("\n详细结果文件:")
    for _, key, _, _ in modes:
        print(f"  {key:12s}: tuned={key}.tuned.yaml  eval={key}.eval.txt  pareto={key}.pareto.jsonl")

if __name__ == "__main__":
    main()
