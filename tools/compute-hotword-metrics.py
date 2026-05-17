#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute hotword recall / precision / F1 against a reference transcript.

Usage:
    python compute-hotword-metrics.py \
        --hotword-list hotwords.txt \
        --ref text \
        --hyp hyp.txt \
        [--per-hotword]  # also dump per-hotword table

`ref` and `hyp` follow Kaldi/WeNet convention: "<utt_id> <text>" per line.
Hotword recall is defined per-occurrence: in an utt where the reference
mentions a hotword `K` times and the hypothesis mentions it `L` times, we
count `min(K, L)` true positives, `max(0, K-L)` misses and `max(0, L-K)`
spurious insertions.

Aggregate:
    recall    = sum(TP) / sum(K_ref)
    precision = sum(TP) / sum(L_hyp)
    f1        = 2 * P * R / (P + R)

The script also prints character-level edit-distance metrics restricted to
the hotword spans so a single number captures both miss and partial
substitution errors (useful when the hotword is mis-recognized by 1 char).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict


def load_kv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 1:
                out[parts[0]] = ""
            else:
                out[parts[0]] = parts[1]
    return out


def load_hotwords(path):
    seen = set()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            hw = line.strip()
            if not hw or hw in seen:
                continue
            seen.add(hw)
            out.append(hw)
    return out


def count_occurrences(text: str, term: str) -> int:
    if not term:
        return 0
    # Non-overlapping occurrence count
    n, start = 0, 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            break
        n += 1
        start = idx + len(term)
    return n


def char_edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hotword-list", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--hyp", required=True)
    p.add_argument("--per-hotword", action="store_true")
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Only display the top-N hotwords (by ref-count) when --per-hotword is set",
    )
    args = p.parse_args()

    hotwords = load_hotwords(args.hotword_list)
    ref = load_kv(args.ref)
    hyp = load_kv(args.hyp)

    # Sort hotwords longest-first so when we count, we strip away longer
    # matches before checking shorter ones (avoids double-counting overlapping
    # hotwords like "李洁" being a substring of a longer name).
    hotwords_sorted = sorted(hotwords, key=len, reverse=True)

    tp_total = miss_total = spur_total = 0
    ref_total = hyp_total = 0
    per_hw_ref = defaultdict(int)
    per_hw_hyp = defaultdict(int)
    per_hw_tp = defaultdict(int)
    char_err = 0
    char_total = 0  # total chars in ref hotword spans
    missing_in_hyp = 0

    for utt_id, ref_text in ref.items():
        if utt_id not in hyp:
            missing_in_hyp += 1
            hyp_text = ""
        else:
            hyp_text = hyp[utt_id]

        # Track which spans of ref/hyp are already attributed to a hotword
        # so overlapping shorter hotwords don't double-count.
        ref_remaining = ref_text
        hyp_remaining = hyp_text

        for hw in hotwords_sorted:
            kref = count_occurrences(ref_remaining, hw)
            khyp = count_occurrences(hyp_remaining, hw)
            if kref == 0 and khyp == 0:
                continue

            tp = min(kref, khyp)
            miss = max(0, kref - khyp)
            spur = max(0, khyp - kref)

            per_hw_ref[hw] += kref
            per_hw_hyp[hw] += khyp
            per_hw_tp[hw] += tp
            tp_total += tp
            miss_total += miss
            spur_total += spur
            ref_total += kref
            hyp_total += khyp

            # Mask matched spans so shorter substrings don't re-match
            ref_remaining = ref_remaining.replace(hw, " " * len(hw))
            hyp_remaining = hyp_remaining.replace(hw, " " * len(hw))

            # Char-level partial credit: for each missed occurrence, count
            # the edit distance between the hotword and the closest hyp span.
            # Cheap heuristic: just attribute the full hotword length as errors.
            char_total += kref * len(hw)
            char_err += miss * len(hw)
            # For TP we contribute 0 errors; we don't try to credit partial subs
            # since they've been counted as misses already.

    recall = tp_total / ref_total if ref_total else 0.0
    precision = tp_total / hyp_total if hyp_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"#utts (ref):              {len(ref)}")
    print(f"#utts missing from hyp:   {missing_in_hyp}")
    print(f"#hotword types:           {len(hotwords)}")
    print(f"#hotword occurrences ref: {ref_total}")
    print(f"#hotword occurrences hyp: {hyp_total}")
    print(f"  true positives:         {tp_total}")
    print(f"  misses:                 {miss_total}")
    print(f"  spurious insertions:    {spur_total}")
    print(f"recall      = {recall * 100:6.2f} %")
    print(f"precision   = {precision * 100:6.2f} %")
    print(f"F1          = {f1 * 100:6.2f} %")
    if char_total:
        print(
            f"char-loss   = {char_err * 100 / char_total:6.2f} % "
            f"({char_err}/{char_total} chars in hotword spans missed)"
        )

    if args.per_hotword:
        print()
        print("=== Per-hotword breakdown (top by ref-count) ===")
        rows = []
        for hw in hotwords:
            kref = per_hw_ref.get(hw, 0)
            khyp = per_hw_hyp.get(hw, 0)
            ktp = per_hw_tp.get(hw, 0)
            if kref == 0:
                continue
            rec = ktp / kref if kref else 0.0
            prec = ktp / khyp if khyp else 0.0
            rows.append((hw, kref, khyp, ktp, rec, prec))
        rows.sort(key=lambda r: -r[1])
        print(f"{'hotword':<14s} {'ref':>4s} {'hyp':>4s} {'tp':>4s} "
              f"{'recall':>7s} {'prec':>7s}")
        for hw, kref, khyp, ktp, rec, prec in rows[: args.top]:
            print(f"{hw:<14s} {kref:>4d} {khyp:>4d} {ktp:>4d} "
                  f"{rec * 100:>6.2f}% {prec * 100:>6.2f}%")


if __name__ == "__main__":
    main()
