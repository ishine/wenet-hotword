#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify how a hotword-correction pass changed the hypothesis.

Given two hypothesis files (`--before` produced *without* the correction
pathway, `--after` produced with it) and a reference transcript (`--ref`),
classify every utterance into one of four buckets and report aggregate
counts plus char-level deltas:

  unchanged   before_hyp == after_hyp                       (correction did not fire)
  fix         after has *lower* edit distance to ref        (genuine improvement)
  harm        after has *higher* edit distance to ref       (correction broke something)
  shuffle     edit distances are equal, but texts differ     (lateral change)

The `harm` bucket is the "对的词被纠错" (correctly-recognized text got
clobbered) case. Use `--verbose` to dump the regressing utterances.

Optional `--hotword-list` further splits `harm` into:
  - hotword_spurious : an `after` contains a hotword the ref does NOT
  - hotword_dropped  : `before` contained a hotword (matching ref) but
                       `after` no longer does
  - other_regression : neither of the above (typically punctuation /
                       non-hotword string drift)

Usage:
    python compute-correction-impact.py \
        --ref aishell_test/text \
        --before eval_runs/A_baseline.txt \
        --after  eval_runs/B_phoneme.txt \
        --hotword-list aishell_test/hotwords.txt \
        --verbose
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
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
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


def occurrences(text: str, term: str) -> int:
    if not term:
        return 0
    n, start = 0, 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            break
        n += 1
        start = idx + len(term)
    return n


def classify_harm(before: str, after: str, ref: str, hotwords):
    """Sub-classify a harm utterance via hotword occurrence diff vs ref."""
    spurious = []  # in after, not in ref, not in before
    dropped = []   # in before AND in ref, not in after
    for hw in hotwords:
        in_ref = occurrences(ref, hw)
        in_before = occurrences(before, hw)
        in_after = occurrences(after, hw)
        if in_after > in_ref and in_after > in_before:
            spurious.append(hw)
        elif in_before >= in_ref > 0 and in_after < in_ref:
            dropped.append(hw)
    if spurious:
        return "hotword_spurious", spurious
    if dropped:
        return "hotword_dropped", dropped
    return "other_regression", []


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True)
    p.add_argument("--before", required=True,
                   help="hyp without the correction pass (e.g. A_baseline.txt)")
    p.add_argument("--after", required=True,
                   help="hyp with the correction pass (e.g. D_confidence.txt)")
    p.add_argument("--hotword-list",
                   help="optional; enables hotword-aware harm sub-classification")
    p.add_argument("--verbose", action="store_true",
                   help="dump every harm utt with its before/after/ref")
    p.add_argument("--limit-verbose", type=int, default=20,
                   help="cap on verbose entries per category (default 20)")
    args = p.parse_args()

    ref = load_kv(args.ref)
    before = load_kv(args.before)
    after = load_kv(args.after)
    hotwords = load_hotwords(args.hotword_list) if args.hotword_list else []
    # longest-first so partial-name hotwords don't pre-empt their containers.
    hotwords_sorted = sorted(hotwords, key=len, reverse=True)

    n_unchanged = n_fix = n_harm = n_shuffle = 0
    chars_saved = chars_damaged = 0
    n_missing = 0
    harm_subcats = defaultdict(int)
    harm_examples = defaultdict(list)
    fix_examples = []

    for utt_id, ref_text in ref.items():
        b = before.get(utt_id)
        a = after.get(utt_id)
        if b is None or a is None:
            n_missing += 1
            continue

        if b == a:
            n_unchanged += 1
            continue

        ed_b = char_edit_distance(b, ref_text)
        ed_a = char_edit_distance(a, ref_text)
        delta = ed_a - ed_b
        if delta < 0:
            n_fix += 1
            chars_saved += -delta
            if len(fix_examples) < args.limit_verbose:
                fix_examples.append((utt_id, b, a, ref_text, delta))
        elif delta > 0:
            n_harm += 1
            chars_damaged += delta
            sub, evidence = (classify_harm(b, a, ref_text, hotwords_sorted)
                             if hotwords else ("other_regression", []))
            harm_subcats[sub] += 1
            if len(harm_examples[sub]) < args.limit_verbose:
                harm_examples[sub].append(
                    (utt_id, b, a, ref_text, delta, evidence)
                )
        else:
            n_shuffle += 1

    total = n_unchanged + n_fix + n_harm + n_shuffle
    print(f"#utts compared:       {total}")
    print(f"#utts missing in I/O: {n_missing}")
    print()
    pct = (lambda x: f"{x * 100 / total:6.2f}%" if total else "  n/a ")
    print(f"unchanged  : {n_unchanged:5d}  ({pct(n_unchanged)})  before == after")
    print(f"fix        : {n_fix:5d}  ({pct(n_fix)})  after closer to ref")
    print(f"harm       : {n_harm:5d}  ({pct(n_harm)})  after farther from ref")
    print(f"shuffle    : {n_shuffle:5d}  ({pct(n_shuffle)})  same distance, different text")
    print()
    print(f"chars saved by correction   : {chars_saved}")
    print(f"chars damaged by correction : {chars_damaged}")
    print(f"net chars saved             : {chars_saved - chars_damaged}")
    if total:
        net_ratio = (chars_saved - chars_damaged) / max(1, chars_saved + chars_damaged)
        print(f"fix/(fix+harm) char ratio   : {net_ratio:+.3f}")

    if hotwords:
        print()
        print("=== harm sub-classification ===")
        for k in ("hotword_spurious", "hotword_dropped", "other_regression"):
            v = harm_subcats.get(k, 0)
            print(f"  {k:<18s}: {v}")

    if args.verbose:
        print()
        print("=== fix examples ===")
        for utt, b, a, r, d in fix_examples:
            print(f"[{utt}] delta={d:+d}")
            print(f"  ref    : {r}")
            print(f"  before : {b}")
            print(f"  after  : {a}")
        for sub, examples in harm_examples.items():
            if not examples:
                continue
            print()
            print(f"=== harm examples ({sub}) ===")
            for utt, b, a, r, d, ev in examples:
                ev_str = f"  evidence: {ev}" if ev else ""
                print(f"[{utt}] delta={d:+d}{ev_str}")
                print(f"  ref    : {r}")
                print(f"  before : {b}")
                print(f"  after  : {a}")


if __name__ == "__main__":
    main()
