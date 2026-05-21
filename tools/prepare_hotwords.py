#!/usr/bin/env python3
"""Prepare hotword lists for benchmark evaluation.

Usage:
    # Extract 500 hotwords from test corpus
    python3 tools/prepare_hotwords.py \
        ~/userspace/wenet/aishell2_eval/test1000/text \
        -o ~/userspace/wenet/aishell2_eval/test1000/hotwords_500.txt

    # Filter to hard-case subset (requires baseline hyp)
    python3 tools/prepare_hotwords.py \
        ~/userspace/wenet/aishell2_eval/test1000/text \
        --baseline-hyp /tmp/test1000_baseline_hyp.txt \
        --filter-hard \
        -o ~/userspace/wenet/aishell2_eval/test1000/hotwords.txt
"""

import argparse
import sys
from collections import Counter


def extract_hotwords(text_path: str, top_n: int = 500, min_count: int = 2) -> list:
    """Extract hotwords from text using jieba TextRank + POS filtering."""
    try:
        import jieba.analyse
    except ImportError:
        print("jieba is required. Install:  uv pip install jieba", file=sys.stderr)
        sys.exit(1)

    transcripts = []
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) >= 2:
                transcripts.append(parts[1])

    full_text = " ".join(transcripts)
    allowed_pos = ("nr", "ns", "nt", "nz", "n", "nrfg", "nrt")

    keywords = jieba.analyse.textrank(
        full_text, topK=top_n * 3, withWeight=False, allowPOS=allowed_pos)

    word_counts = Counter()
    for word in keywords:
        word = word.strip()
        if 2 <= len(word) <= 6:
            count = sum(1 for t in transcripts if word in t)
            if count >= min_count:
                word_counts[word] = count

    hotwords = []
    for word, count in word_counts.most_common(top_n):
        hotwords.append(word)
    return hotwords


def filter_hard_case(hotwords: list, text_path: str, baseline_hyp_path: str,
                     threshold: float = 0.90) -> list:
    """Filter hotwords to hard-case subset (baseline recall >= threshold are dropped)."""
    # Build ref/hyp dicts
    ref = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                ref[parts[0]] = parts[1]
            else:
                ref[parts[0]] = ""

    hyp = {}
    with open(baseline_hyp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                hyp[parts[0]] = parts[1]
            else:
                hyp[parts[0]] = ""

    # Count per-hotword occurrences
    hotwords_sorted = sorted(hotwords, key=len, reverse=True)
    per_hw = {}
    for utt_id, ref_text in ref.items():
        hyp_text = hyp.get(utt_id, "")
        ref_rem = ref_text
        hyp_rem = hyp_text
        for hw in hotwords_sorted:
            kref = ref_rem.count(hw)
            khyp = hyp_rem.count(hw)
            if kref == 0 and khyp == 0:
                continue
            tp = min(kref, khyp)
            per_hw[hw] = per_hw.get(hw, {"ref": 0, "tp": 0})
            per_hw[hw]["ref"] += kref
            per_hw[hw]["tp"] += tp
            ref_rem = ref_rem.replace(hw, " " * len(hw))
            hyp_rem = hyp_rem.replace(hw, " " * len(hw))

    # Filter: keep recall < threshold
    hard = []
    for hw in hotwords:
        stats = per_hw.get(hw, {"ref": 0, "tp": 0})
        if stats["ref"] == 0:
            continue
        recall = stats["tp"] / stats["ref"]
        if recall < threshold:
            hard.append(hw)

    return hard


def main():
    p = argparse.ArgumentParser(description="Prepare hotword lists")
    p.add_argument("text", help="Reference text file (Kaldi/WeNet format)")
    p.add_argument("-o", "--output", required=True, help="Output hotword file")
    p.add_argument("-n", "--top-n", type=int, default=500, help="Number of hotwords to extract")
    p.add_argument("-c", "--min-count", type=int, default=2, help="Minimum occurrence count")
    p.add_argument("--baseline-hyp", help="Baseline hypothesis for hard-case filtering")
    p.add_argument("--filter-hard", action="store_true",
                    help="Filter to hard-case subset (requires --baseline-hyp)")
    p.add_argument("--threshold", type=float, default=0.90,
                    help="Baseline recall threshold for hard-case (drop if >= threshold)")
    args = p.parse_args()

    hotwords = extract_hotwords(args.text, args.top_n, args.min_count)
    print(f"Extracted {len(hotwords)} hotwords from {args.text}", file=sys.stderr)

    if args.filter_hard:
        if not args.baseline_hyp:
            print("--filter-hard requires --baseline-hyp", file=sys.stderr)
            sys.exit(1)
        hotwords = filter_hard_case(hotwords, args.text, args.baseline_hyp, args.threshold)
        print(f"Filtered to {len(hotwords)} hard-case hotwords (recall < {args.threshold})",
              file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for hw in hotwords:
            f.write(f"{hw}\n")
    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
