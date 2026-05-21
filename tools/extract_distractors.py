#!/usr/bin/env python3
"""Extract distractor (negative-sample) words from a text file.

These are high-frequency common words that appear in the corpus but should
NOT be boosted as hotwords. Adding them to the hotword list during autotune
forces the optimizer to pick conservative parameters — if bonus_weight or
fuzzy_threshold is too aggressive, these distractors get falsely recalled
and precision drops.

Filters:
  - token length 2–4 chars (short enough to be common, long enough to matter)
  - frequency >= min_count (default: 5)
  - NOT proper nouns (nr, ns, nt, nz) — we only want everyday words
  - NOT in the existing hotword list

Usage:
    python extract_distractors.py text.txt --hotwords hotwords.txt \
        -n 50 -o distractors.txt
"""

import argparse
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Extract distractor words")
    parser.add_argument("text", help="WeNet-style text file")
    parser.add_argument("--hotwords", help="Existing hotwords to exclude")
    parser.add_argument("-n", "--top-n", type=int, default=50, help="Keep top-N distractors")
    parser.add_argument("-c", "--min-count", type=int, default=5, help="Minimum occurrence count")
    parser.add_argument("-o", "--output", default="-", help="Output file")
    args = parser.parse_args()

    try:
        import jieba.posseg as pseg
    except ImportError:
        print("jieba is required. Install:  uv pip install jieba", file=sys.stderr)
        sys.exit(1)

    # Load existing hotwords to exclude
    existing = set()
    if args.hotwords:
        with open(args.hotwords, "r", encoding="utf-8") as f:
            existing = set(line.strip() for line in f if line.strip())

    # Count all words with POS tagging
    counter: Counter[str] = Counter()
    with open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            transcript = parts[1]
            for word, flag in pseg.cut(transcript):
                word = word.strip()
                # Keep only common nouns/verbs/adverbs (not proper nouns)
                if len(word) < 2 or len(word) > 4:
                    continue
                if word in existing:
                    continue
                # Skip proper nouns and foreign names
                if flag.startswith(("nr", "ns", "nt", "nz", "nrfg", "nrt", "eng")):
                    continue
                # Skip numbers, punctuation, single-char
                if flag.startswith(("m", "q", "x", "w", "u", "y", "p", "c", "d", "e")):
                    continue
                counter[word] += 1

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    kept = 0
    for word, count in counter.most_common():
        if count < args.min_count:
            break
        if kept >= args.top_n:
            break
        out.write(f"{word}\n")
        kept += 1

    if out is not sys.stdout:
        out.close()

    print(f"extracted {kept} distractors (min_count={args.min_count})", file=sys.stderr)


if __name__ == "__main__":
    main()
