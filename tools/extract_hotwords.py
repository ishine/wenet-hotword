#!/usr/bin/env python3
"""Extract hotword candidates from a WeNet-style text file via jieba TextRank.

Uses jieba.analyse.textrank with POS filtering to keep only noun categories
(person names, places, organizations, and general nouns).

Filters:
  - token length 2–6 chars
  - POS in (nr, ns, nt, nz, n, nrfg, nrt)
  - frequency >= min_count (default: 2)

Output is one hotword per line, sorted by TextRank score descending.
"""

import argparse
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Extract hotwords from text")
    parser.add_argument("text", help="WeNet-style text file")
    parser.add_argument("-n", "--top-n", type=int, default=200, help="Keep top-N hotwords")
    parser.add_argument("-c", "--min-count", type=int, default=2, help="Minimum occurrence count")
    parser.add_argument("-o", "--output", default="-", help="Output file (default: stdout)")
    args = parser.parse_args()

    try:
        import jieba.analyse
    except ImportError:
        print("jieba is required. Install:  uv pip install jieba", file=sys.stderr)
        sys.exit(1)

    # Read all transcripts into one string
    transcripts = []
    with open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) >= 2:
                transcripts.append(parts[1])

    full_text = " ".join(transcripts)

    # TextRank with POS filtering: keep noun categories only
    #   nr   = person name
    #   ns   = place name
    #   nt   = organization
    #   nz   = other proper noun
    #   n    = common noun
    #   nrfg = foreign person name
    #   nrt  = transliterated foreign name
    allowed_pos = ("nr", "ns", "nt", "nz", "n", "nrfg", "nrt")

    keywords = jieba.analyse.textrank(
        full_text,
        topK=args.top_n * 3,  # oversample for length/count filtering
        withWeight=False,
        allowPOS=allowed_pos,
    )

    # Count actual occurrences for filtering
    word_counts = Counter()
    for word in keywords:
        word = word.strip()
        if 2 <= len(word) <= 6:
            # Count how many transcripts contain this word
            count = sum(1 for t in transcripts if word in t)
            if count >= args.min_count:
                word_counts[word] = count

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    kept = 0
    for word, count in word_counts.most_common(args.top_n):
        out.write(f"{word}\n")
        kept += 1

    if out is not sys.stdout:
        out.close()

    print(f"extracted {kept} hotwords (min_count={args.min_count})", file=sys.stderr)


if __name__ == "__main__":
    main()
