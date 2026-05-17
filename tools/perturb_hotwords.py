#!/usr/bin/env python3
"""Generate perturbed hotword lists for robustness ablations.

Produces two variants from a base hotword list and a reference transcript:

  noisy:    base + N decoys (3-char Chinese strings that NEVER appear as any
            3-char substring of the reference). Stresses the corrector's
            precision when the user supplies extra junk hotwords.

  partial:  top-K hotwords by occurrence count in the reference. Stresses
            the corrector's recall when the user-supplied list is incomplete.

Output goes to <out-dir>/{noisy,partial}/hotwords.txt + a small stats file.

Usage:
  tools/perturb_hotwords.py \
    --ref         /path/to/text \
    --hotwords    /path/to/hotwords.txt \
    --decoys      50 \
    --top-k       30 \
    --out-dir     /path/to/scenarios \
    --seed        17
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SURNAMES = (
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐韩冯邓曹彭曾"
    "肖田董袁潘蔡蒋余沈程毛叶魏吕苏卢丁姚谭廖邹熊金陆郝邵孔白崔康毛"
)
GIVEN = (
    "明伟强磊洋勇杰刚军波涛斌健龙鹏飞鑫鹤腾凯轩浩然博睿"
    "丽芳娟敏静莉婷雪艳琳玉芬秀霞月雯萍梅妍娜佳"
    "天宇昊阳晨晖朗清远逸鸿志诚信德义思智仁安平和"
)
PLACE_PREFIX = "新东西南北中上下安永泰宁福兴临大长广济丰富兴吉康永祥"
PLACE_SUFFIX = "州市城镇川山河谷湾湖港岛峰原口村桥岭洲屯洞站林泉桥湾"


def occupied_trigrams(ref_path: Path) -> set[str]:
    occupied: set[str] = set()
    with ref_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) < 2:
                continue
            text = parts[1]
            for i in range(len(text) - 2):
                occupied.add(text[i : i + 3])
    return occupied


def gen_decoys(n: int, occupied: set[str], existing: set[str], rng: random.Random) -> list[str]:
    out: list[str] = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        if rng.random() < 0.5:
            cand = rng.choice(SURNAMES) + rng.choice(GIVEN) + rng.choice(GIVEN)
        else:
            cand = rng.choice(PLACE_PREFIX) + rng.choice(PLACE_SUFFIX) + rng.choice(PLACE_SUFFIX)
        if cand in occupied or cand in existing or cand in out:
            continue
        out.append(cand)
    if len(out) < n:
        raise SystemExit(f"only generated {len(out)} / {n} decoys after {tries} tries")
    return out


def hotword_counts(ref_path: Path, hotwords: list[str]) -> Counter[str]:
    refs: list[str] = []
    with ref_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) >= 2:
                refs.append(parts[1])
    counts: Counter[str] = Counter()
    for hw in hotwords:
        counts[hw] = sum(text.count(hw) for text in refs)
    return counts


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--hotwords", required=True, type=Path)
    ap.add_argument("--decoys", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = [h.strip() for h in args.hotwords.read_text(encoding="utf-8").splitlines() if h.strip()]

    occupied = occupied_trigrams(args.ref)
    decoys = gen_decoys(args.decoys, occupied, set(base), rng)

    counts = hotword_counts(args.ref, base)
    top_k = [h for h, _ in counts.most_common(args.top_k)]

    # noisy = base ∪ decoys, original order then decoys (matches a realistic "user appended junk").
    noisy_path = args.out_dir / "noisy" / "hotwords.txt"
    partial_path = args.out_dir / "partial" / "hotwords.txt"
    write_lines(noisy_path, base + decoys)
    write_lines(partial_path, top_k)

    # Drop a small stats file for the eval write-up.
    stats = args.out_dir / "perturb_stats.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with stats.open("w", encoding="utf-8") as f:
        f.write(f"base hotwords:     {len(base)}\n")
        f.write(f"decoys added:      {len(decoys)}  (sample: {' '.join(decoys[:5])})\n")
        f.write(f"top-{args.top_k} kept:        {len(top_k)}\n")
        f.write("\ntop-{} by ref count:\n".format(args.top_k))
        for h, c in counts.most_common(args.top_k):
            f.write(f"  {h:<10} {c}\n")
        zero_count = sum(1 for h in base if counts[h] == 0)
        f.write(f"\nbase hotwords with 0 ref hits: {zero_count}\n")

    print(f"wrote {noisy_path}  ({len(base)} + {len(decoys)} entries)")
    print(f"wrote {partial_path}  (top-{args.top_k} entries)")
    print(f"wrote {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
