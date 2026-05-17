#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesize a streaming-style wav.scp from an IID test set.

The LRU `HotwordCache` is designed for sessions where the same hotword
recurs over consecutive utterances (a user dictates several lines all
mentioning 佟健, 庞清, etc.). On IID benchmarks like AISHELL-hotwords most
hotwords appear in only one utterance, so the cache's
`activate_threshold = 2` is never satisfied and the cache is a no-op.

This script reshuffles the existing test set into a synthetic streaming
order: utterances containing the same primary hotword are placed in a run,
groups are sorted by recurrence count (most-recurring first). The result is
not a new dataset — it's the same 235 audio clips in a deliberately
hotword-clustered order, designed to surface whatever recall gain the
cache *would* provide in a real streaming scenario.

To make the experiment meaningful, run `decoder_main` with `thread_num=1`
afterwards; with multiple decoder threads each holds an independent
`AsrDecoder` (and an independent cache), which fragments the streaming
signal across threads.

Usage:
    python3 tools/build_streaming_scp.py \
        --ref aishell_test/text \
        --scp aishell_test/wav.scp \
        --hotwords aishell_test/hotwords.txt \
        --out-scp aishell_test/wav.stream.scp \
        --out-text aishell_test/text.stream
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Optional


def load_kv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, _, rest = line.partition(" ")
            out[uid] = rest
    return out


def load_hotwords(path: str) -> List[str]:
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            hw = line.strip()
            if hw and hw not in seen:
                seen.add(hw)
                out.append(hw)
    # longest-first so that 高桥大辅 wins over a substring 高桥
    return sorted(out, key=len, reverse=True)


def primary_hotword(text: str, hotwords: List[str]) -> Optional[str]:
    for hw in hotwords:
        if hw in text:
            return hw
    return None


def reorder(uids: List[str], texts: Dict[str, str],
            hotwords: List[str]) -> List[str]:
    groups: Dict[Optional[str], List[str]] = defaultdict(list)
    for uid in uids:
        groups[primary_hotword(texts[uid], hotwords)].append(uid)

    # most-recurring hotword groups first; single-utt hotwords last in
    # alphabetical order; no-hotword utts trailing.
    ordered: List[str] = []
    multi = sorted([(hw, lst) for hw, lst in groups.items()
                    if hw is not None and len(lst) >= 2],
                   key=lambda kv: -len(kv[1]))
    singles = sorted([(hw, lst) for hw, lst in groups.items()
                      if hw is not None and len(lst) == 1])
    no_hw = groups.get(None, [])
    for hw, lst in multi:
        ordered.extend(lst)
    for hw, lst in singles:
        ordered.extend(lst)
    ordered.extend(no_hw)
    return ordered


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference text file")
    p.add_argument("--scp", required=True, help="wav.scp file")
    p.add_argument("--hotwords", required=True)
    p.add_argument("--out-scp", required=True)
    p.add_argument("--out-text", required=True)
    args = p.parse_args()

    texts = load_kv(args.ref)
    scp = load_kv(args.scp)
    hotwords = load_hotwords(args.hotwords)

    common = [uid for uid in scp if uid in texts]
    if len(common) < len(scp):
        missing = set(scp) - set(texts)
        print(f"[warn] {len(missing)} utts in scp without ref text — skipped",
              file=sys.stderr)

    ordered = reorder(common, texts, hotwords)

    runs = []
    cur_hw = primary_hotword(texts[ordered[0]], hotwords) if ordered else None
    cur_len = 0
    for uid in ordered:
        hw = primary_hotword(texts[uid], hotwords)
        if hw == cur_hw:
            cur_len += 1
        else:
            if cur_len:
                runs.append((cur_hw, cur_len))
            cur_hw, cur_len = hw, 1
    if cur_len:
        runs.append((cur_hw, cur_len))

    print(f"[info] reordered {len(ordered)} utts into "
          f"{len([r for r in runs if r[1] >= 2])} multi-utt runs + "
          f"{len([r for r in runs if r[1] == 1])} singletons")
    print("[info] top runs (hotword × consecutive utts):")
    for hw, n in sorted(runs, key=lambda r: -r[1])[:6]:
        print(f"         {hw or '<no-hotword>'}: {n}")

    with open(args.out_scp, "w", encoding="utf-8") as f:
        for uid in ordered:
            f.write(f"{uid} {scp[uid]}\n")
    with open(args.out_text, "w", encoding="utf-8") as f:
        for uid in ordered:
            f.write(f"{uid} {texts[uid]}\n")

    print(f"[ok] wrote {args.out_scp} and {args.out_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
