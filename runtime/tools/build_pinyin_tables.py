#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the hanzi / pinyin / hanzi->pinyin tables required by the runtime
hotword pinyin context graph.

Given a model `units.txt` and a `cpp-pinyin/res/dict/mandarin/word.txt` dict,
emit four files in `--out-dir`:

  hanzi_unit.txt   <char> <id>            (FST SymbolTable text format)
  pinyin_unit.txt  <pinyin> <id>          (FST SymbolTable text format)
  hanzi_pinyin.txt <char> <py1> [py2 ...] (whitespace separated)
  context_pinyin.txt  -- optional, only when --hotwords is given.
       Each non-empty hotword in --hotwords becomes
         "<text> <py1> <py2> ... <score>"
       where pyN is the first listed pronunciation of each character.

`units.txt` is read as the source of truth for what characters the e2e model
covers. Multi-pronunciation characters keep every pronunciation found in
word.txt; the *first* pronunciation is used as a default when emitting
context_pinyin.txt. Tones (1-5 markers in word.txt) are stripped so that the
runtime PinyinMapper sees plain syllables.

Usage example:
    python build_pinyin_tables.py \
        --units /path/to/units.txt \
        --word-dict runtime/libtorch/build/bin/dict/mandarin/word.txt \
        --hotwords aishell_test/hotwords.txt \
        --score 3.0 \
        --out-dir aishell_test/pinyin_tables
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from typing import Dict, List, Sequence

_TRAILING_NUM_TONE_RE = re.compile(r"[1-5]$")


def strip_tone(syllable: str) -> str:
    """Normalize a pinyin syllable: drop trailing numeric tone, strip Unicode
    combining marks, lowercase. 'dèng' -> 'deng', 'pai4' -> 'pai', 'yī'->'yi'.
    Keep 'ü' as 'v' since cpp-pinyin and OpenFst SymbolTable both treat 'v' as
    the ASCII-safe stand-in (matches WeNet examples and Kaldi convention)."""
    s = syllable.strip()
    if not s:
        return s
    s = _TRAILING_NUM_TONE_RE.sub("", s)
    nfd = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in nfd if not unicodedata.combining(c))
    return no_marks.replace("ü", "v").replace("Ü", "v").lower()


def load_units(path: str) -> List[str]:
    chars = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            chars.append(token)
    return chars


def load_word_dict(path: str) -> Dict[str, List[str]]:
    """Parse cpp-pinyin word.txt. Format: '汉:py1,py2,...' or '汉:py' or
    multiple lines per char. Returns char -> list of unique pronunciations."""
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            ch, prons = line.split(":", 1)
            ch = ch.strip()
            if not ch:
                continue
            tokens = re.split(r"[,\s]+", prons.strip())
            tokens = [strip_tone(t) for t in tokens if t]
            if not tokens:
                continue
            bucket = out.setdefault(ch, [])
            for t in tokens:
                if t not in bucket:
                    bucket.append(t)
    return out


def write_symbol_table(path: str, symbols: Sequence[str], with_eps: bool = True):
    """Write an OpenFst SymbolTable-compatible text file. `<eps>` claims id 0
    to match `fst::SymbolTable::ReadText` expectations."""
    with open(path, "w", encoding="utf-8") as f:
        offset = 0
        if with_eps:
            f.write("<eps> 0\n")
            offset = 1
        for i, sym in enumerate(symbols):
            f.write(f"{sym} {i + offset}\n")


def build_pinyin_inventory(word_dict: Dict[str, List[str]]) -> List[str]:
    seen, out = set(), []
    for prons in word_dict.values():
        for p in prons:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def emit_context_pinyin(
    hotwords_path: str,
    out_path: str,
    word_dict: Dict[str, List[str]],
    score: float,
):
    """Build a `<text> <py1> ... <pyN> <score>` line per hotword. If a hotword
    contains any character without a pronunciation, drop it with a warning to
    stderr."""
    n_in = n_out = 0
    with open(hotwords_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            n_in += 1
            pinyins = []
            ok = True
            for ch in text:
                prons = word_dict.get(ch)
                if not prons:
                    print(f"[warn] no pinyin for '{ch}' in '{text}', skipping",
                          file=sys.stderr)
                    ok = False
                    break
                pinyins.append(prons[0])
            if not ok:
                continue
            fout.write(f"{text} {' '.join(pinyins)} {score:.4f}\n")
            n_out += 1
    print(f"[context-pinyin] wrote {n_out}/{n_in} hotwords to {out_path}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", required=True, help="model units.txt")
    ap.add_argument("--word-dict", required=True,
                    help="cpp-pinyin res/dict/mandarin/word.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hotwords",
                    help="optional hotword list to emit context_pinyin.txt")
    ap.add_argument("--score", type=float, default=3.0,
                    help="default per-hotword context_score in context_pinyin.txt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    units = load_units(args.units)
    word_dict = load_word_dict(args.word_dict)

    # The hanzi table for the context graph keeps only model-known Chinese
    # characters that also have a pronunciation in word.txt.
    hanzi = [ch for ch in units if len(ch) == 1 and ch in word_dict
             and "一" <= ch <= "鿿"]
    write_symbol_table(
        os.path.join(args.out_dir, "hanzi_unit.txt"), hanzi, with_eps=True
    )

    pinyin_inventory = build_pinyin_inventory(
        {ch: word_dict[ch] for ch in hanzi}
    )
    write_symbol_table(
        os.path.join(args.out_dir, "pinyin_unit.txt"),
        pinyin_inventory,
        with_eps=True,
    )

    hp_path = os.path.join(args.out_dir, "hanzi_pinyin.txt")
    with open(hp_path, "w", encoding="utf-8") as f:
        for ch in hanzi:
            prons = word_dict[ch]
            f.write(f"{ch} {' '.join(prons)}\n")

    print(f"[ok] {len(hanzi)} hanzi, {len(pinyin_inventory)} pinyin syllables",
          file=sys.stderr)

    if args.hotwords:
        emit_context_pinyin(
            args.hotwords,
            os.path.join(args.out_dir, "context_pinyin.txt"),
            word_dict,
            args.score,
        )


if __name__ == "__main__":
    main()
