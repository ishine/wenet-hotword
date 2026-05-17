#!/usr/bin/env python3
"""Extract a wenet-style test set from a HuggingFace AISHELL-1 parquet shard.

`AudioLLMs/aishell_1_zh_test` packs the AISHELL-1 test split as parquet rows
with audio embedded as RIFF/WAV bytes under `context.bytes`. This script
splits one shard into two partitions:

  no_hw/      utterances whose reference contains NO hotword (substring match
              against `hotwords.txt`). Subsampled to `--subsample` rows
              (deterministic with `--seed`).
  has_hw/     utterances whose reference contains ≥1 hotword.

Each partition gets `wav.scp`, `text`, and a `wavs/` subdir of extracted
.wav files. Utt-ids are `<prefix>_<row_idx_in_shard>`. The shard row index
is stable for a given parquet file, so re-running the script reproduces the
same split.

`hotwords.txt` is NOT copied — it is a hotword-list property, not a test-set
property. Symlink it from the original test set after extraction:

  ln -s ../aishell_test/hotwords.txt    <out_dir>/no_hw/hotwords.txt

Used to build the H_oov and I_indep robustness ablations
(see `runtime/libtorch/eval_runs/HOTWORD_EVAL.md`).

Usage:
  tools/extract_aishell1_parquet.py \\
    --parquet   aishell1_hf_raw/data/test-00000-of-00003.parquet \\
    --hotwords  aishell_test/hotwords.txt \\
    --out-dir   . \\
    --no-hw-prefix     aishell1_oov_test \\
    --has-hw-prefix    aishell1_indep_hotword \\
    --subsample 2000 \\
    --seed      17
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--hotwords", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--no-hw-prefix", default="aishell1_oov_test")
    ap.add_argument("--has-hw-prefix", default="aishell1_indep_hotword")
    ap.add_argument("--subsample", type=int, default=2000, help="0 = keep all eligible no-hotword utts")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    hotwords = [h.strip() for h in args.hotwords.read_text(encoding="utf-8").splitlines() if h.strip()]
    print(f"[hotwords] {len(hotwords)} entries from {args.hotwords}")

    pf = pq.ParquetFile(str(args.parquet))
    print(f"[parquet] {pf.metadata.num_rows} rows in {args.parquet.name}")

    no_hw: list[tuple[int, str, bytes]] = []
    has_hw: list[tuple[int, str, bytes]] = []
    idx = 0
    for batch in pf.iter_batches(batch_size=256):
        answers = batch.column("answer").to_pylist()
        contexts = batch.column("context").to_pylist()
        for ans, ctx in zip(answers, contexts):
            rec = (idx, ans, ctx["bytes"])
            (has_hw if any(h in ans for h in hotwords) else no_hw).append(rec)
            idx += 1
    print(f"[partition] no_hw={len(no_hw)}  has_hw={len(has_hw)}")

    if args.subsample > 0 and args.subsample < len(no_hw):
        rng = random.Random(args.seed)
        no_hw = rng.sample(no_hw, args.subsample)
        print(f"[subsample] kept {len(no_hw)} no-hw rows (seed={args.seed})")

    write_partition(args.out_dir / args.no_hw_prefix, no_hw, args.no_hw_prefix)
    write_partition(args.out_dir / args.has_hw_prefix, has_hw, args.has_hw_prefix)
    return 0


def write_partition(out_dir: Path, rows: list[tuple[int, str, bytes]], utt_prefix: str) -> None:
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r[0])

    scp_lines: list[str] = []
    text_lines: list[str] = []
    for row_idx, ans, audio in rows:
        utt = f"{utt_prefix}_{row_idx:06d}"
        wav_path = wav_dir / f"{utt}.wav"
        wav_path.write_bytes(audio)
        scp_lines.append(f"{utt} {wav_path.resolve()}\n")
        text_lines.append(f"{utt} {ans}\n")

    (out_dir / "wav.scp").write_text("".join(scp_lines), encoding="utf-8")
    (out_dir / "text").write_text("".join(text_lines), encoding="utf-8")
    print(f"[wrote] {out_dir/'wav.scp'} ({len(rows)} entries)")


if __name__ == "__main__":
    sys.exit(main())
