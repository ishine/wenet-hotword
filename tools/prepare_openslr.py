#!/usr/bin/env python3
"""Convert ST-CMDS-20170001_1-OS tree to WeNet-style wav.scp + text.

ST-CMDS ships audio files with adjacent transcript files:
    20170001P00001A0001.wav
    20170001P00001A0001.wav.trn   (or .txt)

This script walks the tree, pairs them, and emits:
    <out-dir>/wav.scp   # <utt-id> <absolute-wav-path>
    <out-dir>/text      # <utt-id> <hanzi-string-without-spaces>
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Prepare ST-CMDS for WeNet decoder_main")
    parser.add_argument("src_dir", help="Path to ST-CMDS-20170001_1-OS root")
    parser.add_argument("out_dir", help="Output directory for wav.scp and text")
    args = parser.parse_args()

    src = os.path.abspath(args.src_dir)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)

    suffixes = (".wav.trn", ".txt")
    n = 0

    with open(os.path.join(out, "wav.scp"), "w", encoding="utf-8") as scp, \
         open(os.path.join(out, "text"), "w", encoding="utf-8") as txt:
        for root, _dirs, files in os.walk(src):
            for fname in sorted(files):
                if not fname.endswith(".wav"):
                    continue
                wav_path = os.path.join(root, fname)
                utt = os.path.splitext(fname)[0]

                trn_path = None
                for sfx in suffixes:
                    candidate = wav_path + sfx
                    if os.path.isfile(candidate):
                        trn_path = candidate
                        break

                if trn_path is None:
                    print(f"warn: no transcript for {wav_path}", file=sys.stderr)
                    continue

                with open(trn_path, "r", encoding="utf-8") as f:
                    label = f.read().strip().replace(" ", "")

                scp.write(f"{utt} {wav_path}\n")
                txt.write(f"{utt} {label}\n")
                n += 1

    print(f"wrote {n} utterances to {out}")


if __name__ == "__main__":
    main()
