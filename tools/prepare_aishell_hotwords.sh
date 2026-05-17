#!/usr/bin/env bash
# Fetch ModelScope `speech_asr/speech_asr_aishell1_hotwords_testsets` and emit
# the WeNet-style layout decoder_main and run_ablations.sh expect:
#
#   <out-dir>/
#   ├── wav.scp        # <utt-id> <absolute-wav-path>
#   ├── text           # <utt-id> <hanzi-string-without-spaces>
#   └── hotwords.txt   # one hotword per line
#
# The ModelScope dataset splits its content across two stores:
#   - git repo:  README.md, hotword.txt, .csv metadata          → modelscope CLI
#   - OSS blob:  speech_asr_aishell_hotwords_testsets.zip (wav) → signed URL
# This script handles both.
#
# Usage:
#   tools/prepare_aishell_hotwords.sh <out-dir>
#
# Example:
#   tools/prepare_aishell_hotwords.sh ~/userspace/wenet/aishell_test

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <out-dir>" >&2
  exit 2
fi

OUT="$1"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
STAGE="$OUT/.stage"
mkdir -p "$STAGE"

DATASET="speech_asr/speech_asr_aishell1_hotwords_testsets"

# 1. Metadata files (CSV + hotword list) via modelscope CLI.
echo "[1/4] Fetching metadata via modelscope CLI ..." >&2
modelscope download --dataset "$DATASET" --repo-type dataset \
  --include '*' --local_dir "$STAGE" >/dev/null

CSV="$STAGE/speech_asr_aishell_hotwords_testsets.csv"
HOT="$STAGE/hotword.txt"
[[ -f "$CSV" ]] || { echo "missing $CSV after CLI fetch" >&2; exit 1; }
[[ -f "$HOT" ]] || { echo "missing $HOT after CLI fetch" >&2; exit 1; }

# 2. Audio zip via ModelScope's OSS tree API (signed URL).
echo "[2/4] Resolving signed URL for audio zip ..." >&2
ZIP_URL=$(curl -sfL --max-time 30 \
  "https://www.modelscope.cn/api/v1/datasets/$DATASET/oss/tree?Revision=master&Recursive=True" \
  | python3 -c 'import sys, json
data = json.load(sys.stdin).get("Data", [])
for entry in data:
    if entry.get("Path", "").endswith(".zip"):
        print(entry["Url"]); break')
[[ -n "$ZIP_URL" ]] || { echo "no .zip entry in OSS tree response" >&2; exit 1; }

ZIP="$STAGE/audio.zip"
echo "[3/4] Downloading audio zip (~32 MB) ..." >&2
curl -fL --max-time 600 -o "$ZIP" "$ZIP_URL"
unzip -qo "$ZIP" -d "$OUT"

# 3. Build wav.scp + text from the CSV.
echo "[4/4] Building wav.scp / text / hotwords.txt ..." >&2
python3 - "$CSV" "$OUT" <<'PY'
import csv, os, sys
csv_path, out_dir = sys.argv[1:3]
n = 0
with open(csv_path, newline="", encoding="utf-8") as f, \
     open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as scp, \
     open(os.path.join(out_dir, "text"),    "w", encoding="utf-8") as txt:
    reader = csv.reader(f)
    header = next(reader)
    if header[:2] != ["Audio:FILE", "Text:LABEL"]:
        raise SystemExit(f"unexpected CSV header: {header}")
    for row in reader:
        if not row or not row[0].strip():
            continue
        rel_wav, label = row[0].strip(), row[1].strip()
        wav = os.path.join(out_dir, rel_wav)
        if not os.path.isfile(wav):
            raise SystemExit(f"wav not found after unzip: {wav}")
        utt = os.path.splitext(os.path.basename(wav))[0]
        scp.write(f"{utt} {wav}\n")
        txt.write(f"{utt} {label.replace(' ', '')}\n")
        n += 1
print(f"wrote {n} utterances")
PY

cp -f "$HOT" "$OUT/hotwords.txt"
echo "wrote $(wc -l < "$OUT/hotwords.txt") hotwords to $OUT/hotwords.txt"

rm -rf "$STAGE"
