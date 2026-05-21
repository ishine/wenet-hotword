#!/usr/bin/env bash
# One-shot preparation of the full benchmark:
#   - AISHELL-1 hotword test (tune set)
#   - AISHELL-2 iOS eval (test set + 1000-utt subset)
#   - 500-hotword list + 103-hard hotword list
#
# Usage:
#   bash tools/prepare_benchmark.sh [output_root]
#   default output_root = ~/userspace/wenet

set -euo pipefail

ROOT="${1:-$HOME/userspace/wenet}"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"

echo "=== Benchmark preparation ==="
echo "Output root: $ROOT"
echo ""

# ---------------------------------------------------------------------------
# 1. AISHELL-1 hotword test (tune set)
# ---------------------------------------------------------------------------
if [[ ! -f "$ROOT/aishell_test/wav.scp" ]]; then
    echo "[1/5] Downloading AISHELL-1 hotword test ..."
    bash "$(dirname "$0")/prepare_aishell_hotwords.sh" "$ROOT/aishell_test"
else
    echo "[1/5] AISHELL-1 hotword test already exists, skipping"
fi

# ---------------------------------------------------------------------------
# 2. AISHELL-2 iOS eval (test set)
# ---------------------------------------------------------------------------
AISHELL2_DIR="$ROOT/aishell2_eval"
if [[ ! -f "$AISHELL2_DIR/test5000/text" ]]; then
    echo "[2/5] Downloading AISHELL-2 iOS eval ..."
    mkdir -p "$AISHELL2_DIR"
    ZIP="$AISHELL2_DIR/TEST_DEV_DATA.zip"
    if [[ ! -f "$ZIP" ]]; then
        wget -q --show-progress -O "$ZIP" \
            "http://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST%26DEV%20DATA.zip"
    fi
    unzip -qo "$ZIP" -d "$AISHELL2_DIR"

    # AISHELL-2 ZIP has nested tar.gz — extract them
    for tgz in "$AISHELL2_DIR"/AISHELL-DEV-TEST-SET/*/test.tar.gz; do
        [ -f "$tgz" ] || continue
        dir=$(dirname "$tgz")
        tar -xzf "$tgz" -C "$dir"
    done

    # Build absolute paths
    TEST_ROOT="$AISHELL2_DIR/AISHELL-DEV-TEST-SET/iOS/test"
    mkdir -p "$AISHELL2_DIR/test5000"
    awk -v root="$TEST_ROOT" '{print $1 " " root "/" $2}' "$TEST_ROOT/wav.scp" \
        > "$AISHELL2_DIR/test5000/wav.scp"
    cp "$TEST_ROOT/trans.txt" "$AISHELL2_DIR/test5000/text"

    # Sample 1000-utt held-out subset (seed=42)
    mkdir -p "$AISHELL2_DIR/test1000"
    python3 -c "
import random
lines = open('$AISHELL2_DIR/test5000/wav.scp').readlines()
random.seed(42)
random.shuffle(lines)
with open('$AISHELL2_DIR/test1000/wav.scp', 'w') as f:
    f.writelines(lines[:1000])
"
    awk 'NR==FNR{a[$1];next} $1 in a' \
        "$AISHELL2_DIR/test1000/wav.scp" \
        "$AISHELL2_DIR/test5000/text" \
        > "$AISHELL2_DIR/test1000/text"
else
    echo "[2/5] AISHELL-2 already exists, skipping"
fi

# ---------------------------------------------------------------------------
# 3. 500-hotword list
# ---------------------------------------------------------------------------
if [[ ! -f "$AISHELL2_DIR/test1000/hotwords_500.txt" ]]; then
    echo "[3/5] Extracting 500-hotword list ..."
    python3 "$(dirname "$0")/prepare_hotwords.py" \
        "$AISHELL2_DIR/test5000/text" \
        -n 500 -c 2 \
        -o "$AISHELL2_DIR/test1000/hotwords_500.txt"
else
    echo "[3/5] 500-hotword list already exists, skipping"
fi

# ---------------------------------------------------------------------------
# 4. 103-hard hotword list (requires baseline)
# ---------------------------------------------------------------------------
if [[ ! -f "$AISHELL2_DIR/test1000/hotwords.txt" ]]; then
    echo "[4/5] Running baseline to filter hard-case hotwords ..."
    DECODER="$(dirname "$0")/../runtime/libtorch/build/bin/decoder_main"
    MODEL="$ROOT/models/u2pp_conformer-asr-cn-16k-online"
    if [[ ! -f "$DECODER" ]]; then
        echo "ERROR: decoder_main not found at $DECODER"
        echo "Please build first: cd runtime/libtorch && cmake -B build ..."
        exit 1
    fi
    if [[ ! -d "$MODEL" ]]; then
        echo "ERROR: model not found at $MODEL"
        echo "Please download: modelscope download --model wenet/u2pp_conformer-asr-cn-16k-online"
        exit 1
    fi
    "$DECODER" \
        --model_path "$MODEL/final.zip" \
        --unit_path "$MODEL/units.txt" \
        --wav_scp "$AISHELL2_DIR/test5000/wav.scp" \
        --result /tmp/test5000_baseline.txt \
        --thread_num 8

    python3 "$(dirname "$0")/prepare_hotwords.py" \
        "$AISHELL2_DIR/test5000/text" \
        --baseline-hyp /tmp/test5000_baseline.txt \
        --filter-hard --threshold 0.90 \
        -o "$AISHELL2_DIR/test1000/hotwords.txt"
else
    echo "[4/5] 103-hard hotword list already exists, skipping"
fi

# ---------------------------------------------------------------------------
# 5. Tune-set distractors (for Conservative / Ultra modes)
# ---------------------------------------------------------------------------
HOTWORDS_ORIG="$ROOT/aishell_test/hotwords_aggressive.txt"
HOTWORDS_FULL="$ROOT/aishell_test/hotwords_conservative.txt"
if [[ ! -f "$HOTWORDS_ORIG" ]] || [[ ! -f "$HOTWORDS_FULL" ]]; then
    echo "[5/5] Preparing tune-set hotword lists ..."
    # Extract original hotwords (first 187, before distractors)
    python3 "$(dirname "$0")/prepare_hotwords.py" \
        "$ROOT/aishell_test/text" \
        -n 200 -c 2 \
        -o "$HOTWORDS_ORIG"

    # Extract distractors and append
    cat "$ROOT/aishell_test/text" "$AISHELL2_DIR/test5000/text" > /tmp/merged_text.txt
    python3 "$(dirname "$0")/extract_distractors.py" \
        /tmp/merged_text.txt \
        --hotwords "$HOTWORDS_ORIG" \
        -n 180 -c 2 \
        -o /tmp/distractors.txt
    cp "$HOTWORDS_ORIG" "$HOTWORDS_FULL"
    cat /tmp/distractors.txt >> "$HOTWORDS_FULL"
else
    echo "[5/5] Tune-set hotword lists already exist, skipping"
fi

echo ""
echo "=== Done ==="
echo "Tune set:  $ROOT/aishell_test"
echo "Test set:  $ROOT/aishell2_eval/test1000"
echo "Hotwords:  $ROOT/aishell2_eval/test1000/hotwords_500.txt (500 words)"
echo "           $ROOT/aishell2_eval/test1000/hotwords.txt (hard-case)"
