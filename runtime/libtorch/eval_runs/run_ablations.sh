#!/usr/bin/env bash
# Hotword ablation runner.
#
# Drives 5 decoder_main invocations on the AISHELL-1 hotword test set
# (235 utts, 187 hotwords) and emits per-condition CER + hotword
# recall/precision into eval_runs/summary.tsv.
#
# Conditions:
#   A_baseline                  no hotword pathway at all
#   B_phoneme                   + phoneme corrector (fuzzy recall, no confidence)
#   C_pinyin_ctx                + phoneme corrector + pinyin context graph
#   D_confidence                + acoustic-confidence weighted bonus
#   E_cache                     + LRU hotword cache (full stack)
#
# Override paths via env vars: MODEL, TESTSET, OUT_DIR, THREAD_NUM.

set -euo pipefail

# --- paths (override via env) ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WENET_DIR="$(cd "$RUNTIME_DIR/../.." && pwd)"
MODEL="${MODEL:-$HOME/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online}"
TESTSET="${TESTSET:-$HOME/userspace/wenet/aishell_test}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
THREAD_NUM="${THREAD_NUM:-$(nproc)}"

DECODER="$RUNTIME_DIR/build/bin/decoder_main"
PINYIN_DICT="$RUNTIME_DIR/build/bin/dict"   # cpp-pinyin runtime dict
TABLES="$TESTSET/pinyin_tables"

# --- sanity --------------------------------------------------------------------
for f in "$DECODER" "$MODEL/final.zip" "$MODEL/units.txt" \
         "$TESTSET/wav.scp" "$TESTSET/text" "$TESTSET/hotwords.txt" \
         "$TABLES/hanzi_unit.txt" "$TABLES/pinyin_unit.txt" \
         "$TABLES/hanzi_pinyin.txt" "$TABLES/context_pinyin.txt" \
         "$PINYIN_DICT/mandarin/word.txt"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "condition\twall_s\tCER%%\trecall%%\tprecision%%\tF1%%\tTP\tref\thyp\n" > "$SUMMARY"

# --- shared decoder flags ------------------------------------------------------
common_flags=(
  --chunk_size -1
  --thread_num "$THREAD_NUM"
  --model_path "$MODEL/final.zip"
  --unit_path  "$MODEL/units.txt"
  --wav_scp    "$TESTSET/wav.scp"
)

# Build per-condition extras as arrays so flag groups stack cleanly.
hotword_flags=(
  --hotword_path     "$TESTSET/hotwords.txt"
  --pinyin_dict_path "$PINYIN_DICT"
  --use_confidence_reward=false
)
pinyin_ctx_flags=(
  --hanzi_unit_path    "$TABLES/hanzi_unit.txt"
  --pinyin_unit_path   "$TABLES/pinyin_unit.txt"
  --hanzi_pinyin_path  "$TABLES/hanzi_pinyin.txt"
  --context_pinyin_path "$TABLES/context_pinyin.txt"
  --context_score      "${CONTEXT_SCORE:-3.0}"
)
cache_flag=( --enable_hotword_cache=true )
nocache_flag=( --enable_hotword_cache=false )
conf_on=( --use_confidence_reward=true )

run_condition () {
  local name="$1"; shift
  local hyp="$OUT_DIR/${name}.txt"
  local log="$OUT_DIR/${name}.log"
  echo "=== Condition $name ===" >&2
  local t0=$SECONDS
  "$DECODER" "${common_flags[@]}" "$@" --result "$hyp" >"$log" 2>&1 || {
      echo "decoder failed for $name; tail of log:" >&2
      tail -n 20 "$log" >&2
      exit 1
  }
  local wall=$((SECONDS - t0))

  # Char-level WER (which is CER for Chinese) using compute-cer.py
  local cer
  cer=$(python3 "$WENET_DIR/tools/compute-cer.py" --char=1 --v=0 \
        "$TESTSET/text" "$hyp" 2>/dev/null | grep -oE 'Overall -> [0-9.]+' \
        | awk '{print $3}')

  # Hotword metrics
  python3 "$WENET_DIR/tools/compute-hotword-metrics.py" \
      --hotword-list "$TESTSET/hotwords.txt" \
      --ref "$TESTSET/text" --hyp "$hyp" > "$OUT_DIR/${name}.metrics.txt"
  local recall precision f1 tp refn hypn
  recall=$(awk -F'=' '/^recall/{gsub(/ %/,"",$2);print $2}'    "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')
  precision=$(awk -F'=' '/^precision/{gsub(/ %/,"",$2);print $2}' "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')
  f1=$(awk -F'=' '/^F1/{gsub(/ %/,"",$2);print $2}'            "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')
  tp=$(awk -F':' '/true positives:/{print $2}'                "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')
  refn=$(awk -F':' '/hotword occurrences ref:/{print $2}'    "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')
  hypn=$(awk -F':' '/hotword occurrences hyp:/{print $2}'    "$OUT_DIR/${name}.metrics.txt" | tr -d ' ')

  printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$wall" "$cer" "$recall" "$precision" "$f1" "$tp" "$refn" "$hypn" \
    | tee -a "$SUMMARY"
}

# --- A: baseline -- no hotword pathway whatsoever ------------------------------
run_condition "A_baseline"

# --- B: + phoneme corrector (broad fuzzy recall, NO confidence reward) ---------
run_condition "B_phoneme" "${hotword_flags[@]}" "${nocache_flag[@]}"

# --- C: + pinyin context graph (semantic/acoustic rescoring) -------------------
run_condition "C_pinyin_ctx" \
  "${hotword_flags[@]}" "${pinyin_ctx_flags[@]}" "${nocache_flag[@]}"

# --- D: + acoustic-confidence weighted reward ----------------------------------
run_condition "D_confidence" \
  "${hotword_flags[@]}" "${pinyin_ctx_flags[@]}" "${nocache_flag[@]}" "${conf_on[@]}"

# --- E: + LRU hotword cache (full stack) ---------------------------------------
run_condition "E_cache" \
  "${hotword_flags[@]}" "${pinyin_ctx_flags[@]}" "${cache_flag[@]}" "${conf_on[@]}"

echo
echo "=== Summary ==="
column -ts $'\t' "$SUMMARY"
