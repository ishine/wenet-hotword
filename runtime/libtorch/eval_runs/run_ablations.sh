#!/usr/bin/env bash
# Hotword ablation runner.
#
# Drives up to 7 decoder_main invocations on the AISHELL-1 hotword test set
# (235 utts, 187 hotwords) and emits per-condition CER + hotword
# recall/precision into eval_runs/summary.tsv.
#
# Conditions:
#   A_baseline                  no hotword pathway at all
#   B_phoneme                   + phoneme corrector (fuzzy recall, no confidence)
#   D_confidence                + acoustic-confidence weighted bonus
#   E_cache                     + LRU hotword cache (full stack)
#   F_autotune                  E_cache + Optuna NSGA-II knee knobs from
#                               configs/default.tuned.yaml (skipped if absent)
#   G_wenet_native              Off-ladder fair baseline: upstream WeNet's
#                               character-FST biasing alone (no corrector,
#                               no cache). The A_baseline row turns hotword
#                               biasing fully off, so this one is the
#                               apples-to-apples comparator.
#   FG_stacked                  Off-ladder orthogonality check: F_autotune's
#                               rescoring-time stack layered on top of G's
#                               search-time FST bias. Skipped with F_autotune.
#
# Override paths via env vars: MODEL, TESTSET, OUT_DIR, THREAD_NUM, TUNED_YAML.

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

# --- sanity --------------------------------------------------------------------
for f in "$DECODER" "$MODEL/final.zip" "$MODEL/units.txt" \
         "$TESTSET/wav.scp" "$TESTSET/text" "$TESTSET/hotwords.txt" \
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

# --- D: + acoustic-confidence weighted reward ----------------------------------
run_condition "D_confidence" \
  "${hotword_flags[@]}" "${nocache_flag[@]}" "${conf_on[@]}"

# --- E: + LRU hotword cache (full stack) ---------------------------------------
run_condition "E_cache" \
  "${hotword_flags[@]}" "${cache_flag[@]}" "${conf_on[@]}"

# --- F: + Optuna NSGA-II knee config (full hotword stack + tuned knobs) --------
# F_autotune = E_cache flag stack overlaid with the knee config that
# tools/autotune.py writes to configs/default.tuned.yaml. Skipped cleanly if
# the tuned yaml is missing (e.g. fresh checkout, autotune not yet run).
TUNED_YAML="${TUNED_YAML:-$RUNTIME_DIR/configs/default.tuned.yaml}"
if [[ -f "$TUNED_YAML" ]]; then
  mapfile -t autotune_flags < <(WENET_DIR="$WENET_DIR" python3 - "$TUNED_YAML" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["WENET_DIR"], "tools"))
from decoder_config import DecoderConfig
cfg = DecoderConfig.from_yaml(sys.argv[1])
d, h = cfg.decode, cfg.hotword
print(f"--rescoring_weight={d.rescoring_weight}")
print(f"--ctc_weight={d.ctc_weight}")
print(f"--reverse_weight={d.reverse_weight}")
print(f"--length_penalty={d.length_penalty}")
print(f"--nbest={d.nbest}")
print(f"--fuzzy_threshold={h.fuzzy_threshold}")
print(f"--max_append_path={h.max_append_path}")
print(f"--use_confidence_reward={'true' if h.use_confidence_reward else 'false'}")
PY
)
  run_condition "F_autotune" \
    "${hotword_flags[@]}" "${cache_flag[@]}" \
    "${autotune_flags[@]}"
else
  echo "[skip] F_autotune: $TUNED_YAML not found; run tools/autotune.py first" >&2
fi

# --- G: WeNet upstream's native character-FST biasing (off-ladder baseline) ----
# Not part of the additive A → F ladder: this is upstream's hotword pathway
# via --context_hanzi_path (the unmodified upstream BuildContextGraph code
# path). No corrector, no cache; just the character-level FST built from
# hotwords.txt with context_score=3.0. Included so the corrector stack is
# benchmarked against upstream's hotword surface rather than against "no
# hotword pathway at all" (which is what A_baseline measures).
run_condition "G_wenet_native" \
  --context_hanzi_path "$TESTSET/hotwords.txt" \
  --context_score      "${CONTEXT_SCORE:-3.0}"

# --- FG_stacked: F_autotune + upstream context-graph (orthogonality check) ----
# The corrector / cache / confidence-reward path operates at rescoring time
# on the n-best, while --context_hanzi_path biases the CTC prefix beam at
# search time. They occupy different pipeline layers, so stacking should
# be additive. This row exists to verify that.
if [[ -f "$TUNED_YAML" ]]; then
  run_condition "FG_stacked" \
    "${hotword_flags[@]}" "${cache_flag[@]}" \
    "${autotune_flags[@]}" \
    --context_hanzi_path "$TESTSET/hotwords.txt" \
    --context_score      "${CONTEXT_SCORE:-3.0}"
fi

echo
echo "=== Summary ==="
column -ts $'\t' "$SUMMARY"
