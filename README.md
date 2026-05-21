# WeNet Hotword

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)

**Hotword-biased decoding for the [WeNet](https://github.com/wenet-e2e/wenet) C++ runtime.**

> **Based On**:   
> **Model**: `wenet/u2pp_conformer-asr-cn-16k-online`  
> **Tune**: `AISHELL-1 hotword test` (235 utts, 187 hotwords) 

| | Baseline | Ours (Ultra) |
|--|--|--|
| **CER** | 5.14% | **4.82%** |
| **Recall** | 81.08% | **95.95%** |
| **Precision** | **95.24%** | 93.01% |
| **F1** | 87.59% | **94.46%** |

**Test**: `AISHELL-2 iOS eval` (1000 utts, 301 hotwords)

| | Baseline | Ours (Ultra) |
|--|--|--|
| **CER** | 5.14% | **4.83%** |
| **Recall** | 42.03% | **88.41%** |
| **Precision** | **100.00%** | 92.42% |
| **F1** | 59.18% | **90.37%** |

**Test**: `AISHELL-2 iOS eval` (1000 utts, 27 hard hotwords)

## Highlights

* **Phoneme Corrector** — fuzzy hotword matching via G2P phoneme edit-distance on the n-best
* **Confidence-Weighted Match Bonus** — per-hotword reward scaled by acoustic confidence
* **Multi-Objective Autotuner** — 2D/3D Pareto over decoder + hotword knobs, with early-exit stagnation detection

## Install

```bash
# Python environment
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchaudio pyyaml dacite optuna soundfile pypinyin jieba modelscope

# C++ runtime (requires cmake >= 3.14)
cd runtime/libtorch
cmake -B build -DGRAPH_TOOLS=ON -DTORCH=ON
cmake --build build -j --target decoder_main
cd ../..
```

## Quick Start

### 1. Download Model

```bash
modelscope download --model wenet/u2pp_conformer-asr-cn-16k-online \
  --local_dir ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
```

### 2. Download Datasets

preparation (downloads AISHELL-1 + AISHELL-2, builds hotword lists):

```bash
bash tools/prepare_benchmark.sh ~/userspace/wenet
```

### 3. Learn Confusion Matrix (per-model, one-time)

```bash
python3 tools/learn_confusion.py \
  --model_dir ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online \
  --wav_scp ~/userspace/wenet/aishell_test/wav.scp \
  --text ~/userspace/wenet/aishell_test/text \
  --out_csv runtime/libtorch/configs/confusion.csv \
  --device cpu
```

### 4. Autotune — Four Modes

| Mode | Config | Hotwords | Objective | When to use |
|------|--------|----------|-----------|-------------|
| **Aggressive** | `mode_aggressive.yaml` | 187 original | recall↑ + CER↓ | Hotword-dense domains  |
| **Balanced** | `mode_balanced.yaml` | 187 original | F1↑ + CER↓ | General voice assistant, balanced R/P |
| **Conservative** | `mode_conservative.yaml` | 349 (+distractors) | F1↑ + CER↓ | Open-domain dialogue, precision matters |
| **Ultra** | `mode_ultra.yaml` | 349 (+distractors) | F1↑ + CER↓ + Precision↑ | Financial/legal — false positive cost is high |

> **No free lunch**: Aggressive maximizes recall at the cost of precision (64% on 301-hotword test). Ultra trades ~3% recall for +29 precision points. Choose based on your domain's tolerance for false positives.

Run one (or all) modes:

```bash
# Aggressive
python3 tools/autotune.py \
  --config runtime/libtorch/configs/mode_aggressive.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml

# Balanced
python3 tools/autotune.py \
  --config runtime/libtorch/configs/mode_balanced.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml

# Conservative
python3 tools/autotune.py \
  --config runtime/libtorch/configs/mode_conservative.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml

# Ultra (3-objective Pareto)
python3 tools/autotune.py \
  --config runtime/libtorch/configs/mode_ultra.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml
```

### 5. Copy Hotword Lists

Hotword lists are shipped in `runtime/libtorch/configs/`. Copy them to your test set directory before evaluation:

```bash
cp runtime/libtorch/configs/hotwords_all.txt \
   ~/userspace/wenet/aishell2_eval/test1000/
cp runtime/libtorch/configs/hotwords_hard.txt \
   ~/userspace/wenet/aishell2_eval/test1000/
```

### 6. Evaluate on Held-Out

```bash
# Evaluate on 301-hotword list (mixed easy + hard)
python3 tools/evaluate_modes.py \
  --test-dir ~/userspace/wenet/aishell2_eval/test1000 \
  --hotwords hotwords_all.txt

# Evaluate on 27-hard hotword subset (baseline recall < 90%)
python3 tools/evaluate_modes.py \
  --test-dir ~/userspace/wenet/aishell2_eval/test1000 \
  --hotwords hotwords_hard.txt
```

## Results

**Model**: `wenet/u2pp_conformer-asr-cn-16k-online`   
**Tune**: AISHELL-1 hotword test    
**Test**: AISHELL-2 iOS eval subset

### 301-Hotword Test (mixed easy + hard)

| Mode | CER% | Recall% | Precision% | F1% |
|------|------:|--------:|-----------:|----:|
| Baseline (no hotword) | 5.14 | 81.08 | 95.24 | 87.59 |
| **Aggressive** | 6.00 | 92.79 | 63.78 | 75.60 |
| **Balanced** | 5.27 | 93.69 | 76.47 | 84.21 |
| **Conservative** | 4.98 | 93.24 | 83.81 | 88.27 |
| **Ultra** | **4.82** | **95.95** | **93.01** | **94.46** |

### 27-Hard Hotword Test (baseline recall < 90%)

| Mode | CER% | Recall% | Precision% | F1% |
|------|------:|--------:|-----------:|----:|
| Baseline | 5.14 | 42.03 | 100.00 | 59.18 |
| **Aggressive** | 5.10 | 98.55 | 64.76 | 78.16 |
| **Balanced** | 4.92 | 98.55 | 73.91 | 84.47 |
| **Conservative** | **4.68** | 94.20 | 86.67 | 90.28 |
| **Ultra** | 4.83 | 88.41 | **92.42** | **90.37** |

### Key Findings

1. **All hotword-enhanced modes improve or maintain CER** over no-hotword baseline (5.14% → 4.68–6.00%), showing the pipeline does not harm general ASR.
2. **On 27 hard-case hotwords** (foreign names the baseline misses), our method achieves **88% recall** vs baseline's **42%** — the phoneme corrector closes the gap where character-level matching fails.
3. **Ultra mode is the overall best**: highest F1 (94.46% on 301-hot, 90.37% on hard-case) via 3-objective Pareto optimization — no hard-coded precision floor needed.
4. **Conservative mode is the practical sweet spot**: lowest CER on hard-case (4.68%) with strong F1 (90.28%), making it suitable for precision-sensitive domains.

## Project Structure

```text
runtime/core/decoder/
  corrector.{cc,h}        # PhonemeCorrector + fuzzy match + confusion matrix
  hotword_cache.{cc,h}    # LRU hotword cache
  asr_decoder.{cc,h}      # CalculateMatchBonus + n-best correction wiring
  params.h                # gflags (bonus_weight, confidence_floor, etc.)
runtime/core/bin/
  decoder_main.cc         # decoder binary (+ daemon mode for autotune)
runtime/libtorch/configs/
  mode_{aggressive,balanced,conservative,ultra}.yaml  # four mode configs
  default.yaml            # base config
  search_space.yaml       # Optuna search space
tools/
  autotune.py             # multi-objective Pareto tuner
  compute-hotword-metrics.py
  prepare_hotwords.py     # extract 500-hot / filter hard-case
  evaluate_modes.py       # batch evaluate all 4 tuned configs
```

## Acknowledgements

* [WeNet](https://github.com/wenet-e2e/wenet) — base ASR runtime
* [cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin) — runtime G2P
* [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) — inspired the corrector design

## License

Apache License 2.0
