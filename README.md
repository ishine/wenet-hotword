

# WeNet Hotword Pipeline

**Hotword-biased decoding for the [WeNet](https://github.com/wenet-e2e/wenet) C++ runtime.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![LibTorch 2.2.0](https://img.shields.io/badge/LibTorch-2.2.0-EE4C2C.svg)](https://pytorch.org/cppdocs/)

[**Eval Writeup**](runtime/libtorch/eval_runs/HOTWORD_EVAL.md)



**recall ↑ 4.6×** &nbsp;&nbsp; **CER ↓ 41%**

| | baseline | ours (autotuned) |
|--|:--:|:--:|
| hotword recall | 15.96% | **72.70%** |
| CER | 14.20% | **8.37%** |

<sub>Model: `wenet/u2pp_conformer-asr-cn-16k-online`</sub>
<br>
<sub>Dataset: AISHELL-1 hotword test</sub>


## 🌟 Features

- **Phoneme Corrector** — fuzzy hotword matching via G2P phoneme edit-distance on the n-best.
- **Confidence-Weighted Match Bonus** — per-hotword reward scaled by acoustic confidence.
- **LRU Hotword Cache** — recurring hotwords get a lowered fuzzy threshold in streaming.
- **Multi-Objective Autotuner** — Optuna TPE over decoder + hotword knobs, optimizing recall and CER jointly.

---

## 🚀 Quick Start

### 1. Install Python deps

```bash
cd /path/to/wenet-main
source .venv/bin/activate
pip install pyyaml dacite optuna
```

### 2. Download model + test set

```bash
modelscope download --model wenet/u2pp_conformer-asr-cn-16k-online \
  --local_dir ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
bash tools/prepare_aishell_hotwords.sh ~/userspace/wenet/aishell_test
```

### 3. Build decoder_main

```bash
cd runtime/libtorch
cmake -B build -DGRAPH_TOOLS=ON -DTORCH=ON
cmake --build build -j --target decoder_main
cd ../..
```

### 4. Smoke test

```bash
head -1 ~/userspace/wenet/aishell_test/wav.scp > /tmp/one.scp
runtime/libtorch/build/bin/decoder_main \
  --model_path ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online/final.zip \
  --unit_path  ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online/units.txt \
  --wav_scp    /tmp/one.scp \
  --hotword_path     ~/userspace/wenet/aishell_test/hotwords.txt \
  --pinyin_dict_path runtime/libtorch/build/bin/dict \
  --result     /dev/stdout
```

### 5. Run ablations

```bash
bash runtime/libtorch/eval_runs/run_ablations.sh
column -ts $'\t' runtime/libtorch/eval_runs/summary.tsv
```

### 6. Autotune

```bash
python3 tools/autotune.py \
  --config       runtime/libtorch/configs/default.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml
```
---

## ⚙️ Configuration

Edit `runtime/libtorch/configs/default.yaml` 

```yaml
paths:
  model_dir:         ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
  testset_dir:       ~/userspace/wenet/aishell_test
  eval_testset_dir:  ~/userspace/wenet/aishell1_indep_hotword
  pinyin_dict_dir:   runtime/libtorch/build/bin/dict

decode:
  chunk_size:       -1
  ctc_weight:       0.5
  rescoring_weight: 1.0
  reverse_weight:   0.0
  nbest:            10

hotword:
  hotword_path:          hotwords.txt
  fuzzy_threshold:       0.5
  max_append_path:       20
  use_confidence_reward: true
  enable_hotword_cache:  true
  confusion_matrix_path: runtime/libtorch/configs/confusion.wenetspeech.csv
  bonus_weight:          2.0
  confidence_floor:      0.4
  neighbor_threshold:    0.5

autotune:
  n_trials:  100
  sampler:   tpe
  cer_baseline: 14.20
```

Search space: `runtime/libtorch/configs/search_space.yaml`.

---

## 📊 Results

`u2pp_conformer-asr-cn-16k-online` on AISHELL hotword test (235 utts, 187 hotwords).

| Condition | What it is | CER% | recall% | precision% | F1% |
|-----------|-----------|------:|--------:|-----------:|----:|
| A_baseline | Plain CTC + attention rescoring, no hotword | 14.20 | 15.96 | 97.83 | 27.44 |
| B_phoneme | + phoneme corrector (G2P + fuzzy match) | 12.62 | 32.62 | 98.92 | 49.07 |
| D_confidence | + confidence-weighted match bonus | 12.04 | 36.17 | 99.03 | 52.99 |
| E_cache | + LRU hotword cache | 12.04 | 36.17 | 99.03 | 52.99 |
| F_autotune | E_cache + TPE-autotuned knobs (11 params) | 8.37 | 72.70 | 99.51 | 84.02 |
| G_wenet_native | Upstream WeNet character-FST biasing only | 10.97 | 46.45 | 99.24 | 63.29 |
| FG_stacked | F_autotune + G_wenet_native layered together | 8.48 | 73.76 | 97.20 | 83.87 |

**Held-out** (`aishell1_indep_hotword`, 115 utts — never seen during tuning):

| Condition | CER% | recall% | precision% | F1% |
|-----------|------:|--------:|-----------:|----:|
| D_confidence | 11.88 | 48.15 | 98.48 | 64.68 |
| F_autotune | 8.83 | 80.74 | 99.09 | 88.98 |
| G_wenet_native | 10.49 | 59.26 | 98.77 | 74.07 |
| FG_stacked | 8.89 | 80.74 | 97.32 | 88.26 |

Full write-up: [`HOTWORD_EVAL.md`](runtime/libtorch/eval_runs/HOTWORD_EVAL.md)

---

## 📂 Project Structure

```text
wenet-main/
├── runtime/core/decoder/
│   ├── corrector.{cc,h}        # PhonemeCorrector + fuzzy match + confusion matrix
│   ├── hotword_cache.{cc,h}    # LRU hotword cache
│   ├── asr_decoder.{cc,h}      # CalculateMatchBonus + n-best correction wiring
│   ├── params.h                # gflags (bonus_weight, confidence_floor, etc.)
│   └── context_graph.{cc,h}    # upstream WeNet character-FST context graph
├── runtime/core/bin/
│   └── decoder_main.cc         # decoder binary (+ daemon mode for autotune)
├── runtime/libtorch/configs/
│   ├── default.yaml            # base config
│   ├── default.yaml            # base config (includes 11-knob autotune)
│   └── search_space.yaml       # Optuna search space
├── runtime/libtorch/eval_runs/
│   ├── run_ablations.sh        # A→G ablation runner
│   └── HOTWORD_EVAL.md         # full evaluation report
└── tools/                      # autotune, metrics, data prep scripts
```

---

## 🙏 Acknowledgements

- **[WeNet](https://github.com/wenet-e2e/wenet)** — base ASR runtime.
- **[cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin)** — runtime G2P.

---

## 📜 License

Apache License 2.0, inherited from upstream WeNet.
