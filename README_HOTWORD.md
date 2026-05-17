<div align="center">

# ⚡ WeNet Hotword Pipeline

**Hotword-biased decoding for the [WeNet](https://github.com/wenet-e2e/wenet) C++ runtime.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![LibTorch 2.2.0](https://img.shields.io/badge/LibTorch-2.2.0-EE4C2C.svg)](https://pytorch.org/cppdocs/)

[**Eval Writeup**](runtime/libtorch/eval_runs/HOTWORD_EVAL.md) | [**Autotune**](tools/autotune.py) | [**Runtime**](runtime/libtorch/)

</div>

## 🌟 Features

- **Phoneme Corrector** — fuzzy-matches phoneme spans in the CTC nbest against the hotword list.
- **Confidence-Weighted Match Bonus** — per-hotword reward scales by 1 / acoustic-confidence.
- **LRU Hotword Cache** — recurring hotwords get a lowered fuzzy threshold in streaming.
- **Multi-Objective Autotuner** — Optuna NSGA-II over (F1, CER); held-out eval on a disjoint test set.

---

## 🚀 Quick Start


### 1. Create the project venv and install Python deps

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope "huggingface_hub[cli]" pyyaml dacite pyarrow optuna
```

### 2. Download model + hotword test set

```bash
modelscope download --model wenet/u2pp_conformer-asr-cn-16k-online \
  --local_dir ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
bash tools/prepare_aishell_hotwords.sh ~/userspace/wenet/aishell_test
```

### 3. Build the C++ runtime

```bash
cd runtime/libtorch
WENET_GH_MIRROR=https://gh-proxy.com/https://github.com \
  cmake -B build -DGRAPH_TOOLS=ON -DTORCH=ON
cmake --build build -j --target decoder_main
cd ../
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

Add confidence / cache / context-graph flags from `run_ablations.sh` to enable the rest.

### 5. Run the ablation

```bash
bash runtime/libtorch/eval_runs/run_ablations.sh
column -ts $'\t' runtime/libtorch/eval_runs/summary.tsv
```

### 6. Autotune

Tunes on `paths.testset_dir` then re-runs the knee config on `paths.eval_testset_dir` to report the held-out number. 100 NSGA-II trials, ~75 min on the AISHELL set; the SQLite study at `autotune.study_db` is resumable.

```bash
python3 tools/autotune.py \
  --config       runtime/libtorch/configs/default.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml
```

---

## ⚙️ Configuration

Knobs in `runtime/libtorch/configs/default.yaml`; autotuner search space in `runtime/libtorch/configs/search_space.yaml`.

```yaml
paths:
  decoder_bin:       runtime/libtorch/build/bin/decoder_main
  model_dir:         ~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
  testset_dir:       ~/userspace/wenet/aishell_test                # tune set
  eval_testset_dir:  ~/userspace/wenet/aishell1_indep_hotword      # held-out test
  pinyin_dict_dir:   runtime/libtorch/build/bin/dict
  out_dir:           runtime/libtorch/eval_runs
```

```yaml
decode:
  chunk_size:       -1               # [-1, 16, 32, 64]  -1 = non-streaming
  ctc_weight:       0.5              # CTC vs attention at rescoring
  rescoring_weight: 1.0              # [0.0..5.0]  0.0 collapses precision
  reverse_weight:   0.0
  nbest:            10
  beam:             16.0
  lattice_beam:     10.0
```

```yaml
hotword:
  hotword_path:          hotwords.txt   # relative to paths.testset_dir
  fuzzy_threshold:       0.5            # [0.3, 0.5, 0.7]  CN phoneme cutoff
  fuzzy_threshold_en:    0.5            # EN phoneme cutoff
  max_append_path:       20             # corrected candidates fed into rescoring
  use_confidence_reward: true           # CalculateMatchBonus scales by 1/avg_conf
  context_score:         3.0            # hanzi context graph weight
  enable_hotword_cache:  true           # LRU cache for streaming recurrence
```

```yaml
autotune:
  n_trials:         100                  # NSGA-II budget; resumes via study_db
  sampler:          nsga2                # [nsga2, tpe]
  cer_baseline:     14.20                # knee pick = max F1 with CER ≤ this
  study_db:         runtime/libtorch/configs/default.study.db
  tuned_config_out: runtime/libtorch/configs/default.tuned.yaml
  pareto_out:       runtime/libtorch/configs/default.pareto.jsonl
  eval_metrics_out: runtime/libtorch/configs/default.eval.txt
```

---

## 📊 Results

`u2pp_conformer-asr-cn-16k-online` on the AISHELL hotword test set (235 utts × 187 hotwords). The list is co-curated with the audio (232/235 utts contain ≥1 hotword by construction) — recall is an upper bound; see H_oov / I_indep below for the unaligned case.

| Condition       | CER %  | recall % | precision % | F1 %  |
|-----------------|-------:|---------:|------------:|------:|
| A_baseline      |  14.20 |    15.96 |       97.83 | 27.44 |
| B_phoneme       |  12.62 |    32.62 |       98.92 | 49.07 |
| D_confidence    |  12.04 |    36.17 |       99.03 | 52.99 |
| E_cache         |  12.04 |    36.17 |       99.03 | 52.99 |
| D2_no_rescore   |  67.57 |    87.59 |       17.73 | 29.49 |
| D3_high_rescore |  13.70 |    20.21 |       98.28 | 33.53 |
| *G_wenet_native* | *10.97* | *46.45* |    *99.24* | *63.29* |
| **F_autotune**  |  **8.37** |  **70.92** |   **99.50** | **82.82** |
| *FG_stacked*    | *8.40* | *72.70* |     *97.62* | *83.33* |

F_autotune = E_cache stack with the Optuna NSGA-II knee config from `tools/autotune.py` (tuned on aishell_test; held-out numbers below).

G_wenet_native = fair-baseline comparator using upstream WeNet's character-level FST biasing alone (`--context_hanzi_path` + `context_score=3.0`, no corrector / no cache / no pinyin tables). It beats the hand-picked D_confidence anchor on every column, and F_autotune beats it back by +19.5 pp F1.

FG_stacked = F_autotune's rescoring-time corrector stack layered on top of G_wenet_native's search-time FST bias. The two pathways are orthogonal (different pipeline stages), so stacking them is additive: +0.51 pp F1 on aishell_test, +0.55 pp F1 on I_indep. Precision dips ~1–2 pp because `context_score=3.0` is the upstream default and was not part of the NSGA-II search. See `HOTWORD_EVAL.md` *Head-to-head vs WeNet's native context-graph biasing*.

Robustness under perturbed hotword inputs and AISHELL-1 hold-out audio:

| Scenario                              | Condition    | recall % | precision % | spurious ins. |
|---------------------------------------|--------------|---------:|------------:|--------------:|
| Original 235 / 187 hot                | D_confidence |    36.17 |       99.03 |             1 |
| Original 235 / 187 hot                | F_autotune   |    70.92 |       99.50 |             1 |
| Original 235 / 187 hot                | FG_stacked   |    72.70 |       97.62 |             5 |
| F_noisy (187 + 50 decoys)             | D_confidence |    36.17 |       99.03 |             1 |
| G_partial (top-30)                    | D_confidence |    34.48 |      100.00 |             0 |
| H_oov (2000 AISHELL-1 utts, no hot)   | D_confidence |        — |           — |             0 |
| I_indep (115 AISHELL-1 utts, w/ hot)  | D_confidence |    48.15 |       98.48 |             1 |
| I_indep (115 AISHELL-1 utts, w/ hot)  | F_autotune   |    79.26 |       99.07 |             1 |
| I_indep (115 AISHELL-1 utts, w/ hot)  | FG_stacked   |    80.74 |       98.20 |             2 |

Full write-up: `runtime/libtorch/eval_runs/HOTWORD_EVAL.md`.

---

## 📂 Project Structure

```text
wenet-main/
├── runtime/
│   ├── core/decoder/
│   │   ├── corrector.{cc,h}             # PhonemeCorrector + fuzzy match + boost wiring
│   │   ├── hotword_cache.{cc,h}         # LRU hotword cache with dynamic boost
│   │   ├── context_graph.{cc,h}         # Aho-Corasick context-FST over unit table
│   │   ├── asr_decoder.{cc,h}           # AppendPath / CalculateMatchBonus / TextToIds
│   │   ├── search_interface.h           # shared by prefix-beam and WFST search
│   │   └── params.h                     # gflags for the hotword surface
│   └── libtorch/
│       ├── configs/
│       │   ├── default.yaml             # canonical decoder + hotword + autotune config
│       │   ├── search_space.yaml        # Optuna search space
│       │   ├── default.tuned.yaml       # knee-point config (written by autotune.py)
│       │   ├── default.pareto.jsonl     # full Pareto front (written by autotune.py)
│       │   ├── default.eval.txt         # held-out eval metrics (written by autotune.py)
│       │   └── default.study.db         # Optuna SQLite study (resumable)
│       └── eval_runs/
│           ├── run_ablations.sh         # A → G reproducer (A → F additive ladder + G fair baseline)
│           ├── HOTWORD_EVAL.md          # full evaluation write-up
│           ├── summary.tsv              # per-condition metrics table
│           └── {A..G}_*.{txt,log,metrics.txt}
├── tools/
│   ├── prepare_aishell_hotwords.sh      # ModelScope → wenet wav.scp + text + hotwords.txt
│   ├── extract_aishell1_parquet.py      # HF aishell_1_zh_test parquet → wenet test set (H_oov / I_indep)
│   ├── perturb_hotwords.py              # base hotwords → noisy (+decoys) / partial (top-k)
│   ├── compute-hotword-metrics.py       # per-occurrence recall / precision / F1
│   ├── compute-correction-impact.py     # fix / harm / shuffle classification
│   ├── build_streaming_scp.py           # cluster utts by hotword → streaming order
│   ├── autotune.py                      # Optuna multi-objective tuner (F1, CER) + held-out eval
│   └── decoder_config.py                # YAML-backed dataclass config
└── README_HOTWORD.md                    # this file
```

---

## 🙏 Acknowledgements

- **[WeNet](https://github.com/wenet-e2e/wenet)** — base ASR runtime; all non-hotword code paths are upstream.
- **[cpp-pinyin](https://github.com/wolfgitpr/cpp-pinyin)** — runtime G2P used by `PhonemeCorrector`.
- **[ModelScope `wenet/u2pp_conformer-asr-cn-16k-online`](https://www.modelscope.cn/models/wenet/u2pp_conformer-asr-cn-16k-online)** — AISHELL-trained U2++ Conformer.
- **[ModelScope `speech_asr/speech_asr_aishell1_hotwords_testsets`](https://www.modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets)** — 235-utt AISHELL hotword test set.

---

## 📜 License

Apache License 2.0, inherited from upstream WeNet.
