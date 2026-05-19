# 热词增强模块评估报告

> 本文档旨在展示展示热词 pipeline 在特定模型/数据集的效果
---

## 数据集

- **测试集**：AISHELL-1 热词子集，235 句，8 说话人，16 kHz 单声道。
- **热词表**：187 条，以专有名词为主。
- **分布**：232/235 句包含至少一个热词；参考文本中共 282 次热词出现。

---

## 模型

`u2pp_conformer-asr-cn-16k-online`

---

## 实验条件

| 条件 | 说明 | 相较 A 新增的 flag |
|------|------|------------------|
| A_baseline | 纯 CTC + attention rescoring，无热词通路 | — |
| B_phoneme | + 音素纠错器（G2P + 编辑距离模糊匹配） | `--hotword_path`, `--pinyin_dict_path`, `--use_confidence_reward=false` |
| D_confidence | + 按 CTC 置信度加权的匹配奖励 | + `--use_confidence_reward=true` |
| E_cache | + LRU 热词缓存 | + `--enable_hotword_cache=true` |
| F_autotune | E_cache + Optuna TPE 调参后的 knee 配置 | 覆盖 11 个参数（见「多目标自动调参」） |
| G_wenet_native | 上游 WeNet 字符级 FST 偏置基线 | `--context_hanzi_path`, `--context_score=3.0` |
| FG_stacked | F_autotune + G_wenet_native 叠加 | F 的 flags + `--context_hanzi_path`, `--context_score=3.0` |

A → F 为累加关系；G 和 FG 为独立对照。

---

## 主结果

### 符号约定

| 参数 | 说明 | 计算方式 |
|------|------|----------|
| **CER** | 字符级错误率 | 编辑距离 / 参考总字符数 |
| **recall** | 热词出现级召回 | `TP / ref_occurrences`|
| **precision** | 热词出现级精确率 | `TP / hyp_occurrences` |
| **F1** | 热词出现级 F1 | `2 · recall · precision / (recall + precision)` |
| **TP** | 正确识别的热词出现次数 | - |
| **char-loss** | - |

- `ref_occurrences`：参考文本中该热词表所有热词的出现总次数。
- `hyp_occurrences`：hyp 中匹配到热词表的总次数。


### 结果

| 条件 | wall(s) | CER% | recall% | precision% | F1% | TP | char-loss% |
|------|--------:|-----:|--------:|-----------:|----:|---:|-----------:|
| A_baseline | 35 | 14.20 | 15.96 | 97.83 | 27.44 | 45 | 84.99 |
| B_phoneme | 41 | 12.62 | 32.62 | 98.92 | 49.07 | 92 | 66.27 |
| D_confidence | 45 | 12.04 | 36.17 | 99.03 | 52.99 | 102 | 63.03 |
| E_cache | 43 | 12.04 | 36.17 | 99.03 | 52.99 | 102 | 63.03 |
| F_autotune | 53 | 8.37 | 72.70 | 99.51 | 84.02 | 205 | 24.11 |
| G_wenet_native | 41 | 10.97 | 46.45 | 99.24 | 63.29 | 131 | 49.46 |
| FG_stacked | 54 | 8.48 | 73.76 | 97.20 | 83.87 | 208 | 22.81 |

---

## 各优化层贡献拆解

### A → B：音素模糊召回

**指标变化**：recall 15.96% → 32.62%，CER 14.20% → 12.62%。

音素纠错器拉回 CTC beam 完全漏掉的热词。

### B → D：声学置信度加权

**指标变化**：recall 32.62% → 36.17%，CER 12.62% → 12.04%。

高置信度匹配得更高奖励，低置信度被抑制。

### D → E：LRU 热词缓存

**指标变化**：recall / CER / precision / F1 / TP 均无变化。

cache 为流式会话设计，需同一会话内热词重复出现且命中次数 ≥ 2 才激活。AISHELL-IID 中多数热词仅出现一次，因此 cache 在本基准上为 no-op。

### E → F：多目标自动调参

**指标变化**：recall 36.17% → 72.70%，CER 12.04% → 8.37%。

Optuna TPE 联合调整 decode + hotword 共 11 个参数，将默认锚点移动到更优的联合工作点。详见「多目标自动调参」节。

---

## 纠错影响分析

`tools/compute-correction-impact.py` 按编辑距离变化分类：

- **fix**：纠错后更接近参考
- **harm**：纠错后更远离参考
- **shuffle**：距离不变，文本变化
- **unchanged**：纠错未触发

| 条件 | unchanged | fix | harm | shuffle | chars saved | chars damaged | fix/(fix+harm) |
|------|----------:|----:|-----:|--------:|------------:|--------------:|---------------:|
| B_phoneme | 183 | 46 | 4 | 2 | 64 | 4 | +0.882 |
| D_confidence | 171 | 58 | 4 | 2 | 86 | 4 | +0.910 |
| E_cache | 171 | 58 | 4 | 2 | 85 | 4 | +0.910 |
| F_autotune | 76 | 149 | 3 | 7 | 225 | 3 | +0.974 |
---

## Rescoring 的守门作用

固定 D_confidence 其余参数，仅调整 `--rescoring_weight`：

| 条件 | CER% | recall% | precision% | F1% | spur ins | harm utts |
|------|-----:|--------:|-----------:|----:|---------:|----------:|
| D_confidence (w=1.0) | 12.04 | 36.17 | 99.03 | 52.99 | 1 | 4 |
| D2_no_rescore (w=0) | 67.57 | 87.59 | 17.73 | 29.49 | 1146 | 231 |
| D3_high_rescore (w=5) | 13.70 | 20.21 | 98.28 | 33.53 | 1 | 11 |

- w=0：recall 暴涨至 87.59%，但 precision 崩溃至 17.73%，231/235 句变差。
- w=5：recall 跌至 20.21%，harm 上升至 11 句。

rescoring 是防止纠错过激的关键机制。

---

## 多目标自动调参

### 设置

- **调参器**：Optuna TPE multivariate，100 trials，目标 `(recall↑, CER↓)`。
- **搜索空间**：`runtime/libtorch/configs/search_space.yaml`。
- **划分**：在 `aishell_test`上搜索，在 `aishell1_indep_hotword`上验证。

### Knee 配置

| 参数 | 默认值 | Knee | 搜索范围 |
|------|-------:|-----:|----------|
| `decode.rescoring_weight` | 1.00 | 0.378 | [0.3, 2.5] |
| `decode.ctc_weight` | 0.50 | 0.729 | [0.2, 0.8] |
| `decode.reverse_weight` | 0.00 | 0.346 | [0.0, 0.5] |
| `decode.length_penalty` | 0.00 | 0.072 | [-1.0, 1.0] |
| `decode.nbest` | 10 | 10 | {5, 10, 20} |
| `hotword.fuzzy_threshold` | 0.50 | 0.457 | [0.2, 0.7] |
| `hotword.max_append_path` | 20 | 10 | {10, 20} |
| `hotword.use_confidence_reward` | true | true | {true, false} |
| `hotword.bonus_weight` | 2.0 | — | [0.5, 4.0] |
| `hotword.neighbor_threshold` | 0.5 | — | [0.3, 0.7] |
| `hotword.confidence_floor` | 0.4 | — | [0.2, 0.8] |

### 结果

| 数据集 | 条件 | CER% | recall% | precision% | F1% |
|--------|------|-----:|--------:|-----------:|----:|
| aishell_test (tune, 235) | D_confidence | 12.04 | 36.17 | 99.03 | 52.99 |
| aishell_test (tune, 235) | F_autotune | 8.37 | 72.70 | 99.51 | 84.02 |
| I_indep (held-out, 115) | D_confidence | 11.88 | 48.15 | 98.48 | 64.68 |
| I_indep (held-out, 115) | F_autotune | 8.83 | 80.74 | 99.09 | 88.98 |

Tune set：+31.03 pp F1，−3.67 pp CER。Held-out：+24.30 pp F1，−3.05 pp 。

---

## 鲁棒性测试

鲁棒性测试验证 pipeline 在非理想条件下的行为。四个测试分别覆盖：**垃圾热词耐受**、**热词表不完整**、**无关音频零插入**、**跨 utterance 泛化**。

| 条件 | 测试目的 | 音频 | 热词表 |
|------|----------|------|--------|
| F_noisy | 用户粘贴了含垃圾热词的大列表时，precision 是否保持稳定 | 原 235 句 | 187 原表 + 50 个合成 decoys |
| G_partial | 用户只提供了部分热词时，recall 是否按比例下降 | 原 235 句 | 按参考出现次数 top-30 |
| H_oov | 音频与热词完全无关时，是否零伪插入 | 2000 句无热词 AISHELL-1 | 187 原表 |
| I_indep | **独立测试集**：同一热词表、与 tune set 完全不重叠的 115 句音频，验证泛化能力 | 115 句含热词 AISHELL-1（与 235 句 tune set  disjoint） | 187 原表 |


### F_noisy（+50 decoys）

| 条件 | CER% | recall% | precision% | F1% | TP | ref | hyp |
|------|-----:|--------:|-----------:|----:|---:|----:|----:|
| A_baseline | 14.20 | 15.96 | 97.83 | 27.44 | 45 | 282 | 46 |
| B_phoneme | 12.44 | 33.33 | 98.95 | 49.87 | 94 | 282 | 95 |
| D_confidence | 12.02 | 36.17 | 99.03 | 52.99 | 102 | 282 | 103 |
| E_cache | 12.04 | 36.17 | 99.03 | 52.99 | 102 | 282 | 103 |

### G_partial（仅 top-30）

| 条件 | CER% | recall% | precision% | F1% | TP | ref | hyp |
|------|-----:|--------:|-----------:|----:|---:|----:|----:|
| A_baseline | 14.20 | 14.66 | 100.00 | 25.56 | 17 | 116 | 17 |
| B_phoneme | 13.67 | 31.90 | 100.00 | 48.37 | 37 | 116 | 37 |
| D_confidence | 13.41 | 34.48 | 100.00 | 51.28 | 40 | 116 | 40 |
| E_cache | 13.43 | 34.48 | 100.00 | 51.28 | 40 | 116 | 40 |

### H_oov（2000 句无热词）

| 条件 | 耗时(s) | CER% | recall% | precision% | F1% | TP | ref | hyp |
|------|--------:|-----:|--------:|-----------:|-----:|---:|----:|----:|
| A_baseline | 242 | 5.20 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 |
| B_phoneme | 356 | 5.19 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 |
| D_confidence | 339 | 5.19 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 |
| E_cache | 351 | 5.19 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 |


### I_indep（115 句独立热词 utterance）

| 条件 | CER% | recall% | precision% | F1% | TP | ref | hyp |
|------|-----:|--------:|-----------:|----:|---:|----:|----:|
| A_baseline | 13.76 | 25.93 | 97.22 | 40.94 | 35 | 135 | 36 |
| B_phoneme | 12.31 | 44.44 | 98.36 | 61.22 | 60 | 135 | 61 |
| D_confidence | 11.88 | 48.15 | 98.48 | 64.68 | 65 | 135 | 66 |
| E_cache | 11.88 | 48.15 | 98.48 | 64.68 | 65 | 135 | 66 |
| F_autotune | 8.83 | 80.74 | 99.09 | 88.98 | 109 | 135 | 108 |
| G_wenet_native | 10.49 | 59.26 | 98.77 | 74.07 | 80 | 135 | 81 |
| FG_stacked | 8.89 | 80.74 | 97.32 | 88.26 | 109 | 135 | 111 |

### 鲁棒性测试充分性评估

当前四项测试覆盖：

1. **垃圾热词耐受**（F_noisy）：precision 不受列表膨胀影响。
2. **热词表不完整**（G_partial）：recall 按有效子集比例下降。
3. **无关音频零插入**（H_oov）：零 false positive。
4. **跨 utterance 泛化**（I_indep）：同一热词表、不同音频上的增益可复制。

---

## 跨模型验证

同一热词 pipeline 在两个 WeNet 模型上的对比（AISHELL-1 hotword test, 235 utts）。

| 条件 | u2pp<br>CER% / recall% | multi_cn<br>CER% / recall% |
|------|------------------------|---------------------------|
| A_baseline | 14.20 / 15.96 | 4.28 / 74.11 |
| G_wenet_native | 10.97 / 46.45 | 2.34 / 91.49 |
| F_autotune | 8.37 / 72.70 | 2.02 / 95.74 |

multi_cn 为 unidirectional decoder（11008 units），autotune 时 `reverse_weight` 锁 0.0。

---

## 扩展到其他数据集/模型

上述数值特定于 `u2pp_conformer-asr-cn-16k-online` × AISHELL-hotwords。迁移步骤：

1. 复制 `default.yaml`，修改 `paths.model_dir`、`paths.testset_dir`、`paths.eval_testset_dir`。
2. （可选）调整 `search_space.yaml` 范围。
3. 运行 autotune：

```bash
python3 tools/autotune.py \
    --config configs/your_dataset.yaml \
    --search-space configs/search_space.yaml
```

候选数据集：

- **AISHELL-2 dev/iOS**（~2.5K 句）：不同录音条件，测试参数跨声学域迁移。
- **MagicData (RAMC)**：流式对话，含重复命名实体，可真正 exercise `HotwordCache`。

---