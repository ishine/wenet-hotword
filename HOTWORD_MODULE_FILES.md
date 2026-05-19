# 热词增强模块文件清单

范围：C++ 运行时、Python 工具脚本、配置文件、评估脚本。

---

## 一、C++ 运行时（`runtime/core/`）

| 文件 | 作用 | 关键方法 / 数据结构 |
|------|------|---------------------|
| `runtime/core/decoder/corrector.cc` | 音素纠错器实现：拼音/英文→音素、FastRAG 模糊匹配、稠密混淆矩阵加载。 | `PhonemeCorrector`, `LoadConfusionMatrix`, `CalculateAvgConfidenceInRange` |
| `runtime/core/decoder/corrector.h` | 纠错器公共接口头文件。 | `PinyinProvider`, `EnglishProvider`, `MatchResult` |
| `runtime/core/decoder/hotword_cache.cc` | LRU 热词缓存实现：按命中次数动态降低 fuzzy threshold。 | `HotwordCache` |
| `runtime/core/decoder/hotword_cache.h` | 热词缓存类声明。 | `HotwordCache` |
| `runtime/core/decoder/asr_decoder.cc` | 解码器热词增强集成：n-best 热词校正、匹配奖励计算。 | `CalculateMatchBonus`, `ApplyHotwordCorrection` |
| `runtime/core/decoder/asr_decoder.h` | 解码器类及解码选项定义。 | `AsrDecoder`, `DecodeOptions` |
| `runtime/core/decoder/params.h` | gflags 定义与从 flag 初始化配置/资源的辅助函数。 | `InitDecodeOptionsFromFlags`, `InitDecodeResourceFromFlags` |
| `runtime/core/decoder/context_graph.cc/h` | 上游 WeNet 字符级 FST 上下文图（G_wenet_native 基线）。 | `ContextGraph` |
| `runtime/core/bin/decoder_main.cc` | 解码器主程序，支持 daemon 模式（模型常驻，JSON 协议调参）。 | `RunDaemon`, `BuildTrialResource` |

---

## 二、Python 工具脚本（`tools/`）

| 文件 | 作用 | 支持功能 |
|------|------|----------|
| `tools/autotune.py` | Optuna 多目标自动调参。 | TPE 搜索、daemon 模式、Pareto 输出、knee 选点、held-out 评估 |
| `tools/decoder_config.py` | YAML 配置 dataclass 定义。 | 配置解析与序列化 |
| `tools/compute-hotword-metrics.py` | 热词召回指标计算。 | recall / precision / F1 / char-loss |
| `tools/compute-cer.py` | 字符级错误率计算。 | CER，支持中文模式 |
| `tools/compute-correction-impact.py` | 纠错效果分类统计。 | fix / harm / shuffle / spurious / dropped |
| `tools/learn_confusion.py` | 从 CTC 后验学习音素混淆矩阵。 | 输出稠密 CSV 混淆表 |
| `tools/perturb_hotwords.py` | 热词列表扰动生成。 | decoys、partial、noisy 变体 |
| `tools/build_streaming_scp.py` | 按热词聚类重排 wav.scp。 | 流式顺序测试集构建 |
| `tools/extract_aishell1_parquet.py` | 从 HuggingFace parquet 提取子集。 | H_oov、I_indep 子集提取 |
| `tools/prepare_aishell_hotwords.sh` | 准备 AISHELL 热词测试数据。 | 数据下载与格式化 |

---

## 三、配置文件（`runtime/libtorch/configs/`）

| 文件 | 作用 |
|------|------|
| `default.yaml` | 基础配置：解码参数、热词参数、autotune 参数（含 11 个可调参数）。 |
| `search_space.yaml` | Optuna 搜索空间定义。 |

---

## 四、评估与运行脚本（`runtime/libtorch/eval_runs/`）

| 文件 | 作用 |
|------|------|
| `run_ablations.sh` | A→G 消融实验一键运行。 |
| `HOTWORD_EVAL.md` | 完整评估报告。 |

---

## 五、第三方依赖

| 文件/目录 | 作用 |
|-----------|------|
| `runtime/libtorch/fc_base/cpp_pinyin-src/` | cpp-pinyin G2P 库（汉字→拼音→音素）。 |
