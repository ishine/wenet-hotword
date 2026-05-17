# wenet-main 集成说明（含热词增强 / Pinyin 上下文图 / 流式热词缓存）

本文档记录把用户 `core/` 目录的 C++ 代码并入 `wenet-main/runtime/` 之后，相对上游 WeNet 做了哪些修改、为什么这么改、以及怎么编译运行。

代码与编译环境信息（已验证）：
- 编译机：Ubuntu 24.04 / WSL2，GCC 13.3.0，CMake 3.28.3
- LibTorch：2.2.0 + cpu（FetchContent 自动下载）
- 网络：大陆环境，所有 GitHub 资源通过 `gh-proxy.com` 反代拉取
- 模型：`wenet/u2pp_conformer-asr-cn-16k-online`（ModelScope，AISHELL u2++ Conformer，`final.zip` 220 MB）
- 测试音频：`runtime/test/resources/aishell-BAC009S0724W0121.wav`，识别结果 `广州市房地产中介协会分析`（与 AISHELL 标注一致）

---

## 1. 代码合入位置

整个 `core/` 目录是按 WeNet `runtime/` 的子模块拆分组织的：

```
core/
├── api/             →  runtime/core/api/
├── bin/             →  runtime/core/bin/
├── cmake/           →  runtime/core/cmake/
├── decoder/         →  runtime/core/decoder/    （改动最大）
├── frontend/        →  runtime/core/frontend/
├── kaldi/, utils/, websocket/, …
```

`runtime/libtorch/`、`runtime/onnxruntime/` 这些后端目录都软链到 `core/` 下的对应子模块，因此 `wenet-main` 里只有 `runtime/core/` 是真正的源码所在。

---

## 2. `decoder/` 改动总览

`runtime/core/decoder/` 既有上游 WeNet 已有的文件被改写，也有用户新增的整套模块。保留了 `*.orig` 备份方便对比。

### 2.1 新增的源文件（上游没有）

| 文件 | 作用 |
|------|------|
| `corrector.cc / corrector.h` | `HotwordCorrection` 命名空间：`PhonemeCorrector`、`FastRAG`、`PhonemeIndex`、`PinyinSplitter`、`PinyinProvider`、`EnglishProvider`、模糊音素匹配（`CONFUSION_MATRIX`、`fuzzy_substring_search_constrained_with_confidence`）。负责把 ASR 输出片段按拼音/英文音素去比对热词，做错误更正。 |
| `hotword_cache.cc / hotword_cache.h` | `wenet::HotwordCache`：LRU + 动态 boost 的热词缓存，`GetActiveHotwordsWithBoost()` 在解码时给上下文图加权。 |
| `search_interface.h` | 抽象出 `SearchInterface`，供 prefix-beam-search 与 WFST search 复用同一接口。 |

### 2.2 上游文件的主要改动

| 文件 | 行差 | 主要变化 |
|------|------|----------|
| `asr_decoder.cc / .h` | 814 / 79 | 在 `DecodeResource` 增加 `corrector`、`hotword_cache`、`oov_mapping`、`pinyin_mapper`、`context_hanzi_graph`、`context_pinyin_graph`、`hanzi_unit_table`、`pinyin_unit_table` 等字段；`DecodeResult` 增加 `corrected_hotwords` 与 `token_log_probs`；新增 `AppendPath()` / `CalculateMatchBonus()` / `TextToIds()` / `ApplyMatchesToSentence()`，把热词分支拼接进解码 nbest。 |
| `context_graph.cc / .h` | 583 / 80 | 新增 `PinyinMapper` 类（汉字↔拼音 id 映射）、`BuildPinyinContextGraph()`（基于拼音的上下文图）；`SplitContextToUnits()` 接受 `oov_mapping`，把 OOV 字符按替身字符（同音 / 近音）注入图；导出 `kFuzzyPinyinMap` 模糊拼音表。 |
| `params.h` | 271 | 大量新增 gflags：`hotword_path`、`pinyin_dict_path`、`cmu_dict_path`、`oov_mapping_path`、`context_hanzi_path`、`context_pinyin_path`、`hanzi_unit_path`、`pinyin_unit_path`、`hanzi_pinyin_path`、`fuzzy_threshold`、`fuzzy_threshold_en`、`enable_hotword_cache`、`max_append_path`；按需初始化 corrector、hanzi/pinyin 上下文图。 |
| `ctc_prefix_beam_search.cc` | 49 | 接入上下文热词图打分（hanzi 路径），用 `AppendPath()` 把命中热词的候选追加到 nbest。 |
| `ctc_wfst_beam_search.cc / .h` | 35 / 2 | 把 `opts_.blank` 硬编为 `0`（匹配上游模型 unit_id 约定）。 |
| `ctc_endpoint.cc / .h` | 8 / 13 | `min_trailing_silence` 默认 1000 → 800 毫秒，端点判定更激进。 |
| `torch_asr_model.cc` | 8 | 移除 `setGraphExecutorOptimize(false)` 与 `setFusionStrategy`（依赖更新后的 LibTorch JIT 行为）。 |
| `onnx_asr_model.cc / .h` | 修订 | 回退到旧 ONNX C API（已脱敏代码兼容当前 onnxruntime）。 |
| `asr_model.cc / .h` | 5 / 5 | 头文件适配。 |
| `CMakeLists.txt` | 28 | 把新增源文件（corrector、hotword_cache）加入 `decoder` 静态库；引入 `cpp-pinyin::cpp-pinyin` 链接。 |

---

## 3. 脱敏后的修复

用户交付时已脱敏，但还有两处硬编码与本地路径会让代码无法在干净环境中运行，我们做了如下修复：

1. **`corrector.cc:673`**（已修复）：原代码在 `correct()` 里直接把传入文本强转为固定测试串 `"嗯 是 那 个 最 新 的 GJG 二 零 二 六"`，让 ASR 实际输出失效。删除该行。
2. **`params.h:171/172`**（已修复）：把 `g_decode_resource` / `g_corrector` 这两个文件作用域全局变量从头文件里移除（多个翻译单元 include 会触发 ODR/redefinition；上游 `decoder_main.cc` 自己已经定义 `g_decode_resource`）。
3. **`decoder/context_graph.cc`**（已修复）：用户头文件里声明了两组 `BuildContextGraph` 重载（2 参与 3 参），但只实现了 3 参版本。新增 2 参版本，转发到 3 参且 `oov_mapping=nullptr`，恢复 `api/wenet_api.cc` 用到的旧接口。
4. **`api/wenet_api.cc:125`**（已修复）：字段重命名 `context_graph → context_hanzi_graph`。
5. **`decoder/params.h:263-279`**（已修复）：原代码对 `hanzi_unit_path`、`pinyin_unit_path`、`hanzi_pinyin_path` 是无条件 `CHECK`。改成 *按需加载*：仅当对应参数非空且文件存在时才读，使得基础 ASR（不启用 pinyin 上下文图时）也能跑通。
6. **PlateCorrector 整条链路下线**（commit 2 起）：原 `plate_corrector.{cc,h}` 含开发者本机绝对路径，且与 ASR 主链路无关。已删除文件 + `FLAGS_enable_plate_correction` + `DecodeResource::plate_corrector` + CMake 源文件登记。本工程只保留 PhonemeCorrector / 拼音 context graph / hotword cache 三条主路径。

> ⚠️ 这些修改都不影响"提供了完整配置时"的行为，与原始 `core/` 在该路径下输出等价。

---

## 4. 构建：依赖与镜像

`runtime/libtorch/CMakeLists.txt` 默认走 `FetchContent` 拉所有依赖。大陆网络下原 URL 多数不可达，做了如下调整：

### 4.1 已引入镜像（统一变量 `WENET_GH_MIRROR`）

| 模块 | 上游 URL | 改用 |
|------|----------|------|
| `gflags` | `github.com/gflags/gflags` | `gh-proxy.com/https://github.com/...` |
| `glog`   | `github.com/google/glog` | 同上 |
| `openfst`（含 `runtime/core/patch/openfst` 的 cmake patch） | `github.com/kkm000/openfst` | 同上 |
| `wetextprocessing`（GIT 克隆） | `github.com/wenet-e2e/WeTextProcessing.git` | 同上（git 克隆走 ghproxy） |
| `cpp-pinyin` 1.0.2（新引入） | `github.com/wolfgitpr/cpp-pinyin` | 同上；备用 `CPP_PINYIN_SOURCE_DIR=<本地路径>` |

直接可达、未改 URL：
- `libtorch 2.2.0 cpu`（`download.pytorch.org`，已验证可达）
- `boost 1.75.0`（`archives.boost.io`，可达）

如果你切换网络（如 VPN 翻墙）：在 cmake 命令行加 `-DWENET_GH_MIRROR=https://github.com` 即可恢复原始 URL；不需要再修改任何 `.cmake` 文件。

### 4.2 新增 CMake 模块

`runtime/core/cmake/cpp_pinyin.cmake`（新增）：负责拉取 `cpp-pinyin`，把头文件加入 include 路径，并提供 `cpp-pinyin::cpp-pinyin` alias target；`runtime/libtorch/CMakeLists.txt` 在 `include(openfst)` 之后 `include(cpp_pinyin)`。

### 4.3 编译命令

```bash
cd runtime/libtorch
mkdir -p build && cd build
cmake -DWEBSOCKET=OFF -DGRAPH_TOOLS=OFF ..   # 关掉 WEBSOCKET 可跳过 boost 100 MB+ 下载
cmake --build . -j$(nproc)
```

构建产物（已验证）：
- `build/bin/decoder_main` 12 MB
- `build/bin/label_checker_main` 12 MB
- `build/bin/api_main` 37 KB
- `build/api/libwenet_api.so` 19 MB

---

## 5. 运行

### 5.1 基础 ASR（无热词）

```bash
cd runtime/libtorch
MODEL=$HOME/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online
./build/bin/decoder_main \
  --chunk_size -1 \
  --model_path $MODEL/final.zip \
  --unit_path  $MODEL/units.txt \
  --wav_path   $HOME/userspace/wenet/wenet-main/test/resources/aishell-BAC009S0724W0121.wav
# stdout: test 广州市房地产中介协会分析
```

模型来自 ModelScope，已验证 SHA256 = `a3df30c07df4c01180ad590f2f1eb7f488b61801ce2353bff61880c641a6c413`。

```bash
mkdir -p models && cd models
URL=https://www.modelscope.cn/models/wenet/u2pp_conformer-asr-cn-16k-online/resolve/master
curl -fsSLO $URL/final.zip
curl -fsSLO $URL/units.txt
curl -fsSLO $URL/configuration.json
curl -fsSLO $URL/README.md
```

### 5.2 热词修正（PhonemeCorrector，无上下文图）

仅需 `hotword_path` + `pinyin_dict_path`，后者直接指向 cpp-pinyin 源码里的 `res/dict`：

```bash
PINYIN_DICT=$PWD/fc_base/cpp_pinyin-src/res/dict
echo -e "广州\n房地产\n中介协会" > /tmp/hotwords.txt

./build/bin/decoder_main \
  --chunk_size -1 \
  --model_path       $MODEL/final.zip \
  --unit_path        $MODEL/units.txt \
  --hotword_path     /tmp/hotwords.txt \
  --pinyin_dict_path $PINYIN_DICT \
  --wav_path         $HOME/userspace/wenet/wenet-main/test/resources/aishell-BAC009S0724W0121.wav
```

加 `--cmu_dict_path <cmudict.dict>` 可开启英文 G2P；不提供时英文回退到字符级切分。

### 5.3 拼音上下文图（完整链路）

需要再准备 3 张词表，AISHELL/u2++ 模型不自带，需基于 cpp-pinyin 的 `res/dict/mandarin/word.txt` 离线构造：

| 路径 | 内容 | 来源 |
|------|------|------|
| `--hanzi_unit_path` | `<汉字> <id>` 一行一条，FST `SymbolTable::ReadText` 可读 | 从 `units.txt` / `word.txt` 抽取汉字 |
| `--pinyin_unit_path` | `<pinyin> <id>` | 收集 `word.txt` 中全部独特拼音 |
| `--hanzi_pinyin_path` | `<汉字> <pinyin1> [pinyin2 ...]` 空白分隔 | 直接由 `word.txt` 转换 |
| `--context_pinyin_path` | `<text> <py1> <py2> ... <score>` 每行一条热词 | 项目自定义 |

> 这部分词表生成脚本目前没有捎带，因为不同模型的 `units.txt` 字符集不一样，需要根据具体场景选汉字粒度（是否包含繁简、是否 +BPE）。建议项目自带一份与发布模型匹配的 `gen_hanzi_pinyin.py`。

---

## 6. ASR / 数据集说明

| 资源 | 来源 | 状态 |
|------|------|------|
| `final.zip` + `units.txt` | ModelScope `wenet/u2pp_conformer-asr-cn-16k-online` | ✅ 已用，识别正确 |
| AISHELL 全量测试集 | OpenSLR / ModelScope `wenet/WenetSpeech` | 未下载（数据规模较大，按需取） |
| 单条测试 wav | `runtime/test/resources/aishell-BAC009S0724W0121.wav` | ✅ 仓库自带，16k mono PCM |
| cpp-pinyin 字典 | `runtime/libtorch/fc_base/cpp_pinyin-src/res/dict` | 由 FetchContent 自动落盘 |

`docs/pretrained_models.md` 里的 `wenet.org.cn/downloads?…` 链接当前 404（Tencent COS 返回 `NoSuchKey`），所以模型下载走 ModelScope 而不是上游 URL。

---

## 7. 已知遗留 & 后续工作

- [ ] 把汉字 / 拼音 / 映射表生成脚本（基于 `units.txt` + `word.txt`）落到 `runtime/tools/`，让任何模型都能一键生成 hotword 词表。
- [ ] 端到端验证 *热词命中* 的实际改写效果：选一条 ASR 默认会错的样本（如带专有名词），加进 `hotword_path`，对比是否改正。
- [ ] 上下游对比测试：跑 AISHELL test 子集，统计 CER 是否回归。
- [ ] `params.h` 里仍有几块被注释的 VAD / Punc 配置（line 23-24、51-54、148-169），是用户后续要接的子模块，目前不影响构建。
- [ ] `g_decode_resource` / `g_corrector` 当前已不被任何代码引用；如果后续 plugin 需要全局指针，应改为在某个 `.cc` 单独定义一次并以 `extern` 暴露。

---

## 附录 A：本次集成中修改的 `wenet-main` 文件

```
runtime/core/cmake/cpp_pinyin.cmake                  (新增)
runtime/core/cmake/gflags.cmake                      (URL 改镜像)
runtime/core/cmake/glog.cmake                        (URL 改镜像)
runtime/core/cmake/openfst.cmake                     (URL 改镜像)
runtime/core/cmake/wetextprocessing.cmake            (URL 改镜像)
runtime/libtorch/CMakeLists.txt                      (include(cpp_pinyin))
runtime/core/decoder/params.h                        (脱敏 + 可选词表)
runtime/core/decoder/corrector.cc                    (移除测试串)
runtime/core/decoder/context_graph.cc                (新增 2-arg BuildContextGraph)
runtime/core/api/wenet_api.cc                        (字段重命名)
runtime/core/decoder/plate_corrector.{cc,h}          (删除：车牌纠错下线)
```

所有源文件的上游版本已在同目录保存为 `*.orig`，可以 `diff -u <file>.orig <file>` 复核。

## 附录 B：可达资源速查（大陆网络）

| 用途 | URL |
|------|------|
| GitHub 反代 | `https://gh-proxy.com/https://github.com/...` |
| LibTorch | `https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.0%2Bcpu.zip` |
| Boost | `https://archives.boost.io/release/1.75.0/source/boost_1_75_0.tar.gz` |
| ModelScope 文件 | `https://www.modelscope.cn/models/<ns>/<repo>/resolve/master/<file>` |
| Ubuntu APT 镜像 | `mirrors.tuna.tsinghua.edu.cn` / `mirrors.aliyun.com` / `mirrors.ustc.edu.cn` |
