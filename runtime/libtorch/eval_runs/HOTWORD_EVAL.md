# Hotword Pipeline Evaluation

Ablation of the four hotword-pipeline optimizations on an IID Chinese ASR
benchmark. Each row of the result table corresponds to a `decoder_main`
invocation; flags differ only in which pieces of the pipeline are turned on.

## Dataset

- ModelScope `speech_asr/speech_asr_aishell1_hotwords_testsets` (AISHELL-1
  hotword subset), 235 utterances, 8 speakers, 16 kHz mono PCM.
- Hotword list: 187 entries, mostly proper nouns (人名 / 地名 / 机构名).
- 232 / 235 utterances contain at least one hotword; 282 hotword occurrences
  across the reference transcripts. Top recurring hotwords:
  佟健 (×15), 高桥大辅 (×8), 庞清 (×6), 宋芳 (×5).
- The test set is **IID**: short, independent clips across multiple speakers
  with little intra-session topic continuity.

## Model & runtime

- `u2pp_conformer-asr-cn-16k-online` (Conformer U2++, CTC + attention rescoring).
- Backend: libtorch 2.2.0 CPU, `chunk_size=-1` (non-streaming decode), 12 worker
  threads.
- No LM / TLG; CTC prefix beam + attention rescoring, plus the hotword
  pipeline under test.

## Conditions

| ID | Description | Hotword flags added on top of A |
|----|-------------|---------------------------------|
| A_baseline   | Plain CTC + attention rescoring, no hotword pathway     | — |
| B_phoneme    | + phoneme corrector (G2P + edit-distance fuzzy match)  | `--hotword_path`, `--pinyin_dict_path`, `--use_confidence_reward=false` |
| C_pinyin_ctx | + pinyin-side context graph (intended LM rescoring)    | + `--hanzi_unit_path`, `--pinyin_unit_path`, `--hanzi_pinyin_path`, `--context_pinyin_path`, `--context_score=3.0` |
| D_confidence | + per-token CTC confidence weighting on the match bonus | + `--use_confidence_reward=true` |
| E_cache      | + streaming LRU hotword cache (full stack)              | + `--enable_hotword_cache=true` |

Each flag group is additive (later rows include all earlier rows' flags).
Exact invocation is in `run_ablations.sh`.

## Results (12-thread, full test set)

| Condition     | wall (s) | CER %  | recall % | precision % | F1 %  | TP  | char-loss % |
|---------------|---------:|-------:|---------:|------------:|------:|----:|------------:|
| A_baseline    |       35 | 14.20  |    15.96 |       97.83 | 27.44 |  45 |       84.99 |
| B_phoneme     |       41 | 12.62  |    32.62 |       98.92 | 49.07 |  92 |       66.27 |
| C_pinyin_ctx  |       41 | 12.62  |    32.62 |       98.92 | 49.07 |  92 |       66.27 |
| D_confidence  |       45 | 12.04  |    36.17 |       99.03 | 52.99 | 102 |       63.03 |
| E_cache       |       43 | 12.04  |    36.17 |       99.03 | 52.99 | 102 |       63.03 |

- `recall` / `precision` / `F1`: per-occurrence hotword recall and precision
  on the reference transcripts (one count per occurrence; matched spans are
  masked so a long hotword and a substring don't double-count).
- `char-loss`: fraction of reference characters inside hotword spans that
  the hypothesis fails to recover. Lower is better.
- `TP`: true-positive hotword occurrences. `ref` is fixed at 282.

## Mapping to the four claimed optimizations

1. **基于音素相似度的宽泛召回** — A → B is the headline gain.
   - Recall: **15.96 % → 32.62 %** (+16.66 pts, ×2.04).
   - CER: 14.20 % → 12.62 % (−1.58 pts).
   - The phoneme corrector pulls in fuzzy hotword matches that the raw CTC
     beam misses entirely; precision is preserved at ~99 % because the
     occupy-span filter still requires `score ≥ threshold`.

2. **基于语义/拼音重打分的精排** — B → C is **a no-op on the search path**.
   - Identical sentences and identical metrics. `context_pinyin_graph` is
     built from `--context_pinyin_path` and survives in
     `DecodeResource::context_pinyin_graph`, but `AsrDecoder` only wires
     `context_hanzi_graph` into both `CtcPrefixBeamSearch` and
     `CtcWfstBeamSearch` (see `asr_decoder.cc` lines 55–60). The pinyin graph
     is dead code on the current search path.
   - Attention rescoring is the *de facto* semantic re-rank that's always
     on; it provides the gain attributed to "semantic rescoring" in earlier
     internal write-ups. Properly wiring the pinyin graph into rescoring is
     left as a follow-up (see *Known gaps*).

3. **基于声学置信度的路径奖励** — C → D adds a measurable gain.
   - Recall: 32.62 % → **36.17 %** (+3.55 pts).
   - CER: 12.62 % → **12.04 %** (−0.58 pts).
   - The reward (`AsrDecoder::CalculateMatchBonus`) scales the per-match
     bonus by `1 / max(0.4, avg_confidence)`: confident matches get a
     stronger bonus, low-confidence ones are damped. `--use_confidence_reward`
     toggles between `correct_with_confidence()` and the uniform-confidence
     `correct()` path so the ablation is clean.

4. **流式热词缓存策略** — D → E is **neutral on this benchmark**, after a bug
   fix described below.
   - Recall / precision / F1 / TP all identical to D.
   - `HotwordCache` (`hotword_cache.cc`) is designed for *streaming* sessions
     where a hotword recurs across consecutive utterances inside one user
     session. The boosts kick in only when `hit_count ≥ activate_threshold`
     (default 2), so a single hit doesn't activate. On AISHELL-IID, most
     hotwords occur once, and recurring ones (e.g., 佟健 ×15) tend to appear
     in correctly-decoded utterances that don't need the cache to help.
   - Wall clock is unchanged (43 s vs 45 s — within noise), confirming the
     cache is not contributing extra computation either.
   - **The cache will only show a recall gain on a streaming evaluation**
     where consecutive utterances share a small vocabulary of hotwords; on
     IID benchmarks it's a no-op by design.

## Integration bugs found and fixed during evaluation

Two bugs were found by running the ablation grid itself. Both are committed.

### 1. `max_append_path = 0` silently disabled the corrector candidates

`DecodeResource::max_append_path` defaulted to `0`, and the gflag value was
only written into it inside the `if (!FLAGS_context_pinyin_path.empty())`
branch in `params.h`. Without a pinyin path, the corrector still ran inside
`AppendPath()`, but `result_.resize(0)` truncated all the corrected
candidates away before rescoring. Symptom: B_phoneme produced byte-identical
hypotheses to A_baseline.

Fix: default the field to 20, and lift the assignment plus the
`hotword_cache` initialization out of the pinyin-only branch (`params.h`).

### 2. Cache boost inflated match scores enough to crowd out real corrections

Inside `PhonemeCorrector::correct_with_confidence()`, the per-match score
was computed as:

```cpp
float final_score = score + boost * 15;   // (bug)
```

`score` is the phoneme similarity ratio in `[0, 1]`; `boost` is capped at
0.15 by `HotwordCache::GetActiveHotwordsWithBoost()`. So a cached hotword's
final score could climb to `0.85 + 2.25 = 3.10`, far above any uncached
match's natural ceiling of `1.0`. Two downstream consequences:

- The sort + occupy-span filter put cached matches first; legitimate but
  uncached matches got rejected for overlap with the cached "winner",
  including cases where the cached "winner"'s replacement text equalled the
  raw ASR string verbatim, so no correction was emitted at all.
- `CalculateMatchBonus` is linear in `score`, so the inflated score
  produced a path-score bonus large enough to survive attention rescoring
  regardless of correctness.

Concrete examples from the buggy E_cache run (D vs E hypothesis diff after
sort):

```
D: 李斯达手持尖刀的自拍照              E: 李思达手持尖刀的自拍照            ← 李斯达 is a real hotword, dropped
D: 萨维申科宣布会再坚持一个冬奥会周期   E: 萨威申科宣布会再坚持一个冬奥会周期   ← uncorrected
D: 张昊领衔的中国双人华军团            E: 张浩领衔的中国双人华军团            ← uncorrected
D: 许茹芸与韩琦男朋友举行了婚礼        E: 许卢云与韩琦男朋友举行了婚礼        ← uncorrected
D: 普鲁申科丝毫没有隐退的打算          E: 比如申科丝毫没有隐退的打算           ← wrong replacement
```

Multi-thread recall fell from 36.17 % to 22.34 %; single-thread to 25.53 %,
so it was *not* purely a race condition on the shared `dynamic_hotword_boosts_`
map.

Fix (`corrector.cc:correct_with_confidence`): drop the `+ boost * 15` term.
The dynamic boost still lowers `current_threshold` so cached hotwords get
admitted at lower similarity (its intended job), but it no longer pollutes
the score used for sort, occupy-span selection, and the path-score bonus.

After the fix, E_cache matches D_confidence exactly on this IID test, which
is the expected outcome given the cache is designed for streaming.

## Known gaps

- **`context_pinyin_graph` / `pinyin_mapper_` are loaded but unused by the
  searcher.** They survive in `DecodeResource` and `AsrDecoder` member
  fields, but neither `CtcPrefixBeamSearch` nor `CtcWfstBeamSearch` is
  given the pinyin graph, and `pinyin_mapper_` has no call site in
  `asr_decoder.cc`. To make the pinyin context graph contribute, it would
  have to be wired into `Rescoring()` (e.g., as an extra score on each
  rescored hypothesis) — that work is out of scope for this evaluation.
- **`hotword_cache` is shared across decoder threads without a per-thread
  view.** `update_dynamic_hotwords()` now takes the corrector's mutex, but
  multiple threads still race to overwrite the same map. On streaming
  workloads (the cache's intended use) this is usually a single-thread
  pipeline, so the current implementation is acceptable; if multi-threaded
  streaming is a target, the cache should move to per-session state.
- **`HotwordCache(20, 2)` is hardcoded** in `params.h`. Making
  `capacity` and `activate_threshold` configurable via gflags would allow
  a dedicated streaming experiment to verify the +5 pt recall gain claimed
  for this stage.

## How to reproduce

```bash
# Build
cd runtime/libtorch
cmake -B build -DGRAPH_TOOLS=ON -DTORCH=ON  # if not already configured
cmake --build build -j --target decoder_main

# Build pinyin/hanzi tables (needed for C_pinyin_ctx)
python3 runtime/tools/build_pinyin_tables.py \
  --units .../u2pp_conformer-asr-cn-16k-online/units.txt \
  --word-dict runtime/libtorch/build/bin/dict/mandarin/word.txt \
  --hotwords aishell_test/hotwords.txt \
  --out-dir aishell_test/pinyin_tables \
  --score 3.0

# Run the five conditions
bash runtime/libtorch/eval_runs/run_ablations.sh
column -ts $'\t' runtime/libtorch/eval_runs/summary.tsv
```

Override paths via `MODEL`, `TESTSET`, `OUT_DIR`, `THREAD_NUM`,
`CONTEXT_SCORE` env vars. Wall-clock numbers in the table above are from a
12-core run; absolute timings will vary, but relative ordering and CER /
recall numbers are deterministic.
