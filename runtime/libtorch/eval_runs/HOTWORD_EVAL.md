# Hotword Pipeline Evaluation

Ablation of the hotword-pipeline optimizations on an IID Chinese ASR
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
| D_confidence | + per-token CTC confidence weighting on the match bonus | + `--use_confidence_reward=true` |
| E_cache      | + streaming LRU hotword cache (full stack)              | + `--enable_hotword_cache=true` |
| F_autotune   | E_cache flag stack + Optuna NSGA-II knee config         | overrides `--rescoring_weight`, `--ctc_weight`, `--reverse_weight`, `--length_penalty`, `--nbest`, `--fuzzy_threshold`, `--max_append_path` (values from `configs/default.tuned.yaml`; see *Multi-objective autotune*) |
| G_wenet_native | **Off-ladder fair baseline.** Upstream WeNet character-level FST biasing alone — no corrector, no cache. | `--context_hanzi_path`, `--context_score=3.0` only (see *Head-to-head vs WeNet's native context-graph biasing*) |
| FG_stacked   | **Off-ladder orthogonality check.** F_autotune's rescoring-time stack layered on top of G's search-time FST bias — both layers active simultaneously. | F_autotune flags + `--context_hanzi_path`, `--context_score=3.0` |

A → F is additive (later rows include all earlier rows' flags).
G_wenet_native and FG_stacked are off-ladder. G_wenet_native is included so
the corrector stack is compared against upstream's hotword pathway rather
than against "no hotword pathway at all". FG_stacked is included to verify
the orthogonality argument (the corrector operates on the n-best at
rescoring time; the context FST biases the CTC prefix beam at search time;
stacking should be additive). Exact invocation for every row is in
`run_ablations.sh`.

## Results (12-thread, full test set)

| Condition     | wall (s) | CER %  | recall % | precision % | F1 %  | TP  | char-loss % |
|---------------|---------:|-------:|---------:|------------:|------:|----:|------------:|
| A_baseline    |       35 | 14.20  |    15.96 |       97.83 | 27.44 |  45 |       84.99 |
| B_phoneme     |       41 | 12.62  |    32.62 |       98.92 | 49.07 |  92 |       66.27 |
| D_confidence  |       45 | 12.04  |    36.17 |       99.03 | 52.99 | 102 |       63.03 |
| E_cache       |       43 | 12.04  |    36.17 |       99.03 | 52.99 | 102 |       63.03 |
| F_autotune    |       53 |  8.37  |    70.92 |       99.50 | 82.82 | 200 |       26.65 |
| G_wenet_native|       41 | 10.97  |    46.45 |       99.24 | 63.29 | 131 |       49.46 |
| FG_stacked    |       54 |  8.40  |    72.70 |       97.62 | 83.33 | 205 |       23.77 |

- `recall` / `precision` / `F1`: per-occurrence hotword recall and precision
  on the reference transcripts (one count per occurrence; matched spans are
  masked so a long hotword and a substring don't double-count).
- `char-loss`: fraction of reference characters inside hotword spans that
  the hypothesis fails to recover. Lower is better.
- `TP`: true-positive hotword occurrences. `ref` is fixed at 282.

## Mapping to the hotword-pipeline optimizations

1. **基于音素相似度的宽泛召回** — A → B is the headline gain.
   - Recall: **15.96 % → 32.62 %** (+16.66 pts, ×2.04).
   - CER: 14.20 % → 12.62 % (−1.58 pts).
   - The phoneme corrector pulls in fuzzy hotword matches that the raw CTC
     beam misses entirely; precision is preserved at ~99 % because the
     occupy-span filter still requires `score ≥ threshold`.

2. **基于声学置信度的路径奖励** — B → D adds a measurable gain.
   - Recall: 32.62 % → **36.17 %** (+3.55 pts).
   - CER: 12.62 % → **12.04 %** (−0.58 pts).
   - The reward (`AsrDecoder::CalculateMatchBonus`) scales the per-match
     bonus by `1 / max(0.4, avg_confidence)`: confident matches get a
     stronger bonus, low-confidence ones are damped. `--use_confidence_reward`
     toggles between `correct_with_confidence()` and the uniform-confidence
     `correct()` path so the ablation is clean.

3. **流式热词缓存策略** — D → E is **neutral on this benchmark**, after a bug
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

An earlier draft of this pipeline also included a pinyin-level context
graph (loaded from `--context_pinyin_path`) intended as a semantic /
pinyin rescoring step. That code path was never wired into the searcher
— the FST was built but neither `CtcPrefixBeamSearch` nor
`CtcWfstBeamSearch` ever received it — so its ablation row produced
identical metrics to B_phoneme. The dead code (and its `C_pinyin_ctx`
condition) has been removed; the comparable layer is upstream WeNet's
character-level FST biasing, which is now evaluated as a separate
off-ladder baseline (see *Head-to-head vs WeNet's native context-graph
biasing*).

## Correction-impact metric — separating "fix" from "harm"

`tools/compute-correction-impact.py` diffs the hyp produced *with* the
correction pathway against the no-correction baseline and classifies each
utterance by its edit-distance delta to the reference:

  - **unchanged** — correction did not fire
  - **fix** — `after` is closer to `ref` than `before` (genuine win)
  - **harm** — `after` is farther from `ref` (the "对的词被纠错" case)
  - **shuffle** — same distance, different text (lateral change)

With `--hotword-list`, harm utterances are further split into
`hotword_spurious` (a hotword was added in `after` that's not in `ref`),
`hotword_dropped` (a hotword present in `before` matching `ref` got
overwritten in `after`), and `other_regression` (non-hotword drift).

Across A → D on the AISHELL hotword test (compared against A_baseline):

| condition | unchanged | fix | harm | chars saved | chars damaged |
|-----------|----------:|----:|-----:|------------:|--------------:|
| B_phoneme   | 183 | 46 | 4 |  64 | 4 |
| D_confidence| 171 | 58 | 4 |  86 | 4 |

Harm sub-classification for D_confidence: `hotword_spurious=0`,
`hotword_dropped=1`, `other_regression=3`. Only **four utterances out of
235** are made worse by the correction pathway, and only one of those is a
hotword being clobbered. The corrector has a +0.91 fix/(fix+harm) char
ratio on this dataset — i.e., 91 % of its char-level impact is corrective.

## Rescoring (重打分) is the gate that keeps the corrector honest

The original walk-through left a question hanging: with attention rescoring
always on across A → E, the +0/−0 contribution of "semantic rescoring" is
invisible. To isolate it we ran two extra conditions varying just
`--rescoring_weight` (with the rest of the stack equal to D_confidence):

| condition           | CER %  | recall % | precision % | F1 %  | spur ins | harm utts |
|---------------------|-------:|---------:|------------:|------:|---------:|----------:|
| D_confidence (w=1.0)| 12.04 |  36.17  |       99.03 | 52.99 |        1 |   4 |
| D2_no_rescore (w=0) | 67.57 |  87.59  |       17.73 | 29.49 |     1146 | 231 |
| D3_high_rescore (w=5)| 13.70|  20.21  |       98.28 | 33.53 |        1 |  11 |

D2 turns rescoring off and the corrector floods the output with
phoneme-fuzzy hotword candidates that pass on raw CTC score alone:
**recall climbs to 87.59 %** because virtually every hotword that
acoustically resembles part of the utterance gets inserted, but
**precision collapses to 17.73 %** and 231 of 235 utterances are made
*worse* than the no-correction baseline. The corrector is a wide net; on
its own it is much too aggressive.

D3 pushes the rescoring weight to 5× to see whether more rescoring is
strictly better. It's not — over-weighted rescoring starts pruning real
hotword corrections too: recall falls to 20.21 % and harm utts climb to 11
(5 of which are `hotword_dropped` — corrections that were legitimate in D
but got rejected by an over-confident attention decoder at w=5).

So **rescoring is the gate that keeps the pipeline usable**, not a
post-hoc score smoother. The default `rescoring_weight = 1.0` /
`ctc_weight = 0.5` happens to be a *workable* operating point for
`u2pp_conformer-cn` on this test set: both 0.0 and 5.0 degrade quality, in
opposite ways. It is **not** the best point — the three-value sweep
(`{0, 1, 5}`) only proves that the optimum is interior, and a follow-up
NSGA-II search puts it at `rescoring_weight=0.378` (see
*Multi-objective autotune* below).

This also explains why the A → E ablation table appeared to show "no
contribution" from rescoring: rescoring is *implicit* in every row except
D2_no_rescore. The right way to demonstrate its effect is to ablate it
explicitly, as above.

## Multi-objective autotune — refuting the hand-picked sweet spot

The A → E table treats `rescoring_weight=1.0, ctc_weight=0.5,
fuzzy_threshold=0.5, max_append_path=20, reverse_weight=0.0` as fixed.
Those values came from the same single-axis ablation that produced D /
D2 / D3 above. An Optuna multi-objective sweep over an eight-knob box
refutes every one of them.

### Setup

- **Tuner**: Optuna 4.8 NSGA-II, 100 trials, objectives (F1↑, CER↓).
- **Search space** (`runtime/libtorch/configs/search_space.yaml`):
  8 knobs across decoder + hotword pathway.
- **Methodology split**: tune on `paths.testset_dir`
  (aishell_test, 235 utts) and report on the disjoint
  `paths.eval_testset_dir` (aishell1_indep_hotword, 115 utts). The
  held-out set is never read during search.
- **Knee pick**: highest-F1 Pareto trial whose CER stays under 14.20
  (the A_baseline CER on aishell_test). SQLite store at
  `configs/default.study.db` makes the study resumable.

### Result — single trial dominates the front

The Pareto front collapses to **one trial (#61)** that strictly
dominates every other completed trial on both axes simultaneously.
Knee config (`runtime/libtorch/configs/default.tuned.yaml`):

| Knob                           |  Default |       Knee | Search range  |
|--------------------------------|---------:|-----------:|---------------|
| `decode.rescoring_weight`      |     1.00 |  **0.378** | [0.3, 2.5]    |
| `decode.ctc_weight`            |     0.50 |  **0.729** | [0.2, 0.8]    |
| `decode.reverse_weight`        |     0.00 |  **0.346** | [0.0, 0.5]    |
| `decode.length_penalty`        |     0.00 |  **0.072** | [-1.0, 1.0]   |
| `decode.nbest`                 |       10 |         10 | {5, 10, 20}   |
| `hotword.fuzzy_threshold`      |     0.50 |  **0.457** | [0.3, 0.7]    |
| `hotword.max_append_path`      |       20 |     **10** | {10, 20, 40}  |
| `hotword.use_confidence_reward`|     true |       true | {true, false} |

| Set                       | Condition    |  CER % | recall % | precision % |  F1 %  | TP / ref |
|---------------------------|--------------|-------:|---------:|------------:|-------:|---------:|
| aishell_test (tune, 235)  | D_confidence |  12.04 |    36.17 |       99.03 |  52.99 |  102/282 |
| aishell_test (tune, 235)  | F_autotune   | **8.37** | **70.92** |   **99.50** | **82.82** | **200/282** |
| I_indep (held-out, 115)   | D_confidence |  11.88 |    48.15 |       98.48 |  64.68 |   65/135 |
| I_indep (held-out, 115)   | F_autotune   | **9.05** | **79.26** |   **99.07** | **88.07** | **107/135** |

Tune set: **+29.83 pp F1, −3.67 pp CER** over D_confidence. Held-out:
**+23.39 pp F1, −2.83 pp CER**. Both axes improve in the same direction on
the held-out set, with no precision regression beyond noise (99.03 → 99.07,
99.50 → 99.07), so the knee generalizes — it does not look like a
tune-set artifact.

### Why the hand-picked anchor missed the optimum

`rescoring_weight = 1.0` was the manually-picked anchor because the
three-value sweep `{0, 1, 5}` showed both extremes degrading quality.
NSGA-II's optimum is **0.378 — below every grid value the original
sweep examined**, including the 0.5–1.0 band that the manual study
*skipped over* on its way from 0 to 1 to 5. The gain lives in `[0.3,
0.5]` and the grid never had a chance to find it.

Three other settings shifted at the same time:

- `fuzzy_threshold` 0.50 → 0.457: the phoneme corrector becomes
  slightly more permissive on the recall side.
- `max_append_path` 20 → 10: the corrected-candidate set fed into
  rescoring shrinks by half.
- `reverse_weight` 0 → 0.346: the right-to-left attention decoder
  finally gets a meaningful weight in the rescore — it had been ignored
  by default.

Together: a wider phoneme net feeds a smaller, more carefully-ranked
candidate set into a bidirectional rescore that is no longer
over-weighted toward the forward pass. The signs are coherent. The
takeaway for future work is that the **default config under-uses
`reverse_weight` and over-uses `rescoring_weight`**, and that a
single-axis sweep can miss the joint optimum by a wide margin even when
each axis was probed in isolation.

### Reproducing

```bash
python3 tools/autotune.py \
  --config       runtime/libtorch/configs/default.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml
```

100 trials × ~50 s/trial ≈ 75 min on this set. Persistent SQLite store
at `autotune.study_db` resumes if the file exists; delete it to start
over. Outputs:

- `configs/default.tuned.yaml` — knee config (consumed by `run_ablations.sh`
  for the F_autotune row).
- `configs/default.pareto.jsonl` — full Pareto front for offline inspection.
- `configs/default.eval.txt` — knee config re-run on `paths.eval_testset_dir`.

## Head-to-head vs WeNet's native context-graph biasing

Upstream WeNet ships its own hotword-biasing mechanism: a character-level
weighted FST built from the hotword list (`context_score=3.0` by default)
that adds an arc-level bonus during CTC prefix-beam search. It is
character-level, not phoneme-level, and it operates at **search time**
inside the prefix beam — no separate corrector, no rescoring-time hotword
surface, no cache.

The corrector stack in this project operates at a different layer: it
runs at **rescoring time** on the n-best emitted by the prefix beam,
correcting phoneme spans through fuzzy match against the hotword pinyin
list and weighting the match bonus by acoustic confidence. These are
**orthogonal pipeline stages** — they touch different intermediate
objects (search lattice vs n-best list), so stacking them should be
additive rather than redundant. The FG_stacked row exists to verify
this empirically.

| Set                       | Condition       |  CER % | recall % | precision % |  F1 %  | TP / ref |
|---------------------------|-----------------|-------:|---------:|------------:|-------:|---------:|
| aishell_test (tune, 235)  | D_confidence    |  12.04 |    36.17 |       99.03 |  52.99 |  102/282 |
| aishell_test (tune, 235)  | G_wenet_native  |  10.97 |    46.45 |       99.24 |  63.29 |  131/282 |
| aishell_test (tune, 235)  | F_autotune      |   8.37 |    70.92 |   **99.50** |  82.82 |  200/282 |
| aishell_test (tune, 235)  | FG_stacked      | **8.40** | **72.70** |       97.62 | **83.33** | **205/282** |
| I_indep (held-out, 115)   | D_confidence    |  11.88 |    48.15 |       98.48 |  64.68 |   65/135 |
| I_indep (held-out, 115)   | G_wenet_native  |  10.49 |    59.26 |       98.77 |  74.07 |   80/135 |
| I_indep (held-out, 115)   | F_autotune      |   9.05 |    79.26 |   **99.07** |  88.07 |  107/135 |
| I_indep (held-out, 115)   | FG_stacked      | **8.89** | **80.74** |       98.20 | **88.62** | **109/135** |

Three observations:

1. **G_wenet_native beats D_confidence on every column, on both
   datasets.** At the hand-picked anchor (`rescoring_weight=1.0`,
   `fuzzy_threshold=0.5`, `max_append_path=20`, `reverse_weight=0`) the
   corrector stack is *worse* than upstream's single-knob character FST.
   This is not evidence that the corrector mechanism is unsound — it is
   evidence that the default operating point of an eight-knob stack is
   not where the optimum sits. The corrector pathway has more degrees
   of freedom than upstream (8 vs 1), so its default has to be replaced
   by a jointly-tuned operating point before the comparison is fair.

2. **F_autotune beats G_wenet_native on every column, on both datasets.**
   Once the eight knobs are jointly tuned by Optuna NSGA-II (see
   *Multi-objective autotune*) the corrector + confidence + cache stack
   pulls ahead of the FST baseline by +19.53 pp F1 on aishell_test and
   +14.00 pp F1 on I_indep, with CER lower by 2.60 and 1.44 pts as well.
   Autotune did not introduce a new mechanism — it located a better
   point in the same eight-knob box that D_confidence picked from. The
   win is "this surface, well-tuned" vs "that surface, untuned", which
   is the right shape of comparison once you accept that the two
   surfaces aren't competing for the same job.

3. **FG_stacked beats F_autotune on recall, F1, and (held-out) CER —
   confirming orthogonality.** Layering G's search-time character-FST
   bias on top of F's rescoring-time corrector stack gives net-positive
   F1 on both datasets (+0.51 pp on tune, +0.55 pp on held-out), with
   recall up +1.78 pp (tune) / +1.48 pp (held-out). CER moves +0.03 pp
   on the tune set (8.37 → 8.40, one character) and −0.16 pp on the
   held-out set (9.05 → 8.89). Precision dips ~1–2 pp because
   `context_score=3.0` is the upstream default — it was *not* part of
   the eight-knob NSGA-II search that produced F, so the second bias
   layer is operating at an un-tuned weight. The stacked
   precision/recall trade-off is what you would expect from adding a
   bias whose strength was not jointly tuned with the rest of the
   stack; the fix is to extend the search space to cover
   `context_score`. The **sign** of the orthogonality argument is
   confirmed on both datasets; the **magnitude** is bounded by joint
   tuning rather than by mechanism.

So the takeaway is structural rather than adversarial: the corrector
stack and the upstream context FST are **complementary layers**, not
competing implementations of the same idea. The corrector's eight-knob
surface is broader than upstream's one knob, so it can be tuned harder;
the character FST is sharper at the prefix-beam level, so it can catch
high-confidence hits the rescorer would otherwise leave on the table.
F_autotune is the right standalone story for the corrector stack;
FG_stacked is the right story for "what to deploy if you can afford
both pipeline stages".

Reproducer:

```bash
# G_wenet_native — WeNet-native upstream hanzi-FST biasing only
runtime/libtorch/build/bin/decoder_main \
  --chunk_size -1 --thread_num $(nproc) \
  --model_path $MODEL/final.zip --unit_path $MODEL/units.txt \
  --wav_scp $TESTSET/wav.scp \
  --context_hanzi_path $TESTSET/hotwords.txt --context_score 3.0 \
  --result runtime/libtorch/eval_runs/G_wenet_native.txt

# FG_stacked — F_autotune stack + upstream hanzi-FST bias (orthogonality)
runtime/libtorch/build/bin/decoder_main \
  --chunk_size -1 --thread_num $(nproc) \
  --model_path $MODEL/final.zip --unit_path $MODEL/units.txt \
  --wav_scp $TESTSET/wav.scp \
  --hotword_path $TESTSET/hotwords.txt \
  --pinyin_dict_path runtime/libtorch/build/bin/dict \
  --use_confidence_reward=true --enable_hotword_cache=true \
  --rescoring_weight=0.378 --ctc_weight=0.729 --reverse_weight=0.346 \
  --length_penalty=0.072 --nbest=10 \
  --fuzzy_threshold=0.457 --max_append_path=10 \
  --context_hanzi_path $TESTSET/hotwords.txt --context_score 3.0 \
  --result runtime/libtorch/eval_runs/FG_stacked.txt
```

## Streaming-order ablation — does the cache help under recurrence?

The IID table puts E_cache numerically equal to D_confidence, but the
cache is *designed for* a different distribution: consecutive utterances
within one session that share a small vocabulary of hotwords. To rule out
"cache might still help, we just didn't give it the right input" we
reshuffled the 235-utt test set into hotword-clustered runs (15 utterances
of 佟健 back-to-back, then 8 of 高桥大辅, …) using
`tools/build_streaming_scp.py`, and re-ran D / E with `thread_num=1` so the
cache is not fragmented across decoder threads.

| Condition              | order      | thread | CER %  | recall % | precision % | F1 %  | TP  |
|------------------------|------------|-------:|-------:|---------:|------------:|------:|----:|
| D_confidence (IID)     | iid        | 12     |  12.04 |    36.17 |       99.03 | 52.99 | 102 |
| E_cache (IID)          | iid        | 12     |  12.04 |    36.17 |       99.03 | 52.99 | 102 |
| D_confidence_stream    | clustered  |  1     |  12.04 |    36.17 |       99.03 | 52.99 | 102 |
| E_cache_stream         | clustered  |  1     |  12.07 |    36.17 |       99.03 | 52.99 | 102 |

Even given a maximally favourable order — every recurring hotword
(`佟健 ×15`, `高桥大辅 ×8`, `宋芳 ×5`, …) presented as a single run, with
the cache un-fragmented — D_confidence and E_cache produce **identical
hotword metrics**, and CER differs by 0.03 % (one character). This is the
strongest statement available on this dataset that the cache is not on the
critical path.

Why the cache is inert even when given recurrence: the cache lowers
`current_threshold` by at most 0.15 (from `base 0.03 + log(hit) × 0.015 +
recency × 0.06`, hard-capped). For the cache to convert a miss into a hit,
the missed hotword's phoneme-similarity score must lie in `[threshold −
0.15, threshold)`. On AISHELL-hotwords the 180 missed occurrences are
predominantly very acoustically distant from any hotword the corrector
considered — they score well below the lowered threshold. So lowering the
admission bar a hair doesn't rescue them. The cache would matter on a
distribution where misses cluster *just under* the regular threshold —
e.g. one user dictating a long meeting transcript where the same name is
recognized 60 % of the time and the other 40 % sit at score ≈ 0.4–0.5.

This is a property of the dataset and the acoustic model's hotword
similarity histogram, not a regression. It is also useful information for
deployment: on highly-IID workloads the cache can be left off without
loss; on streaming-dictation workloads it should be re-evaluated against
a dataset that actually contains borderline-score recurrences.

## Robustness — perturbed hotword inputs and out-of-distribution audio

The 235-utt / 187-hotword benchmark in the main table is a single
distribution. To check the optimization stack is not overfitted to it we
ran four additional configurations:

| ID         | What it stresses                              | Audio                                              | Hotword list                  |
|------------|-----------------------------------------------|----------------------------------------------------|-------------------------------|
| F_noisy    | precision when the user pastes junk hotwords  | original 235 utts                                  | 187 base + 50 synthetic decoys |
| G_partial  | recall when the user-supplied list is partial | original 235 utts                                  | top-30 by ref count           |
| H_oov      | spurious-insertion rate on unrelated audio    | 2000 AISHELL-1 test utts containing no hotword     | 187 base                      |
| I_indep    | recall on the same hotwords, different utts   | 115 AISHELL-1 test utts containing ≥1 hotword      | 187 base                      |

The aishell1 utts (H, I) are drawn from `AudioLLMs/aishell_1_zh_test`
parquet shard `test-00000-of-00003` (2307 rows of the AISHELL-1 test
split with audio embedded as `RIFF` bytes), partitioned by whether
`answer` contains any of the 187 hotwords as a hanzi substring: 2192
without, 115 with. The 2192 are subsampled to 2000 (seed 17). Generation
tooling: `tools/perturb_hotwords.py` (F, G), an inline parquet extractor
+ `tools/prepare_aishell1_subset.py` (H, I).

### F_noisy — 50 decoys appended

| condition     | CER %  | recall % | precision % | F1 %  | TP  | ref | hyp |
|---------------|-------:|---------:|------------:|------:|----:|----:|----:|
| A_baseline    |  14.20 |    15.96 |       97.83 | 27.44 |  45 | 282 |  46 |
| B_phoneme     |  12.44 |    33.33 |       98.95 | 49.87 |  94 | 282 |  95 |
| D_confidence  |  12.02 |    36.17 |       99.03 | 52.99 | 102 | 282 | 103 |
| E_cache       |  12.04 |    36.17 |       99.03 | 52.99 | 102 | 282 | 103 |

The decoys are 3-char Chinese strings drawn from surname / given-name /
place-name char pools and filtered to never appear as any 3-char window
in the reference (`occupied_trigrams()` in `tools/perturb_hotwords.py`).
D_confidence reproduces TP = 102, hyp = 103, precision = 99.03 % to the
third decimal place of the unperturbed run. The single spurious
insertion (`hyp − TP = 1`) is unchanged. The corrector rejects all 50
decoys via the phoneme / pinyin similarity filter — false-positive
count does not grow with list size, only with the size of the overlap
between hotword phonemes and the acoustics actually seen.

### G_partial — top-30 hotwords only

| condition     | CER %  | recall % | precision % | F1 %  | TP | ref | hyp |
|---------------|-------:|---------:|------------:|------:|---:|----:|----:|
| A_baseline    |  14.20 |    14.66 |      100.00 | 25.56 | 17 | 116 |  17 |
| B_phoneme     |  13.67 |    31.90 |      100.00 | 48.37 | 37 | 116 |  37 |
| D_confidence  |  13.41 |    34.48 |      100.00 | 51.28 | 40 | 116 |  40 |
| E_cache       |  13.43 |    34.48 |      100.00 | 51.28 | 40 | 116 |  40 |

Reference denominator drops from 282 (occurrences of all 187 hotwords)
to 116 (occurrences of the kept top-30). The A → D recall delta is
+19.82 pts vs +20.21 pts on the full list. The corrector's gain scales
with the size of the *relevant* hotword set, not with the size of the
curated 235-utt list. (Precision hits 100 % because the dropped
hotwords removed the single FP from the original D run.)

### H_oov — 2000 AISHELL-1 utts with no hotword in the reference

| condition     | wall (s) | CER % | recall % | precision % | F1 % | TP | ref | hyp |
|---------------|---------:|------:|---------:|------------:|-----:|---:|----:|----:|
| A_baseline    |      242 |  5.20 |     0.00 |        0.00 | 0.00 |  0 |   0 |   0 |
| B_phoneme     |      356 |  5.19 |     0.00 |        0.00 | 0.00 |  0 |   0 |   0 |
| D_confidence  |      339 |  5.19 |     0.00 |        0.00 | 0.00 |  0 |   0 |   0 |
| E_cache       |      351 |  5.19 |     0.00 |        0.00 | 0.00 |  0 |   0 |   0 |

This is the harshest test for false-positive rate: the user supplies all
187 hotwords, the audio contains zero of them. **The `hyp` column is 0
across every condition — the corrector inserts zero hotwords into 2000
out-of-distribution utterances.** CER drifts by 0.01 pt across A → E (one
character on 2000 utts); the pipeline is essentially neutral on audio
the hotwords were not built for.

These utts are easier than the 235-utt hotword benchmark (CER ≈ 5.2 %
vs ≈ 14.2 %): they are "ordinary" AISHELL-1 finance / news clips
without the proper-noun density that drove the 235-utt curation. Easy
acoustics is the *worst* case for spurious insertion (the acoustic
model is confident, the corrector has every opportunity to overrule),
so zero injections here is a strong statement about precision.

Wall time on 2000 utts: 242 s (A) → 339 s (D), so the full hotword
stack is +40 % runtime on a worst-case no-hit benchmark. On the original
235-utt set where most utterances do produce corrector candidates, the
relative overhead is closer to +30 % (35 s → 45 s).

### I_indep — 115 independent hotword-bearing utts

| condition     | CER %  | recall % | precision % | F1 %  | TP  | ref | hyp |
|---------------|-------:|---------:|------------:|------:|----:|----:|----:|
| A_baseline    |  13.76 |    25.93 |       97.22 | 40.94 |  35 | 135 |  36 |
| B_phoneme     |  12.31 |    44.44 |       98.36 | 61.22 |  60 | 135 |  61 |
| D_confidence  |  11.88 |    48.15 |       98.48 | 64.68 |  65 | 135 |  66 |
| E_cache       |  11.88 |    48.15 |       98.48 | 64.68 |  65 | 135 |  66 |
| F_autotune    |   9.05 |    79.26 |       99.07 | 88.07 | 107 | 135 | 108 |
| G_wenet_native|  10.49 |    59.26 |       98.77 | 74.07 |  80 | 135 |  81 |
| FG_stacked    |   8.89 |    80.74 |       98.20 | 88.62 | 109 | 135 | 111 |

Same 187 hotwords, disjoint utterances from the 235-utt benchmark (drawn
from a different parquet shard of the AISHELL-1 test split). 135
hotword occurrences across 115 utts, 64 distinct hotwords from the long
tail of the original list. Recall is *higher* than the original at every
stage — A → D delta of **+22.22 pts** vs +20.21 pts on the 235-utt set.
Precision is 98.48 % (vs 99.03 %); one extra FP across 115 utts.

The slightly-higher A → D gain on this set is mostly distributional: the
64 hotwords seen here are predominantly 3- or 4-hanzi names (long
phoneme spans where the corrector's fuzzy match has the most leverage),
while the original 235-utt set includes more 2-hanzi entries like
`东莞` and `平昌` whose short pinyin reads (`dong guan`, `ping chang`)
are too compact for edit-distance fuzzy matching to help much.

F_autotune is the knee config from `tools/autotune.py` re-run on
I_indep (which is `paths.eval_testset_dir` — never read during tuning).
Recall climbs by **+31.11 pts over D_confidence (48.15 → 79.26)** while
precision moves from 98.48 to 99.07. The held-out gap is smaller than on
the tune set (+23.39 pp F1 here vs +29.83 pp F1 on aishell_test) but the
direction is the same on both axes, so the knee is not overfitting to
aishell_test. See *Multi-objective autotune* for the full story.

### Roll-up

|                                          | recall % | precision % | spurious insertions |
|------------------------------------------|---------:|------------:|--------------------:|
| Original 235 / 187 hot, D                |    36.17 |       99.03 |                   1 |
| Original 235 / 187 hot, F_autotune       |    70.92 |       99.50 |                   1 |
| Original 235 / 187 hot, FG_stacked       |    72.70 |       97.62 |                   5 |
| F_noisy (+50 decoys), D                  |    36.17 |       99.03 |                   1 |
| G_partial (top-30), D                    |    34.48 |      100.00 |                   0 |
| H_oov (2000, no hotword), D              |        — |           — |                   0 |
| I_indep (115 hot, held-out), D           |    48.15 |       98.48 |                   1 |
| I_indep (115 hot, held-out), F_autotune  |    79.26 |       99.07 |                   1 |
| I_indep (115 hot, held-out), FG_stacked  |    80.74 |       98.20 |                   2 |

The pipeline is insensitive to junk hotwords (F_noisy matches the
original to 3 decimal places), scales with the size of the relevant
hotword subset (G_partial), does not inject hotwords into unrelated
audio (H_oov), and reproduces the A → D gain on a held-out hotword-bearing
split (I_indep). Stacking the autotune knee with upstream's character FST
(FG_stacked) gives the highest recall on both splits at the cost of a
1–2 pp precision dip, because `context_score=3.0` was not part of the
NSGA-II search that produced F (see *Head-to-head*). The 99 %-precision
/ 53 %-F1 headline from D in the main table is a property of the
hand-picked anchor; the 99.5 %-precision / 82.8 %-F1 headline from
F_autotune is what those same optimizations achieve once their
hyperparameters are jointly tuned; FG_stacked is what they achieve when
layered on top of upstream's orthogonal search-time bias.

## Extending to other datasets / models

The numbers above are specific to `u2pp_conformer-asr-cn-16k-online` ×
AISHELL-hotwords. Optimal thresholds and weights are
*distribution-dependent* — a model with sharper CTC posteriors prefers a
higher `fuzzy_threshold`; a domain with longer utterances prefers a
larger `rescoring_weight`; a dataset where hotwords genuinely recur
within sessions will reward `enable_hotword_cache=true`.

To retarget the pipeline without re-deriving these numbers by hand:

1. Copy `runtime/libtorch/configs/default.yaml` to a new file (e.g.
   `configs/aishell2.yaml`) and edit `paths.model_dir`, `paths.testset_dir`,
   `paths.eval_testset_dir`. The
   `hotword.hotword_path` is resolved relative to `testset_dir`, so dropping
   a `hotwords.txt` next to `text` and `wav.scp` is enough. Leave
   `eval_testset_dir` empty to skip the held-out pass.
2. (Optional) Edit `configs/search_space.yaml` to widen / narrow the
   ranges for the dimensions your dataset is sensitive to. The default
   eight-knob box covers what mattered on AISHELL; reach for a wider box
   only if a sweep flatlines.
3. Run the autotuner:

   ```bash
   python3 tools/autotune.py \
       --config configs/aishell2.yaml \
       --search-space configs/search_space.yaml
   ```

   100 NSGA-II trials by default (`autotune.n_trials`). The full Pareto
   front is persisted to `autotune.pareto_out` (JSONL); the knee config to
   `autotune.tuned_config_out`; the held-out eval to `autotune.eval_metrics_out`;
   the resumable Optuna study to `autotune.study_db`. Trial logs and
   per-trial hyp files live in `paths.out_dir/autotune/`.

Candidate harder datasets for a follow-up evaluation:

- **AISHELL-2 dev / iOS** (~2.5K utts) — bigger, different acoustic
  domain (mobile far-field). Would test whether the optimization
  parameters transfer across recording conditions, not just utterance
  sets.
- **MagicData (RAMC)** — streaming-style dialogue with recurring named
  entities; the right shape to actually exercise `HotwordCache` (which
  is inert on IID benchmarks by design — see *Streaming-order
  ablation*).
- **In-house meeting transcripts** — the dictation regime the cache was
  designed for. Real recurrence + borderline-confidence hotwords.

The AISHELL-1 hold-out runs (H_oov, I_indep) already cover "different
utterances, same acoustic model"; the items above add "different
acoustic distribution" and "real streaming recurrence".

## Known gaps

- **`hotword_cache` is shared across decoder threads without a per-thread
  view.** `update_dynamic_hotwords()` now takes the corrector's mutex, but
  multiple threads still race to overwrite the same map. On streaming
  workloads (the cache's intended use) this is usually a single-thread
  pipeline, so the current implementation is acceptable; if multi-threaded
  streaming is a target, the cache should move to per-session state.
- **`HotwordCache(20, 2)` is hardcoded** in `params.h`. Making
  `capacity` and `activate_threshold` configurable via gflags would allow
  a dedicated streaming experiment to verify the +5 pt recall gain claimed
  for this stage. The streaming-order ablation above shows the cache is
  inert on AISHELL-hotwords *even with* an idealized recurrence pattern,
  so the rationale for exposing those knobs is "let users tune on their
  own streaming corpus", not "fix a regression here".

## How to reproduce

```bash
# Build
cd runtime/libtorch
cmake -B build -DGRAPH_TOOLS=ON -DTORCH=ON  # if not already configured
cmake --build build -j --target decoder_main

# Run A_baseline → E_cache (additive ladder) plus the off-ladder
# G_wenet_native fair baseline. F_autotune and FG_stacked also run if
# configs/default.tuned.yaml exists. Seven rows total.
bash runtime/libtorch/eval_runs/run_ablations.sh
column -ts $'\t' runtime/libtorch/eval_runs/summary.tsv

# F_autotune / FG_stacked rows need a tuned config first — run the autotuner
# if absent. (The script skips both cleanly if configs/default.tuned.yaml is
# missing.)
python3 tools/autotune.py \
  --config       runtime/libtorch/configs/default.yaml \
  --search-space runtime/libtorch/configs/search_space.yaml

# Streaming-order ablation (D / E with hotword-clustered wav.scp, thread=1)
python3 tools/build_streaming_scp.py \
  --ref aishell_test/text \
  --scp aishell_test/wav.scp \
  --hotwords aishell_test/hotwords.txt \
  --out-scp aishell_test/wav.stream.scp \
  --out-text aishell_test/text.stream
# then re-run decoder_main with --wav_scp aishell_test/wav.stream.scp
# and --thread_num=1, toggling --enable_hotword_cache.

# F_noisy / G_partial: build perturbed hotword lists, point run_ablations.sh
# at the new TESTSET dirs
python3 tools/perturb_hotwords.py \
  --ref       aishell_test/text \
  --hotwords  aishell_test/hotwords.txt \
  --decoys    50 --top-k 30 \
  --out-dir   aishell_test/perturbed \
  --seed      17
for scenario in noisy partial; do
  ln -sf $PWD/aishell_test/wav.scp aishell_test/perturbed/$scenario/wav.scp
  ln -sf $PWD/aishell_test/text    aishell_test/perturbed/$scenario/text
  TESTSET=$PWD/aishell_test/perturbed/$scenario \
    OUT_DIR=runtime/libtorch/eval_runs/perturbed/$scenario \
    bash runtime/libtorch/eval_runs/run_ablations.sh
done

# H_oov / I_indep: extract AISHELL-1 hold-out from HF parquet, then run
hf download AudioLLMs/aishell_1_zh_test --repo-type dataset \
  --include 'data/test-00000-of-00003.parquet' --local-dir aishell1_hf_raw
#   (set HF_ENDPOINT=https://hf-mirror.com on mainland-China networks)
python3 tools/extract_aishell1_parquet.py \
  --parquet  aishell1_hf_raw/data/test-00000-of-00003.parquet \
  --hotwords aishell_test/hotwords.txt \
  --out-dir  . \
  --subsample 2000 --seed 17
for d in aishell1_oov_test aishell1_indep_hotword; do
  ln -sf $PWD/aishell_test/hotwords.txt   $d/hotwords.txt
  TESTSET=$PWD/$d \
    OUT_DIR=runtime/libtorch/eval_runs/${d#aishell1_} \
    bash runtime/libtorch/eval_runs/run_ablations.sh
done
```

Override paths via `MODEL`, `TESTSET`, `OUT_DIR`, `THREAD_NUM`,
`CONTEXT_SCORE`, `TUNED_YAML` env vars. Wall-clock numbers in the table
above are from a 12-core run; absolute timings will vary, but relative
ordering and CER / recall numbers are deterministic.
