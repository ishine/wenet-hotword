#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-objective autotune for the hotword-pipeline decoder.

Reads a base config (`--config`) and a search space (`--search-space`),
runs `decoder_main` over `paths.testset_dir` for `n_trials` configurations
chosen by Optuna's NSGA-II sampler, and optimizes (F1↑, CER↓) jointly.

Outputs:
  * `autotune.tuned_config_out`   — knee point on the Pareto front
                                    (max F1 such that CER ≤ cer_baseline)
  * `autotune.pareto_out`         — full Pareto front (JSONL)
  * `autotune.eval_metrics_out`   — knee config re-run on `eval_testset_dir`
  * `autotune.study_db`           — Optuna SQLite store (resumable)

Re-running with the same `study_db` resumes the study; trials already in
the store are not re-executed. Delete the db to start over.

Usage:
    # Sanity check the search space, no trials:
    python3 tools/autotune.py --config runtime/libtorch/configs/default.yaml \
        --search-space runtime/libtorch/configs/search_space.yaml --dry-run

    # Real run (100 trials by default — see autotune.n_trials):
    python3 tools/autotune.py --config runtime/libtorch/configs/default.yaml \
        --search-space runtime/libtorch/configs/search_space.yaml

    # Resume / extend a prior study:
    python3 tools/autotune.py --config runtime/libtorch/configs/default.yaml \
        --search-space runtime/libtorch/configs/search_space.yaml \
        --n-trials 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import optuna
from optuna.samplers import NSGAIISampler, TPESampler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)

from decoder_config import (  # noqa: E402
    DecoderConfig, _load_yaml, merge_overrides, diff_config, _expand,
)


# --- search-space spec --------------------------------------------------------

@dataclasses.dataclass
class SearchSpace:
    """A flat `dotted.key -> spec` mapping; each spec is a dict whose `type`
    decides which `trial.suggest_*` call to dispatch.

    Supported `type` values:
        float       -> trial.suggest_float(key, low, high, step=?, log=?)
        int         -> trial.suggest_int(key, low, high, step=?)
        categorical -> trial.suggest_categorical(key, choices)
    """
    space: Dict[str, Dict[str, Any]]

    @classmethod
    def from_yaml(cls, path: str) -> "SearchSpace":
        raw = _load_yaml(path)
        space = raw.get("space", {}) or {}
        for k, spec in space.items():
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(f"search-space entry {k!r}: expected dict with 'type' key, got {spec!r}")
            if spec["type"] not in ("float", "int", "categorical"):
                raise ValueError(f"search-space entry {k!r}: unknown type {spec['type']!r}")
        return cls(space=space)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, spec in self.space.items():
            t = spec["type"]
            if t == "float":
                out[key] = trial.suggest_float(
                    key, float(spec["low"]), float(spec["high"]),
                    step=spec.get("step"), log=bool(spec.get("log", False)),
                )
            elif t == "int":
                out[key] = trial.suggest_int(
                    key, int(spec["low"]), int(spec["high"]),
                    step=int(spec.get("step", 1)),
                )
            elif t == "categorical":
                choices = spec.get("choices") or []
                out[key] = trial.suggest_categorical(key, choices)
        return out


# --- decoder invocation -------------------------------------------------------

@dataclasses.dataclass
class TrialResult:
    trial_id: int
    overrides: Dict[str, Any]
    cer: Optional[float]
    recall: Optional[float]
    precision: Optional[float]
    f1: Optional[float]
    tp: Optional[int]
    wall_s: float
    error: Optional[str] = None


def _run_trial(base: DecoderConfig, override: Dict[str, Any], trial_id: int,
               work_dir: str, log_decoder: bool,
               testset_override: Optional[str] = None) -> TrialResult:
    """Run decoder_main once with `override` applied; collect CER + hotword
    metrics. `testset_override` swaps `paths.testset_dir` for the held-out
    final-eval pass."""
    cfg = merge_overrides(base, override)
    if testset_override:
        cfg.paths.testset_dir = testset_override
    tag = f"trial_{trial_id:04d}"
    hyp_path = os.path.join(work_dir, f"{tag}.txt")
    log_path = os.path.join(work_dir, f"{tag}.log")

    decoder_bin = _expand(cfg.paths.decoder_bin, REPO_ROOT)
    if not os.path.exists(decoder_bin):
        return TrialResult(trial_id, override, None, None, None, None, None, 0.0,
                           error=f"decoder_main not found at {decoder_bin}")
    argv = [decoder_bin] + cfg.to_decoder_args(repo_root=REPO_ROOT,
                                               result_path=hyp_path)
    t0 = time.time()
    try:
        with open(log_path, "w") as logf:
            subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as exc:
        return TrialResult(trial_id, override, None, None, None, None, None,
                           time.time() - t0,
                           error=f"decoder_main failed (exit {exc.returncode}); see {log_path}")
    wall = time.time() - t0

    testset_dir = _expand(cfg.paths.testset_dir, REPO_ROOT)
    ref_path = os.path.join(testset_dir, "text")
    hotword_path = os.path.join(testset_dir, cfg.hotword.hotword_path) \
        if cfg.hotword.hotword_path and not os.path.isabs(cfg.hotword.hotword_path) \
        else cfg.hotword.hotword_path

    cer = _compute_cer(ref_path, hyp_path)
    metrics = _compute_hotword_metrics(ref_path, hyp_path, hotword_path,
                                       os.path.join(work_dir, f"{tag}.metrics.txt"))
    if not log_decoder:
        try:
            os.remove(log_path)
        except OSError:
            pass
    return TrialResult(
        trial_id=trial_id, overrides=override,
        cer=cer, recall=metrics.get("recall"), precision=metrics.get("precision"),
        f1=metrics.get("f1"), tp=metrics.get("tp"), wall_s=wall,
    )


_CER_RE = re.compile(r"Overall\s*->\s*([0-9.]+)")


def _compute_cer(ref: str, hyp: str) -> Optional[float]:
    tool = os.path.join(REPO_ROOT, "tools", "compute-cer.py")
    try:
        out = subprocess.check_output(["python3", tool, "--char=1", "--v=0",
                                       ref, hyp], stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return None
    m = _CER_RE.search(out)
    return float(m.group(1)) if m else None


def _compute_hotword_metrics(ref: str, hyp: str, hotwords: str,
                             out_path: str) -> Dict[str, float]:
    tool = os.path.join(REPO_ROOT, "tools", "compute-hotword-metrics.py")
    if not (hotwords and os.path.exists(hotwords)):
        return {}
    try:
        out = subprocess.check_output(["python3", tool, "--hotword-list", hotwords,
                                       "--ref", ref, "--hyp", hyp],
                                      stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError:
        return {}
    with open(out_path, "w") as f:
        f.write(out)
    result: Dict[str, float] = {}
    for line in out.splitlines():
        if "=" in line and "%" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().rstrip("%").strip()
            try:
                result[k] = float(v)
            except ValueError:
                pass
        if "true positives:" in line:
            try:
                result["tp"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return {
        "recall": result.get("recall"),
        "precision": result.get("precision"),
        "f1": result.get("F1"),
        "tp": result.get("tp"),
    }


# --- Optuna driver -----------------------------------------------------------

# Sentinel values for failed trials. Optuna requires numeric returns from a
# multi-objective objective; we report worst-possible values so the trial is
# dominated on the Pareto front but the study keeps moving.
_FAIL_F1, _FAIL_CER = -1.0, 1e6


def _build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name in ("nsga2", "nsgaii", "nsga-ii"):
        return NSGAIISampler(seed=seed)
    if name in ("tpe", "motpe"):
        # MOTPESampler was deprecated; modern TPESampler handles multi-objective
        # natively. Keep "motpe" as an alias for backwards compat.
        return TPESampler(seed=seed, multivariate=True, group=True)
    raise ValueError(f"unknown sampler {name!r} (expected nsga2 | tpe)")


def _knee_pick(trials: List[optuna.trial.FrozenTrial], cer_baseline: float
               ) -> Optional[optuna.trial.FrozenTrial]:
    """Pick the Pareto trial with the highest F1 whose CER stays under
    `cer_baseline`. Falls back to the trial closest to the baseline (lowest
    CER) if every Pareto trial violates the cap."""
    if not trials:
        return None
    valid = [t for t in trials if t.values is not None and t.values[1] <= cer_baseline]
    if valid:
        return max(valid, key=lambda t: t.values[0])
    # everything blew through the CER cap; pick the lowest-CER point as a tie-break
    return min(trials, key=lambda t: t.values[1])


def _serialize_trial(t: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    f1, cer = (t.values or [None, None])[:2]
    rec = {
        "trial_id": t.number,
        "f1": f1,
        "cer": cer,
        "state": t.state.name,
        "params": dict(t.params),
        "user_attrs": dict(t.user_attrs),
    }
    return rec


# --- main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="base config YAML (e.g. configs/default.yaml)")
    p.add_argument("--search-space", required=True,
                   help="search space YAML (e.g. configs/search_space.yaml)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="override autotune.n_trials")
    p.add_argument("--sampler", default=None,
                   help="override autotune.sampler (nsga2 | tpe)")
    p.add_argument("--seed", type=int, default=None, help="override autotune.random_seed")
    p.add_argument("--work-dir", default=None,
                   help="per-trial hyp/metrics dir (default: paths.out_dir/autotune)")
    p.add_argument("--keep-decoder-logs", action="store_true",
                   help="keep decoder_main stdout per trial (default: delete)")
    p.add_argument("--skip-eval", action="store_true",
                   help="don't run the held-out eval pass after tuning")
    p.add_argument("--dry-run", action="store_true",
                   help="print search space, sampler, n_trials and exit")
    args = p.parse_args()

    base = DecoderConfig.from_yaml(args.config)
    space = SearchSpace.from_yaml(args.search_space)

    n_trials = args.n_trials if args.n_trials is not None else base.autotune.n_trials
    sampler_name = args.sampler or base.autotune.sampler
    seed = args.seed if args.seed is not None else base.autotune.random_seed

    work_dir = args.work_dir or os.path.join(_expand(base.paths.out_dir, REPO_ROOT), "autotune")
    os.makedirs(work_dir, exist_ok=True)

    study_db = _expand(base.autotune.study_db, REPO_ROOT)
    os.makedirs(os.path.dirname(study_db) or ".", exist_ok=True)
    pareto_out = _expand(base.autotune.pareto_out, REPO_ROOT)
    tuned_out = _expand(base.autotune.tuned_config_out, REPO_ROOT)
    eval_out = _expand(base.autotune.eval_metrics_out, REPO_ROOT)

    print(f"[info] sampler   = {sampler_name}")
    print(f"[info] n_trials  = {n_trials}")
    print(f"[info] search    = {len(space.space)} knobs ({', '.join(space.space)})")
    print(f"[info] study_db  = {study_db}")
    print(f"[info] work_dir  = {work_dir}")

    if args.dry_run:
        for k, spec in space.space.items():
            print(f"  {k}: {spec}")
        return 0

    sampler = _build_sampler(sampler_name, seed)
    study = optuna.create_study(
        study_name=base.autotune.study_name,
        storage=f"sqlite:///{study_db}",
        directions=["maximize", "minimize"],  # F1, CER
        sampler=sampler,
        load_if_exists=True,
    )
    done_before = len(study.trials)
    if done_before:
        print(f"[info] resuming study: {done_before} prior trials in {study_db}")

    def objective(trial: optuna.Trial):
        override = space.suggest(trial)
        t = _run_trial(base, override, trial.number, work_dir, args.keep_decoder_logs)
        # stash everything on the trial so post-hoc analysis works without the JSONL
        for k, v in {"recall": t.recall, "precision": t.precision, "tp": t.tp,
                     "wall_s": t.wall_s, "error": t.error}.items():
            if v is not None:
                trial.set_user_attr(k, v)
        if t.error or t.cer is None or t.f1 is None:
            print(f"[trial {trial.number}] error: {t.error or 'no metrics'}")
            return _FAIL_F1, _FAIL_CER
        print(f"[trial {trial.number}] F1={t.f1:.2f}% CER={t.cer:.2f}% "
              f"recall={t.recall:.2f}% wall={t.wall_s:.1f}s  {override}")
        return t.f1, t.cer

    remaining = max(0, n_trials - done_before)
    if remaining == 0:
        print(f"[info] study already has {done_before} ≥ n_trials={n_trials}; skipping optimize")
    else:
        try:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        except KeyboardInterrupt:
            print("\n[warn] interrupted — partial study preserved at "
                  f"{study_db}", file=sys.stderr)

    # --- Pareto front + knee pick ---
    pareto = study.best_trials
    print()
    print(f"=== Pareto front ({len(pareto)} trials) ===")
    print(f"{'trial':>6} {'F1':>7} {'CER':>7}  overrides")
    for t in sorted(pareto, key=lambda tt: -(tt.values[0] if tt.values else -1)):
        f1, cer = t.values
        ov = ",".join(f"{k}={v}" for k, v in t.params.items())
        print(f"{t.number:>6} {f1:>6.2f}% {cer:>6.2f}%  {ov}")

    # write Pareto JSONL
    os.makedirs(os.path.dirname(pareto_out) or ".", exist_ok=True)
    with open(pareto_out, "w") as f:
        for t in pareto:
            f.write(json.dumps(_serialize_trial(t), ensure_ascii=False) + "\n")
    print(f"\n[ok] Pareto front -> {pareto_out}")

    knee = _knee_pick(pareto, base.autotune.cer_baseline)
    if knee is None:
        print("[error] no Pareto trials produced; nothing to persist", file=sys.stderr)
        return 2
    f1, cer = knee.values
    print(f"[ok] knee pick: trial {knee.number}  F1={f1:.2f}%  CER={cer:.2f}%  "
          f"(cer_baseline={base.autotune.cer_baseline:.2f}%)")

    tuned_cfg = merge_overrides(base, knee.params)
    tuned_cfg.name = f"{base.name}.tuned"
    os.makedirs(os.path.dirname(tuned_out) or ".", exist_ok=True)
    tuned_cfg.to_yaml(tuned_out)
    print(f"[ok] knee config -> {tuned_out}")
    for k, (a, b) in diff_config(base, tuned_cfg).items():
        if k == "name":
            continue
        print(f"       {k}: {a} -> {b}")

    # --- held-out final eval ---
    eval_dir = _expand(base.paths.eval_testset_dir, REPO_ROOT)
    if args.skip_eval or not eval_dir:
        if not eval_dir:
            print("\n[info] paths.eval_testset_dir is empty; skipping held-out eval")
        return 0
    if not os.path.isdir(eval_dir):
        print(f"[warn] eval_testset_dir not a directory: {eval_dir}", file=sys.stderr)
        return 0

    print(f"\n=== Held-out eval on {eval_dir} ===")
    eval_result = _run_trial(base, knee.params, trial_id=999999,
                             work_dir=work_dir, log_decoder=args.keep_decoder_logs,
                             testset_override=eval_dir)
    if eval_result.error:
        print(f"[error] eval failed: {eval_result.error}", file=sys.stderr)
        return 3
    print(f"F1={eval_result.f1:.2f}%  CER={eval_result.cer:.2f}%  "
          f"recall={eval_result.recall:.2f}%  precision={eval_result.precision:.2f}%  "
          f"wall={eval_result.wall_s:.1f}s")

    os.makedirs(os.path.dirname(eval_out) or ".", exist_ok=True)
    with open(eval_out, "w") as f:
        f.write(f"# held-out eval of {tuned_out} on {eval_dir}\n")
        f.write(f"# tune set: {_expand(base.paths.testset_dir, REPO_ROOT)} "
                f"(knee F1={f1:.2f}% CER={cer:.2f}%)\n")
        f.write(f"F1={eval_result.f1:.2f}\n")
        f.write(f"CER={eval_result.cer:.2f}\n")
        f.write(f"recall={eval_result.recall:.2f}\n")
        f.write(f"precision={eval_result.precision:.2f}\n")
        f.write(f"true_positives={eval_result.tp}\n")
        f.write(f"wall_s={eval_result.wall_s:.1f}\n")
    print(f"[ok] eval metrics -> {eval_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
