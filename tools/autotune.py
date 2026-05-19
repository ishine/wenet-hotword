#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-objective autotune for the hotword-pipeline decoder.

Reads a base config (`--config`) and a search space (`--search-space`),
runs `decoder_main` over `paths.testset_dir` for `n_trials` configurations
chosen by Optuna's TPE multivariate sampler, and optimizes (recall↑, CER↓)
jointly. F1 is logged as a user_attr for inspection but is not optimized;
recall is the load-bearing axis because the precision floor is saturated.

Outputs:
  * `autotune.tuned_config_out`   — best Pareto point as scored on the
                                    held-out eval set (not on tune-set knee)
  * `autotune.pareto_out`         — full Pareto front (JSONL) with both
                                    tune-set and held-out metrics per point
  * `autotune.eval_metrics_out`   — final config re-run on `eval_testset_dir`
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


# --- daemon helpers -----------------------------------------------------------

def _cfg_to_daemon_params(cfg: DecoderConfig) -> Dict[str, Any]:
    """Extract trial-varying decoder params for daemon JSON payload."""
    d = cfg.decode
    h = cfg.hotword
    testset_dir = _expand(cfg.paths.testset_dir, REPO_ROOT)
    hotword_path = h.hotword_path
    if hotword_path and not os.path.isabs(hotword_path):
        hotword_path = os.path.join(testset_dir, hotword_path)
    params: Dict[str, Any] = {
        "ctc_weight": d.ctc_weight,
        "rescoring_weight": d.rescoring_weight,
        "reverse_weight": d.reverse_weight,
        "length_penalty": d.length_penalty,
        "nbest": d.nbest,
        "chunk_size": d.chunk_size,
        "num_left_chunks": d.num_left_chunks,
        "fuzzy_threshold": h.fuzzy_threshold,
        "fuzzy_threshold_en": h.fuzzy_threshold_en,
        "max_append_path": h.max_append_path,
        "use_confidence_reward": h.use_confidence_reward,
        "bonus_weight": h.bonus_weight,
        "confidence_floor": h.confidence_floor,
        "neighbor_threshold": h.neighbor_threshold,
        "hotword_path": hotword_path,
        "confusion_matrix_path": _expand(h.confusion_matrix_path, REPO_ROOT),
        "enable_hotword_cache": h.enable_hotword_cache,
    }
    return params


def _start_daemon(cfg: DecoderConfig) -> subprocess.Popen:
    """Start decoder_main in daemon mode with heavy resources pre-loaded."""
    decoder_bin = _expand(cfg.paths.decoder_bin, REPO_ROOT)
    if not os.path.exists(decoder_bin):
        raise FileNotFoundError(f"decoder_main not found at {decoder_bin}")

    model_dir = _expand(cfg.paths.model_dir, REPO_ROOT)
    argv = [
        decoder_bin,
        "--daemon",
        "--model_path", os.path.join(model_dir, "final.zip"),
        "--unit_path", os.path.join(model_dir, "units.txt"),
        "--pinyin_dict_path", _expand(cfg.paths.pinyin_dict_dir, REPO_ROOT),
        "--thread_num", str(cfg.runtime.thread_num or os.cpu_count() or 1),
    ]
    if cfg.hotword.confusion_matrix_path:
        cm = _expand(cfg.hotword.confusion_matrix_path, REPO_ROOT)
        argv += ["--confusion_matrix_path", cm]

    return subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _daemon_decode(proc: subprocess.Popen, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send JSON payload to daemon and return parsed response."""
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    resp_line = proc.stdout.readline().strip()
    if not resp_line:
        return None
    return json.loads(resp_line)


# --- decoder invocation -------------------------------------------------------

def _run_trial(base: DecoderConfig, override: Dict[str, Any], trial_id: int,
               work_dir: str, log_decoder: bool,
               testset_override: Optional[str] = None,
               daemon_proc: Optional[subprocess.Popen] = None) -> TrialResult:
    """Run decoder_main once with `override` applied; collect CER + hotword
    metrics. `testset_override` swaps `paths.testset_dir` for the held-out
    final-eval pass."""
    cfg = merge_overrides(base, override)
    if testset_override:
        cfg.paths.testset_dir = testset_override
    tag = f"trial_{trial_id:04d}"
    hyp_path = os.path.join(work_dir, f"{tag}.txt")
    log_path = os.path.join(work_dir, f"{tag}.log")

    t0 = time.time()
    if daemon_proc is not None:
        # Daemon mode: send JSON payload, reuse loaded model
        testset_dir = _expand(cfg.paths.testset_dir, REPO_ROOT)
        wav_scp = os.path.join(testset_dir, "wav.scp")
        payload = {
            "wav_scp": wav_scp,
            "result": hyp_path,
            "params": _cfg_to_daemon_params(cfg),
        }
        try:
            resp = _daemon_decode(daemon_proc, payload)
            if resp is None:
                return TrialResult(trial_id, override, None, None, None, None, None,
                                   time.time() - t0,
                                   error="daemon returned empty response")
            if resp.get("status") != "ok":
                return TrialResult(trial_id, override, None, None, None, None, None,
                                   time.time() - t0,
                                   error=f"daemon error: {resp.get('message', 'unknown')}")
        except Exception as exc:
            return TrialResult(trial_id, override, None, None, None, None, None,
                               time.time() - t0,
                               error=f"daemon communication failed: {exc}")
    else:
        # Subprocess mode: cold-start decoder_main per trial
        decoder_bin = _expand(cfg.paths.decoder_bin, REPO_ROOT)
        if not os.path.exists(decoder_bin):
            return TrialResult(trial_id, override, None, None, None, None, None, 0.0,
                               error=f"decoder_main not found at {decoder_bin}")
        argv = [decoder_bin] + cfg.to_decoder_args(repo_root=REPO_ROOT,
                                                   result_path=hyp_path)
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
_FAIL_R, _FAIL_CER = -1.0, 1e6


def _build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name in ("tpe", "tpe_multi", "motpe"):
        # Multivariate TPE: learns joint density across knobs, handles
        # cat/int/float mixed types, multi-objective natively. `constant_liar`
        # avoids duplicate samples under n_jobs > 1.
        return TPESampler(seed=seed, multivariate=True, group=True,
                          constant_liar=True)
    if name in ("nsga2", "nsgaii", "nsga-ii"):
        return NSGAIISampler(seed=seed)
    raise ValueError(f"unknown sampler {name!r} (expected tpe | nsga2)")


def _knee_pick(trials: List[optuna.trial.FrozenTrial], cer_baseline: float,
               precision_floor: float = 0.0,
               ) -> Optional[optuna.trial.FrozenTrial]:
    """Pick the Pareto trial with the highest recall whose CER stays under
    `cer_baseline`. If `precision_floor > 0`, further require
    `trial.user_attrs['precision'] >= precision_floor`.
    Falls back to the trial closest to the baseline (lowest CER) if every
    Pareto trial violates the cap. Values are (recall, CER)."""
    if not trials:
        return None
    candidates = [t for t in trials if t.values is not None and t.values[1] <= cer_baseline]
    if precision_floor > 0:
        strict = [t for t in candidates
                  if t.user_attrs.get("precision", 0.0) >= precision_floor]
        if strict:
            return max(strict, key=lambda t: t.values[0])
        # precision floor was too strict; fall back to CER-only among capped
        if candidates:
            return max(candidates, key=lambda t: t.values[0])
    elif candidates:
        return max(candidates, key=lambda t: t.values[0])
    # everything blew through the CER cap; pick the lowest-CER point as a tie-break
    return min(trials, key=lambda t: t.values[1])


def _serialize_trial(t: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    recall, cer = (t.values or [None, None])[:2]
    rec = {
        "trial_id": t.number,
        "recall": recall,
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
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel trials (Optuna n_jobs). Each trial gets "
                        "nproc/n_jobs threads to avoid oversubscribing cores.")
    p.add_argument("--phase", choices=["model", "scene"], default="scene",
                   help="model = calibrate model-level params (recall↑ vs CER↓); "
                        "scene = fine-tune scene-level params "
                        "(recall vs CER Pareto).")
    p.add_argument("--daemon", action="store_true",
                   help="use decoder_main --daemon for model reuse across trials")
    p.add_argument("--model-config", default=None,
                   help="for --phase scene: YAML with frozen model-level params "
                        "(output of a prior --phase model run).")
    args = p.parse_args()

    base = DecoderConfig.from_yaml(args.config)

    # Phase setup
    if args.phase == "model":
        print("[info] phase = model  (multi-objective: recall↑, CER↓)")
    elif args.phase == "scene":
        if args.model_config:
            model_cfg = DecoderConfig.from_yaml(args.model_config)
            # Freeze model-level params into base config
            for dotted in ["decode.ctc_weight", "decode.rescoring_weight",
                           "decode.reverse_weight", "decode.length_penalty",
                           "decode.nbest", "hotword.confidence_floor",
                           "hotword.neighbor_threshold"]:
                val = model_cfg.get_dotted(dotted)
                base.set_dotted(dotted, val)
            print(f"[info] phase = scene  (model-level params frozen from {args.model_config})")
        else:
            print("[info] phase = scene  (no --model-config provided; using base config model params)")

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
    print(f"[info] n_jobs    = {args.n_jobs}")
    print(f"[info] search    = {len(space.space)} knobs ({', '.join(space.space)})")
    print(f"[info] study_db  = {study_db}")
    print(f"[info] work_dir  = {work_dir}")

    # Parallel trials oversubscribe the box if each decoder still grabs all
    # cores. Cap per-trial threads so n_jobs trials share the machine cleanly.
    # n_jobs=1 keeps the legacy "thread_num=0 -> nproc" behavior untouched.
    if args.n_jobs > 1:
        per_trial = max(1, (os.cpu_count() or 1) // args.n_jobs)
        base.runtime.thread_num = per_trial
        print(f"[info] thread_num per trial = {per_trial}")

    if args.dry_run:
        for k, spec in space.space.items():
            print(f"  {k}: {spec}")
        return 0

    # Study setup: multi-objective (recall↑, CER↓) for both phases
    study_name = f"{base.autotune.study_name}_{args.phase}"
    directions = ["maximize", "minimize"]  # recall, CER
    fail_val = (-1.0, 1e6)

    sampler = _build_sampler(sampler_name, seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_db}",
        directions=directions,
        sampler=sampler,
        load_if_exists=True,
    )
    done_before = len(study.trials)
    if done_before:
        print(f"[info] resuming study: {done_before} prior trials in {study_db}")

    # Start daemon if requested
    daemon_proc = None
    if args.daemon:
        try:
            daemon_proc = _start_daemon(base)
            print("[info] daemon started")
        except Exception as e:
            print(f"[error] failed to start daemon: {e}", file=sys.stderr)
            return 1

    # Status file for external monitoring
    status_file = os.path.join(work_dir, "autotune.status.json")

    def _write_status(trial_num: int, completed: int, best_recall: float,
                      best_cer: float, pareto_size: int, wall_s: float):
        import json as _json
        try:
            with open(status_file, "w") as f:
                _json.dump({
                    "phase": args.phase,
                    "trial_completed": completed,
                    "total_trials": n_trials,
                    "best_recall": best_recall,
                    "best_cer": best_cer,
                    "pareto_size": pareto_size,
                    "current_trial": trial_num,
                    "last_wall_s": wall_s,
                }, f, indent=2)
        except Exception:
            pass

    def objective(trial: optuna.Trial):
        override = space.suggest(trial)
        t = _run_trial(base, override, trial.number, work_dir,
                       args.keep_decoder_logs,
                       daemon_proc=daemon_proc)
        for k, v in {"f1": t.f1, "precision": t.precision, "tp": t.tp,
                     "wall_s": t.wall_s, "error": t.error}.items():
            if v is not None:
                trial.set_user_attr(k, v)
        if t.error or t.cer is None:
            print(f"[trial {trial.number}] error: {t.error or 'no metrics'}")
            _write_status(trial.number, trial.number, 0.0, 999.0, 0, 0.0)
            return fail_val
        if t.recall is None:
            print(f"[trial {trial.number}] error: no recall")
            _write_status(trial.number, trial.number, 0.0, 999.0, 0, 0.0)
            return fail_val
        print(f"[trial {trial.number}] R={t.recall:.2f}% CER={t.cer:.2f}% "
              f"F1={t.f1:.2f}% wall={t.wall_s:.1f}s  {override}")
        # Update best metrics
        completed = len([tr for tr in study.trials
                        if tr.state == optuna.trial.TrialState.COMPLETE])
        best_recall = max((tr.values[0] for tr in study.trials
                          if tr.state == optuna.trial.TrialState.COMPLETE
                          and tr.values is not None), default=0.0)
        best_cer = min((tr.values[1] for tr in study.trials
                       if tr.state == optuna.trial.TrialState.COMPLETE
                       and tr.values is not None), default=999.0)
        pareto = study.best_trials if hasattr(study, 'best_trials') else []
        _write_status(trial.number, completed, best_recall, best_cer,
                     len(pareto), t.wall_s)
        return t.recall, t.cer

    remaining = max(0, n_trials - done_before)
    if remaining == 0:
        print(f"[info] study already has {done_before} ≥ n_trials={n_trials}; skipping optimize")
    else:
        try:
            study.optimize(objective, n_trials=remaining,
                           n_jobs=args.n_jobs, gc_after_trial=True)
        except KeyboardInterrupt:
            print("\n[warn] interrupted — partial study preserved at "
                  f"{study_db}", file=sys.stderr)

    # --- Result selection (Pareto front for both phases) ---
    # --- Pareto front (tune-set) ---
    pareto = study.best_trials
    print()
    print(f"=== Pareto front ({len(pareto)} trials) ===")
    print(f"{'trial':>6} {'R':>7} {'CER':>7}  overrides")
    for t in sorted(pareto, key=lambda tt: -(tt.values[0] if tt.values else -1)):
        recall, cer = t.values
        ov = ",".join(f"{k}={v}" for k, v in t.params.items())
        print(f"{t.number:>6} {recall:>6.2f}% {cer:>6.2f}%  {ov}")

    if not pareto:
        print("[error] no Pareto trials produced; nothing to persist", file=sys.stderr)
        return 2

    # --- Held-out eval on EVERY Pareto point ---
    eval_dir = _expand(base.paths.eval_testset_dir, REPO_ROOT)
    do_holdout = (not args.skip_eval) and eval_dir and os.path.isdir(eval_dir)

    holdout_results: Dict[int, TrialResult] = {}
    if do_holdout:
        print(f"\n=== Pareto held-out eval on {eval_dir} ({len(pareto)} points) ===")
        print(f"{'trial':>6} {'tune R':>8} {'tune CER':>9}  "
              f"{'hold R':>8} {'hold CER':>9} {'hold P':>7} {'hold FP':>7}")
        for t in sorted(pareto, key=lambda tt: -(tt.values[0] if tt.values else -1)):
            tune_R, tune_CER = t.values
            res = _run_trial(base, t.params, trial_id=900000 + t.number,
                             work_dir=work_dir,
                             log_decoder=args.keep_decoder_logs,
                             testset_override=eval_dir,
                             daemon_proc=daemon_proc)
            holdout_results[t.number] = res
            if res.error or res.cer is None:
                print(f"{t.number:>6} {tune_R:>7.2f}% {tune_CER:>8.2f}%   "
                      f"error: {res.error}")
                continue
            fp = (res.tp * (100.0 - res.precision) / res.precision
                  if res.precision else None)
            print(f"{t.number:>6} {tune_R:>7.2f}% {tune_CER:>8.2f}%  "
                  f"{res.recall:>7.2f}% {res.cer:>8.2f}% {res.precision:>6.2f}% "
                  f"{fp if fp is None else f'{fp:>6.2f}'}")
    else:
        if not eval_dir:
            print("\n[info] paths.eval_testset_dir is empty; skipping held-out")

    # --- Pareto JSONL ---
    os.makedirs(os.path.dirname(pareto_out) or ".", exist_ok=True)
    with open(pareto_out, "w") as f:
        for t in pareto:
            rec = _serialize_trial(t)
            res = holdout_results.get(t.number)
            if res is not None and res.cer is not None:
                rec["holdout"] = {
                    "recall": res.recall, "cer": res.cer,
                    "precision": res.precision, "tp": res.tp,
                    "f1": res.f1, "wall_s": res.wall_s,
                }
            elif res is not None:
                rec["holdout_error"] = res.error
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[ok] Pareto front -> {pareto_out}")

    # --- Pick the final config ---
    cer_baseline = base.autotune.cer_baseline
    precision_floor = base.autotune.precision_floor
    pick_mode = "tune-set knee (no held-out)"
    if do_holdout and holdout_results:
        scored: List[tuple] = []
        for t in pareto:
            res = holdout_results.get(t.number)
            if res is None or res.cer is None or res.recall is None:
                continue
            scored.append((t, res))
        valid = [pr for pr in scored if pr[1].cer <= cer_baseline]
        if precision_floor > 0:
            strict = [pr for pr in valid
                      if pr[1].precision is not None and pr[1].precision >= precision_floor]
            if strict:
                best_pair = max(strict, key=lambda pr: pr[1].recall)
                pick_mode = (f"held-out knee (R↑ under CER ≤ {cer_baseline:.2f}, "
                             f"P ≥ {precision_floor:.0f}%)")
            elif valid:
                best_pair = max(valid, key=lambda pr: pr[1].recall)
                pick_mode = (f"held-out knee (R↑ under CER ≤ {cer_baseline:.2f}, "
                             f"P floor relaxed)")
            elif scored:
                best_pair = min(scored, key=lambda pr: pr[1].cer)
                pick_mode = f"held-out fallback (every Pareto blew CER cap; min CER)"
            else:
                best_pair = None
        else:
            if valid:
                best_pair = max(valid, key=lambda pr: pr[1].recall)
                pick_mode = f"held-out knee (R↑ under CER ≤ {cer_baseline:.2f})"
            elif scored:
                best_pair = min(scored, key=lambda pr: pr[1].cer)
                pick_mode = f"held-out fallback (every Pareto blew CER cap; min CER)"
            else:
                best_pair = None
        if best_pair is not None:
            best_trial, best_holdout = best_pair
        else:
            best_trial, best_holdout = _knee_pick(pareto, cer_baseline, precision_floor), None
    else:
        best_trial = _knee_pick(pareto, cer_baseline, precision_floor)
        best_holdout = None

    if best_trial is None:
        print("[error] no Pareto trials usable", file=sys.stderr)
        return 2

    tune_R, tune_CER = best_trial.values
    print(f"\n[ok] selected: trial {best_trial.number}  via {pick_mode}")
    print(f"       tune-set: R={tune_R:.2f}% CER={tune_CER:.2f}%")
    if best_holdout is not None:
        print(f"       held-out: R={best_holdout.recall:.2f}% "
              f"CER={best_holdout.cer:.2f}% P={best_holdout.precision:.2f}%")

    tuned_cfg = merge_overrides(base, best_trial.params)
    if args.phase == "model":
        tuned_cfg.name = f"{base.name}.model_tuned"
    else:
        tuned_cfg.name = f"{base.name}.tuned"
    os.makedirs(os.path.dirname(tuned_out) or ".", exist_ok=True)
    tuned_cfg.to_yaml(tuned_out)
    if args.phase == "model":
        print(f"[ok] model config -> {tuned_out}")
    else:
        print(f"[ok] selected config -> {tuned_out}")
    for k, (a, b) in diff_config(base, tuned_cfg).items():
        if k == "name":
            continue
        print(f"       {k}: {a} -> {b}")

    if not do_holdout:
        if daemon_proc is not None:
            daemon_proc.stdin.write("EXIT\n")
            daemon_proc.stdin.flush()
            daemon_proc.wait()
        return 0

    # --- Write eval_metrics_out ---
    res = best_holdout if best_holdout is not None else holdout_results.get(best_trial.number)
    if res is None:
        res = _run_trial(base, best_trial.params, trial_id=999999,
                         work_dir=work_dir, log_decoder=args.keep_decoder_logs,
                         testset_override=eval_dir,
                         daemon_proc=daemon_proc)
    if res.error:
        print(f"[error] eval failed: {res.error}", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(eval_out) or ".", exist_ok=True)
    with open(eval_out, "w") as f:
        f.write(f"# held-out eval of {tuned_out} on {eval_dir}\n")
        f.write(f"# tune set: {_expand(base.paths.testset_dir, REPO_ROOT)} "
                f"(selected trial {best_trial.number}: "
                f"R={tune_R:.2f}% CER={tune_CER:.2f}%; pick={pick_mode})\n")
        f.write(f"recall={res.recall:.2f}\n")
        f.write(f"CER={res.cer:.2f}\n")
        f.write(f"F1={res.f1:.2f}\n")
        f.write(f"precision={res.precision:.2f}\n")
        f.write(f"true_positives={res.tp}\n")
        f.write(f"wall_s={res.wall_s:.1f}\n")
    print(f"[ok] eval metrics -> {eval_out}")

    if daemon_proc is not None:
        daemon_proc.stdin.write("EXIT\n")
        daemon_proc.stdin.flush()
        daemon_proc.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
