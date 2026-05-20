#!/usr/bin/env python3
"""Quick evaluation harness for single-config hotword pipeline experiments.

Runs decoder_main + CER + hotword metrics and reports structured results.
Designed for fast iteration: fixed model, fixed test set, reproducible output.

Usage:
    python tools/quick_eval.py --config runtime/libtorch/configs/multi_cn.tuned.yaml
    python tools/quick_eval.py --config cfg.yaml --override "hotword.bonus_weight=4.0"
    python tools/quick_eval.py --config cfg.yaml --baseline /tmp/baseline.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from decoder_config import DecoderConfig, merge_overrides

_CER_RE = re.compile(r"Overall\s*->\s*([0-9.]+)")


def _compute_cer(ref: str, hyp: str) -> Optional[float]:
    tool = os.path.join(REPO_ROOT, "tools", "compute-cer.py")
    try:
        out = subprocess.check_output(
            [sys.executable, tool, "--char=1", "--v=0", ref, hyp],
            stderr=subprocess.DEVNULL, text=True,
        )
    except subprocess.CalledProcessError:
        return None
    m = _CER_RE.search(out)
    return float(m.group(1)) if m else None


def _compute_hotword_metrics(ref: str, hyp: str, hotwords: str) -> Dict[str, float]:
    tool = os.path.join(REPO_ROOT, "tools", "compute-hotword-metrics.py")
    try:
        out = subprocess.check_output(
            [sys.executable, tool, "--hotword-list", hotwords, "--ref", ref, "--hyp", hyp],
            stderr=subprocess.DEVNULL, text=True,
        )
    except subprocess.CalledProcessError:
        return {}
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
        if "misses:" in line:
            try:
                result["misses"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "spurious insertions:" in line:
            try:
                result["spurious"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return result


def run(cfg_path: str, overrides: Optional[Dict[str, str]] = None,
        result_path: Optional[str] = None, tag: str = "") -> Dict:
    cfg = DecoderConfig.from_yaml(cfg_path)
    if overrides:
        typed_overrides = {}
        for k, v in overrides.items():
            # Best-effort type inference
            if v.lower() in ("true", "false"):
                typed_overrides[k] = v.lower() == "true"
            else:
                try:
                    typed_overrides[k] = int(v)
                except ValueError:
                    try:
                        typed_overrides[k] = float(v)
                    except ValueError:
                        typed_overrides[k] = v
        cfg = merge_overrides(cfg, typed_overrides)

    p = cfg.paths
    testset_dir = os.path.expanduser(p.testset_dir)
    if not os.path.isabs(testset_dir):
        testset_dir = os.path.join(REPO_ROOT, testset_dir)

    if result_path is None:
        out_dir = os.path.join(REPO_ROOT, p.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, f"harness_{tag or 'run'}.txt")

    bin_path = os.path.join(REPO_ROOT, p.decoder_bin)
    argv = [bin_path] + cfg.to_decoder_args(repo_root=REPO_ROOT, result_path=result_path)

    t0 = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True)
    wall = time.time() - t0

    if proc.returncode != 0:
        return {
            "tag": tag, "error": f"decoder exited {proc.returncode}",
            "stderr": proc.stderr[-500:] if proc.stderr else "",
        }

    ref_path = os.path.join(testset_dir, "text")
    hotword_path = cfg.hotword.hotword_path
    if not os.path.isabs(hotword_path):
        hotword_path = os.path.join(testset_dir, hotword_path)

    cer = _compute_cer(ref_path, result_path)
    metrics = _compute_hotword_metrics(ref_path, result_path, hotword_path)

    return {
        "tag": tag,
        "cer": cer,
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("F1"),
        "tp": metrics.get("tp"),
        "misses": metrics.get("misses"),
        "spurious": metrics.get("spurious"),
        "wall_s": round(wall, 1),
        "result_path": result_path,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--override", action="append", default=[],
                   help="Dotted override, e.g. hotword.bonus_weight=3.5")
    p.add_argument("--result", help="Hypothesis output path")
    p.add_argument("--tag", default="", help="Experiment tag")
    p.add_argument("--baseline", help="Baseline JSON to compare against")
    p.add_argument("--out", help="Write result JSON to file")
    args = p.parse_args()

    overrides = {}
    for o in args.override:
        if "=" not in o:
            raise ValueError(f"Override must contain '=': {o}")
        k, v = o.split("=", 1)
        overrides[k.strip()] = v.strip()

    result = run(args.config, overrides, args.result, args.tag)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        if result.get("stderr"):
            print(result["stderr"])
        return 1

    print(f"\n{'='*60}")
    print(f"Tag: {result['tag'] or '(no tag)'}")
    print(f"{'='*60}")
    print(f"CER       = {result['cer']:.2f}%")
    print(f"Recall    = {result['recall']:.2f}%")
    print(f"Precision = {result['precision']:.2f}%")
    print(f"F1        = {result['f1']:.2f}%")
    print(f"TP/Miss/Spur = {result['tp']}/{result['misses']}/{result['spurious']}")
    print(f"Wall      = {result['wall_s']:.1f}s")
    print(f"{'='*60}")

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        print(f"\nvs Baseline ({baseline.get('tag', '?')}):")
        d_cer = result["cer"] - baseline["cer"] if baseline.get("cer") else 0
        d_rec = result["recall"] - baseline["recall"] if baseline.get("recall") else 0
        d_prec = result["precision"] - baseline["precision"] if baseline.get("precision") else 0
        print(f"  CER       {d_cer:+.2f}%")
        print(f"  Recall    {d_rec:+.2f}%")
        print(f"  Precision {d_prec:+.2f}%")
        print(f"{'='*60}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
