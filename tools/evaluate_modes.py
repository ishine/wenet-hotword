#!/usr/bin/env python3
"""Batch evaluate all 4 tuned configs on a test set.

Usage:
    python3 tools/evaluate_modes.py \
        --test-dir ~/userspace/wenet/aishell2_eval/test1000 \
        --hotwords hotwords_500.txt

This runs decoder_main for each of the 4 tuned configs (aggressive, balanced,
conservative, ultra) plus a no-hotword baseline, then computes CER and hotword
metrics for each.
"""

import argparse
import os
import re
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))


def run_decoder(build: str, model_dir: str, wav_scp: str, hyp_path: str,
                cfg: dict, hotword_path: str = "", thread_num: int = 8) -> None:
    """Run decoder_main with config."""
    d = cfg.get("decode", {})
    h = cfg.get("hotword", {})

    argv = [
        build,
        "--chunk_size", str(d.get("chunk_size", -1)),
        "--num_left_chunks", str(d.get("num_left_chunks", -1)),
        "--ctc_weight", str(d.get("ctc_weight", 0.5)),
        "--rescoring_weight", str(d.get("rescoring_weight", 1.0)),
        "--reverse_weight", str(d.get("reverse_weight", 0.0)),
        "--nbest", str(d.get("nbest", 10)),
        "--beam", str(d.get("beam", 16.0)),
        "--lattice_beam", str(d.get("lattice_beam", 10.0)),
        "--acoustic_scale", str(d.get("acoustic_scale", 1.0)),
        "--blank_skip_thresh", str(d.get("blank_skip_thresh", 1.0)),
        "--length_penalty", str(d.get("length_penalty", 0.0)),
        "--max_active", str(d.get("max_active", 7000)),
        "--min_active", str(d.get("min_active", 200)),
        "--model_path", os.path.join(model_dir, "final.zip"),
        "--unit_path", os.path.join(model_dir, "units.txt"),
        "--wav_scp", wav_scp,
        "--result", hyp_path,
        "--thread_num", str(thread_num),
        "--warmup", "0",
        "--simulate_streaming=false",
    ]

    if hotword_path and h.get("hotword_path"):
        argv += [
            "--hotword_path", hotword_path,
            "--pinyin_dict_path", os.path.join(REPO_ROOT, "runtime/libtorch/build/bin/dict"),
            "--fuzzy_threshold", str(h.get("fuzzy_threshold", 0.5)),
            "--fuzzy_threshold_en", str(h.get("fuzzy_threshold_en", 0.5)),
            "--max_append_path", str(h.get("max_append_path", 20)),
            "--bonus_weight", str(h.get("bonus_weight", 2.0)),
            "--confidence_floor", str(h.get("confidence_floor", 0.4)),
            "--neighbor_threshold", str(h.get("neighbor_threshold", 0.5)),
            "--fuzzy_reject_ratio", str(h.get("fuzzy_reject_ratio", 0.8)),
            "--confidence_weight_min", str(h.get("confidence_weight_min", 0.2)),
            "--bonus_length_scale", str(h.get("bonus_length_scale", 0.5)),
            f"--enable_hotword_cache={'true' if h.get('enable_hotword_cache', True) else 'false'}",
        ]
        if h.get("confusion_matrix_path"):
            argv += ["--confusion_matrix_path", os.path.join(REPO_ROOT, h["confusion_matrix_path"])]

    import time
    n_utts = len(open(wav_scp).readlines())
    print(f"  -> Running decoder_main on {n_utts} utterances ...", file=sys.stderr)
    t0 = time.time()
    subprocess.run(argv, check=True, timeout=180)
    print(f"  -> Done in {time.time() - t0:.1f}s", file=sys.stderr)


def compute_cer(ref_path: str, hyp_path: str) -> float:
    out = subprocess.check_output(
        ["python3", os.path.join(REPO_ROOT, "tools/compute-cer.py"),
         "--char=1", "--v=0", ref_path, hyp_path],
        stderr=subprocess.DEVNULL, text=True, timeout=30)
    m = re.search(r'Overall\s*->\s*([0-9.]+)', out)
    return float(m.group(1)) if m else None


def compute_hw_metrics(ref_path: str, hyp_path: str, hotword_path: str) -> dict:
    out = subprocess.check_output(
        ["python3", os.path.join(REPO_ROOT, "tools/compute-hotword-metrics.py"),
         "--hotword-list", hotword_path, "--ref", ref_path, "--hyp", hyp_path],
        stderr=subprocess.DEVNULL, text=True, timeout=30)
    metrics = {}
    for line in out.splitlines():
        if "=" in line and "%" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().rstrip("%").strip()
            try:
                metrics[k] = float(v)
            except ValueError:
                pass
        for kw, mk in [("true positives:", "tp"), ("misses:", "miss"),
                       ("spurious insertions:", "fp"), ("occurrences ref:", "ref_occ"),
                       ("occurrences hyp:", "hyp_occ")]:
            if kw in line:
                try:
                    metrics[mk] = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", required=True, help="Test set directory (contains wav.scp + text)")
    p.add_argument("--hotwords", default="hotwords_500.txt", help="Hotword filename in test-dir")
    p.add_argument("--model-dir", default=os.path.expanduser(
        "~/userspace/wenet/models/u2pp_conformer-asr-cn-16k-online"))
    p.add_argument("--build", default=os.path.join(REPO_ROOT, "runtime/libtorch/build/bin/decoder_main"))
    p.add_argument("--thread-num", type=int, default=8)
    args = p.parse_args()

    test_dir = os.path.abspath(os.path.expanduser(args.test_dir))
    wav_scp = os.path.join(test_dir, "wav.scp")
    text = os.path.join(test_dir, "text")
    hotword_path = os.path.join(test_dir, args.hotwords)

    for f in [wav_scp, text]:
        if not os.path.exists(f):
            print(f"missing: {f}", file=sys.stderr)
            sys.exit(1)

    modes = [
        ("Baseline", None),
        ("Aggressive", "aggressive"),
        ("Balanced", "balanced"),
        ("Conservative", "conservative"),
        ("Ultra", "ultra"),
    ]

    results = []
    for name, key in modes:
        hyp = os.path.join("/tmp", f"eval_{key or 'baseline'}_hyp.txt")

        if key:
            cfg_path = os.path.join(REPO_ROOT, f"runtime/libtorch/configs/{key}.tuned.yaml")
            if not os.path.exists(cfg_path):
                print(f"{name}: missing {cfg_path}, skipping")
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            hw = hotword_path
        else:
            cfg = {}
            hw = ""

        print(f"\n=== {name} ===", file=sys.stderr)
        run_decoder(args.build, args.model_dir, wav_scp, hyp, cfg, hw, args.thread_num)

        cer = compute_cer(text, hyp)
        metrics = compute_hw_metrics(text, hyp, hotword_path) if hotword_path else {}
        results.append((name, cer, metrics))

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Mode':>12} {'CER%':>8} {'Ref':>6} {'Hyp':>6} {'TP':>5} {'Miss':>5} {'FP':>5} "
          f"{'Recall%':>9} {'Precision%':>11} {'F1%':>8}")
    print("-" * 100)
    for name, cer, m in results:
        print(f"{name:>12} {cer:>7.2f} {m.get('ref_occ', 0):>6} {m.get('hyp_occ', 0):>6} "
              f"{m.get('tp', 0):>5} {m.get('miss', 0):>5} {m.get('fp', 0):>5} "
              f"{m.get('recall', 0):>8.2f} {m.get('precision', 0):>10.2f} {m.get('F1', 0):>7.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
