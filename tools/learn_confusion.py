"""Learn a phoneme-confusion matrix from this model's CTC posteriors.

Pipeline per utterance (kept only when utt-level CER < `--cer_threshold`):

    fbank -> JIT encoder -> ctc.log_softmax -> (T, V)
    ref text -> token IDs -> torchaudio.forced_align -> per-frame ref labels
    for each ref-aligned non-blank frame t:
        target_char = units[align[t]]
        for each (alt_char, p_alt) in top-K of probs[t]:
            require {init,fin}(target_char) ∩ {init,fin}(alt_char) != ∅
            mass[ref_init][alt_init] += p_alt   (if both non-empty)
            mass[ref_fin ][alt_fin ] += p_alt   (if both non-empty)

After accumulation:

    cost(a, b) = clip(1 - mass[a][b] / mass[a][a], 0, 1)

Output is a 3-col CSV (`from,to,cost`) consumed by
`HotwordCorrection::LoadConfusionMatrix` (corrector.cc). Tones are dropped on
purpose — confusion matters at the initial/final level, not at tone level. The
splitter mirrors `PinyinSplitter::split` in corrector.cc.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as taF
from pypinyin import Style, lazy_pinyin


# Order matters: two-char initials must come first so the greedy prefix match in
# split_pinyin grabs them before "z"/"c"/"s".
INITIALS: Tuple[str, ...] = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
)


def split_pinyin(tone3: str) -> Tuple[str, str]:
    """Mirror of PinyinSplitter::split in corrector.cc. Returns (initial, final).

    Tone is stripped. Empty initial is legal (e.g. "er", "an", "a").
    """
    if not tone3:
        return "", ""
    s = tone3
    if s[-1].isdigit():
        s = s[:-1]
    if not s:
        return "", ""
    init, fin = "", s
    for cand in INITIALS:
        if s.startswith(cand):
            if cand == "r" and s == "er":  # "er" has no initial
                continue
            init, fin = cand, s[len(cand):]
            break
    # j/q/x/y + (v|yu) -> u  (matches the C++ canonicalization)
    if init in ("j", "q", "x", "y") and fin in ("v", "yu"):
        fin = "u"
    return init, fin


def char_to_init_fin(ch: str, cache: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    """pypinyin TONE3 + splitter, with a per-process cache."""
    hit = cache.get(ch)
    if hit is not None:
        return hit
    py = lazy_pinyin(ch, style=Style.TONE3, errors="ignore")
    if not py:
        cache[ch] = ("", "")
        return cache[ch]
    cache[ch] = split_pinyin(py[0])
    return cache[ch]


def load_units(path: str) -> Tuple[List[str], Dict[str, int]]:
    """Read units.txt. Returns (id_to_unit, unit_to_id). Blank id is 0."""
    id_to_unit: List[str] = []
    unit_to_id: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) < 2:
                continue
            unit, idx = parts[0], int(parts[1])
            while len(id_to_unit) <= idx:
                id_to_unit.append("")
            id_to_unit[idx] = unit
            unit_to_id[unit] = idx
    return id_to_unit, unit_to_id


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def tokenize_ref(ref: str, unit_to_id: Dict[str, int], unk_id: int) -> List[int]:
    """Map each char to its unit ID. Characters not in the table become <unk>."""
    return [unit_to_id.get(c, unk_id) for c in ref if c.strip()]


def load_wav(path: str, expected_sr: int = 16000) -> torch.Tensor:
    """Load mono int16 PCM as a 1D float32 tensor in the [-32768, 32767] range
    that Kaldi-compatible fbank expects."""
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"sample rate mismatch: {path} is {sr}Hz, expected {expected_sr}Hz")
    if data.ndim > 1:
        data = data[:, 0]
    return torch.from_numpy(data).float()


def compute_fbank(wav: torch.Tensor, num_bins: int = 80) -> torch.Tensor:
    """80-bin Kaldi fbank, 25 ms window, 10 ms shift -> (T, 80)."""
    x = wav.unsqueeze(0)
    fbank = kaldi.fbank(
        x,
        num_mel_bins=num_bins,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000.0,
    )
    return fbank


def encode(model, fbank: torch.Tensor, device: str) -> torch.Tensor:
    """Run encoder + ctc.log_softmax. Returns log-probs (T, V)."""
    xs = fbank.unsqueeze(0).to(device)
    xl = torch.tensor([fbank.size(0)], dtype=torch.int32, device=device)
    enc_out, _ = model.encoder(xs, xl, 0, -1)
    log_probs = model.ctc.log_softmax(enc_out)  # (1, T', V)
    return log_probs[0]


def force_align_frames(
    log_probs: torch.Tensor, target_ids: List[int], blank: int = 0
) -> Optional[torch.Tensor]:
    """Wrap torchaudio.forced_align. Returns a (T,) tensor of label IDs per
    frame (with `blank` at blank frames). Returns None when alignment fails,
    typically because T < L."""
    T = log_probs.size(0)
    L = len(target_ids)
    if L == 0 or T < L:
        return None
    lp = log_probs.unsqueeze(0)
    tg = torch.tensor([target_ids], dtype=torch.int32, device=log_probs.device)
    il = torch.tensor([T], dtype=torch.int32, device=log_probs.device)
    tl = torch.tensor([L], dtype=torch.int32, device=log_probs.device)
    try:
        ali, _ = taF.forced_align(lp, tg, il, tl, blank=blank)
    except RuntimeError:
        return None
    return ali[0].cpu()


def accumulate_pair(
    mass: Dict[str, Dict[str, float]],
    ref_ph: str,
    alt_ph: str,
    p: float,
) -> None:
    if ref_ph and alt_ph:
        mass[ref_ph][alt_ph] += p


def run_utt(
    model,
    uid: str,
    wav_path: str,
    ref_text: str,
    id_to_unit: List[str],
    unit_to_id: Dict[str, int],
    py_cache: Dict[str, Tuple[str, str]],
    *,
    top_k: int,
    token_count_mismatch_threshold: float,
    blank_id: int,
    unk_id: int,
    device: str,
    mass_init: Dict[str, Dict[str, float]],
    mass_fin: Dict[str, Dict[str, float]],
) -> Tuple[str, str, float, int, int]:
    """Process one utt. Returns (uid, hyp, cer, n_accepted_frames, n_filtered)."""
    wav = load_wav(wav_path)
    fbank = compute_fbank(wav)
    with torch.no_grad():
        log_probs = encode(model, fbank, device)  # (T, V)
    probs = log_probs.exp()
    argmax_ids = log_probs.argmax(dim=-1).cpu().tolist()

    # CTC greedy decode (collapse repeats, drop blanks).
    hyp_chars: List[str] = []
    prev = -1
    for tok in argmax_ids:
        if tok != prev and tok != blank_id:
            hyp_chars.append(id_to_unit[tok])
        prev = tok
    hyp = "".join(hyp_chars)

    # Token-count mismatch filter: when CER is high due to insertions/deletions,
    # forced alignment is unreliable. Reject utterances where greedy-decode token
    # count differs from reference by >= threshold.
    target_ids = tokenize_ref(ref_text, unit_to_id, unk_id)
    ref_token_count = len(target_ids)
    hyp_token_count = len(hyp_chars)
    u_cer = cer(ref_text, hyp)
    if ref_token_count > 0:
        mismatch = abs(ref_token_count - hyp_token_count) / ref_token_count
        if mismatch >= token_count_mismatch_threshold:
            return uid, hyp, u_cer, 0, 0

    align = force_align_frames(log_probs, target_ids, blank=blank_id)
    if align is None:
        return uid, hyp, u_cer, 0, 0

    n_accept = 0
    n_filter = 0
    T = log_probs.size(0)
    topk_p, topk_i = probs.topk(top_k, dim=-1)
    topk_p = topk_p.cpu()
    topk_i = topk_i.cpu()

    for t in range(T):
        tok = int(align[t].item())
        if tok == blank_id:
            continue
        ref_char = id_to_unit[tok]
        ref_i, ref_f = char_to_init_fin(ref_char, py_cache)
        if not ref_i and not ref_f:
            continue
        ref_set = {x for x in (ref_i, ref_f) if x}
        any_used = False
        for k in range(top_k):
            # Skip top-1: it self-fills m[a][a] at ~0.95/frame because the model
            # is ~95% accurate, which compresses every off-diagonal ratio into
            # [0, 0.05]. The diagonal mass we want is the one accumulated from
            # lower-rank alts that happen to share an initial or final with the
            # ref — that's the real "this phoneme attracts mass" signal.
            if k == 0:
                continue
            alt_id = int(topk_i[t, k].item())
            if alt_id == blank_id:
                continue
            alt_char = id_to_unit[alt_id]
            if not alt_char:
                continue
            alt_i, alt_f = char_to_init_fin(alt_char, py_cache)
            if not alt_i and not alt_f:
                continue
            alt_set = {x for x in (alt_i, alt_f) if x}
            if ref_set.isdisjoint(alt_set):
                n_filter += 1
                continue
            p = float(topk_p[t, k].item())
            accumulate_pair(mass_init, ref_i, alt_i, p)
            accumulate_pair(mass_fin, ref_f, alt_f, p)
            any_used = True
        if any_used:
            n_accept += 1
    return uid, hyp, u_cer, n_accept, n_filter


def normalize_and_write(
    mass_init: Dict[str, Dict[str, float]],
    mass_fin: Dict[str, Dict[str, float]],
    out_path: str,
    *,
    diag_min: float = 1e-6,
    tau: float = 0.02,
    c_low: float = 0.10,
    c_high: float = 0.50,
) -> Tuple[int, int]:
    """Two-step calibration: learn the relative ordering, then snap to a DP-usable range.

    step 1 (raw):  r(a,b) = mass[a][b] / mass[a][a]    for a != b
    step 2 (per-matrix log-affine remap):
        drop pairs with r < tau                            (signal too weak)
        let r_max = max ratio in this matrix
        t = (log r - log tau) / (log r_max - log tau)      in [0, 1]
        cost = c_high - (c_high - c_low) * t               in [c_low, c_high]

    Why this shape: corrector.cc's DP uses ins/del = 1.0, so cost ∈ [0,1] is the
    full range, and the score filter (`dist >= n*0.8 → reject`) means anything
    above ~0.7 barely contributes versus a hard reject. [c_low, c_high] is
    sized to land inside the sparse fallback's working band (0.1-0.3 for the
    strong confusions, 0.5 still nudges the DP).

    Why log on r: raw ratios span orders of magnitude (top ≈ 1+, typical ≈
    1e-2). Linear remap would crush the head into a tiny window; log spreads
    the top quartile across most of [c_low, c_high] while still letting the
    weakest kept pairs occupy the high end.

    Why per-matrix: init has 23 phonemes, finals ~35. Sharing r_max across
    both biases whichever side has the wider raw spread. Calibrating
    independently lets each matrix use its full output range.
    """
    import math
    rows: List[Tuple[str, str, float]] = []

    def emit(matrix: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
        pairs: List[Tuple[str, str, float]] = []
        for a, row in matrix.items():
            diag = row.get(a, 0.0)
            if diag < diag_min:
                continue
            for b, m in row.items():
                if a == b or m <= 0.0:
                    continue
                r = m / diag
                if r < tau:
                    continue
                pairs.append((a, b, r))
        if not pairs:
            return []
        r_max = max(r for _, _, r in pairs)
        log_tau = math.log(tau)
        log_max = math.log(r_max)
        denom = log_max - log_tau
        if denom <= 0.0:
            return [(a, b, c_high) for a, b, _ in pairs]
        out = []
        for a, b, r in pairs:
            t = (math.log(r) - log_tau) / denom
            t = max(0.0, min(1.0, t))
            cost = c_high - (c_high - c_low) * t
            out.append((a, b, cost))
        return out

    rows_i = emit(mass_init)
    rows_f = emit(mass_fin)
    rows = rows_i + rows_f

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# learned phoneme confusion matrix (3-col: from,to,cost)\n")
        f.write("# generated by tools/learn_confusion.py; see corrector.cc::LoadConfusionMatrix\n")
        f.write(f"# tau={tau} c_low={c_low} c_high={c_high}\n")
        w = csv.writer(f)
        for a, b, c in rows:
            w.writerow([a, b, f"{c:.4f}"])
    return len(rows_i), len(rows_f)


def dump_masses(
    mass_init: Dict[str, Dict[str, float]],
    mass_fin: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    """Write raw masses to a JSON sidecar so calibration can be re-run without
    re-encoding wavs. Lets us sweep tau / c_low / c_high cheaply."""
    import json
    payload = {
        "mass_init": {a: dict(row) for a, row in mass_init.items()},
        "mass_fin":  {a: dict(row) for a, row in mass_fin.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="contains final.zip + units.txt")
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--text", required=True,
                    help="kaldi-style ref text (uid<space>chars)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--token_count_mismatch_threshold", type=float, default=0.1,
                    help="drop utts where |ref_tokens - hyp_tokens| / ref_tokens >= this")
    ap.add_argument("--top_k", type=int, default=20,
                    help="alternates per frame to score")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="raw-ratio cutoff: drop pairs with m[a][b]/m[a][a] < tau")
    ap.add_argument("--c_low", type=float, default=0.10,
                    help="output cost of the strongest confusion (top ratio)")
    ap.add_argument("--c_high", type=float, default=0.50,
                    help="output cost of the weakest kept confusion (just above tau)")
    ap.add_argument("--dump_masses", default="",
                    help="optional sidecar JSON path for raw masses")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_utts", type=int, default=0,
                    help="cap utt count for smoke runs (0 = no cap)")
    ap.add_argument("--report_every", type=int, default=25)
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, "final.zip")
    units_path = os.path.join(args.model_dir, "units.txt")
    if not os.path.isfile(model_path):
        print(f"missing model: {model_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(units_path):
        print(f"missing units: {units_path}", file=sys.stderr)
        return 2

    print(f"loading {model_path} on {args.device}", flush=True)
    model = torch.jit.load(model_path, map_location=args.device)
    model.eval()
    id_to_unit, unit_to_id = load_units(units_path)
    blank_id = unit_to_id.get("<blank>", 0)
    unk_id = unit_to_id.get("<unk>", 1)
    print(f"units: {len(id_to_unit)} (blank={blank_id}, unk={unk_id})", flush=True)

    wav_lookup: Dict[str, str] = {}
    with open(args.wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                wav_lookup[parts[0]] = parts[1]
    text_lookup: Dict[str, str] = {}
    with open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, _, ref = line.partition(" ")
            text_lookup[uid] = ref.replace(" ", "")

    uids = [u for u in wav_lookup if u in text_lookup]
    if args.max_utts:
        uids = uids[:args.max_utts]
    print(f"utts to process: {len(uids)}", flush=True)

    mass_init: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mass_fin: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    py_cache: Dict[str, Tuple[str, str]] = {}

    n_kept = 0
    n_skipped = 0
    total_frames = 0
    total_filter = 0
    t0 = time.time()
    for i, uid in enumerate(uids, 1):
        try:
            _, hyp, c, nfr, nfilt = run_utt(
                model, uid, wav_lookup[uid], text_lookup[uid],
                id_to_unit, unit_to_id, py_cache,
                top_k=args.top_k, token_count_mismatch_threshold=args.token_count_mismatch_threshold,
                blank_id=blank_id, unk_id=unk_id, device=args.device,
                mass_init=mass_init, mass_fin=mass_fin,
            )
        except Exception as e:
            print(f"  [{uid}] error: {e}", flush=True)
            n_skipped += 1
            continue
        if nfr == 0:
            n_skipped += 1
        else:
            n_kept += 1
            total_frames += nfr
            total_filter += nfilt
        if i % args.report_every == 0 or i == len(uids):
            elapsed = time.time() - t0
            print(
                f"  [{i}/{len(uids)}] kept={n_kept} skipped={n_skipped} "
                f"frames={total_frames} filt={total_filter} "
                f"({elapsed:.1f}s, {i/elapsed:.2f} utt/s)",
                flush=True,
            )

    if n_kept == 0:
        print("no utts kept; check token_count_mismatch_threshold / inputs", file=sys.stderr)
        return 3

    n_i, n_f = normalize_and_write(mass_init, mass_fin, args.out_csv,
                                   tau=args.tau, c_low=args.c_low, c_high=args.c_high)
    if args.dump_masses:
        dump_masses(mass_init, mass_fin, args.dump_masses)
    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.1f}s; kept {n_kept}/{len(uids)} utts, "
        f"frames={total_frames}, init_rows={n_i}, fin_rows={n_f}",
        flush=True,
    )
    print(f"wrote {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
