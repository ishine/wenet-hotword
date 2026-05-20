"""YAML-backed dataclass config for `decoder_main` invocations.

Mirrors the section structure of `runtime/libtorch/configs/default.yaml`:

    paths      ->  PathConfig
    decode     ->  DecodeConfig
    hotword    ->  HotwordConfig
    runtime    ->  RuntimeConfig
    autotune   ->  AutotuneConfig

The loader prefers PyYAML when available (matches `requirements.txt`); a
minimal stdlib subset parser is included as a fallback so the autotuner runs
out-of-the-box on a clean Python install.

The whole point of the dataclass layer (vs. a raw dict) is `to_decoder_args`:
it produces the exact `decoder_main` argv that maps one-to-one with the gflags
declared in `runtime/core/decoder/params.h` / `decoder_main.cc`. Adding a knob
means editing both this file AND `default.yaml` — the dataclass is the schema.
"""

from __future__ import annotations

import copy
import json
import os
import re
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


# --- YAML I/O ----------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _minimal_yaml_load(path)


def _dump_yaml(data: Dict[str, Any], path: str) -> None:
    try:
        import yaml  # type: ignore
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True,
                           default_flow_style=False)
    except ImportError:
        with open(path, "w", encoding="utf-8") as f:
            _minimal_yaml_dump(data, f, indent=0)


def _minimal_yaml_load(path: str) -> Dict[str, Any]:
    """Read the subset of YAML used in this project: nested mappings, scalars,
    and inline `[a, b, c]` / block `- item` lists. Comments via `#`. No
    anchors, no flow-style mappings, no multi-line scalars.
    """
    root: Dict[str, Any] = {}
    # stack entries are (indent, container, parent_container, key_or_index)
    # so we can promote a dict placeholder to a list when the first child
    # turns out to be a block list item.
    stack: List[tuple] = [(-1, root, None, None)]

    def _scalar(s: str) -> Any:
        s = s.strip()
        if not s:
            return ""
        if s.startswith(("'", '"')) and s.endswith(s[0]) and len(s) >= 2:
            return s[1:-1]
        if s.lower() in ("true", "yes", "on"):
            return True
        if s.lower() in ("false", "no", "off"):
            return False
        if s.lower() in ("null", "~"):
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    def _parse_inline_list(s: str) -> List[Any]:
        s = s.strip()
        assert s.startswith("[") and s.endswith("]"), s
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_scalar(x) for x in _smart_split(inner, ",")]

    def _smart_split(s: str, sep: str) -> List[str]:
        out, buf, depth = [], [], 0
        in_str = None
        for ch in s:
            if in_str:
                buf.append(ch)
                if ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
            elif ch == "[":
                depth += 1
                buf.append(ch)
            elif ch == "]":
                depth -= 1
                buf.append(ch)
            elif ch == sep and depth == 0:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            # strip trailing comment (handles `key: value  # comment`)
            line = re.sub(r"\s+#.*$", "", raw.rstrip("\n"))
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1][0]:
                stack.pop()
            top_indent, top_container, top_parent, top_key = stack[-1]
            content = line.lstrip()

            if content.startswith("- "):
                # block list item. promote the placeholder to a list if needed.
                container = top_container
                if isinstance(container, dict):
                    if container:
                        raise ValueError(
                            f"unexpected list item at {path}: {raw!r} "
                            f"(parent dict already populated)")
                    container = []
                    if top_parent is not None:
                        top_parent[top_key] = container
                    stack[-1] = (top_indent, container, top_parent, top_key)
                container.append(_scalar(content[2:]))
                continue

            if ":" not in content:
                raise ValueError(f"unrecognized line in {path}: {raw!r}")
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            container = top_container
            if not isinstance(container, dict):
                raise ValueError(
                    f"expected mapping context for key {key!r} at {path}: {raw!r}")
            if rest == "":
                placeholder: Any = {}
                container[key] = placeholder
                stack.append((indent, placeholder, container, key))
            elif rest.startswith("["):
                container[key] = _parse_inline_list(rest)
            else:
                container[key] = _scalar(rest)
    return root


def _minimal_yaml_dump(node: Any, f, indent: int) -> None:
    pad = " " * indent
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict) and v:
                f.write(f"{pad}{k}:\n")
                _minimal_yaml_dump(v, f, indent + 2)
            elif isinstance(v, list):
                if not v:
                    f.write(f"{pad}{k}: []\n")
                elif all(isinstance(x, (int, float, str, bool)) for x in v):
                    rendered = ", ".join(_render_scalar(x) for x in v)
                    f.write(f"{pad}{k}: [{rendered}]\n")
                else:
                    f.write(f"{pad}{k}:\n")
                    for item in v:
                        f.write(f"{pad}- {_render_scalar(item)}\n")
            else:
                f.write(f"{pad}{k}: {_render_scalar(v)}\n")
    else:
        f.write(f"{pad}{_render_scalar(node)}\n")


def _render_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        if v == "" or any(c in v for c in ":#") or v.strip() != v:
            return json.dumps(v, ensure_ascii=False)
        return v
    return str(v)


# --- dataclasses --------------------------------------------------------------

@dataclass
class PathConfig:
    decoder_bin: str = "runtime/libtorch/build/bin/decoder_main"
    model_dir: str = ""
    testset_dir: str = ""
    eval_testset_dir: str = ""
    pinyin_dict_dir: str = "runtime/libtorch/build/bin/dict"
    out_dir: str = "runtime/libtorch/eval_runs"


@dataclass
class DecodeConfig:
    chunk_size: int = -1
    num_left_chunks: int = -1
    ctc_weight: float = 0.5
    rescoring_weight: float = 1.0
    reverse_weight: float = 0.0
    nbest: int = 10
    beam: float = 16.0
    lattice_beam: float = 10.0
    acoustic_scale: float = 1.0
    blank_skip_thresh: float = 1.0
    length_penalty: float = 0.0
    max_active: int = 7000
    min_active: int = 200


@dataclass
class HotwordConfig:
    hotword_path: str = "hotwords.txt"
    fuzzy_threshold: float = 0.5
    fuzzy_threshold_en: float = 0.5
    max_append_path: int = 20
    use_confidence_reward: bool = True
    context_score: float = 3.0
    enable_hotword_cache: bool = True
    # CSV of `from,to,cost`; empty leaves the built-in sparse fallback active.
    # Generated offline by `tools/learn_confusion.py` from CTC posteriors.
    confusion_matrix_path: str = ""
    # Master multiplier on CalculateMatchBonus (was hardcoded 2.0f).
    bonus_weight: float = 2.0
    # Lower bound on avg_confidence divisor (was hardcoded 0.4f).
    confidence_floor: float = 0.4
    # FastRAG neighbor-distance cutoff (was hardcoded kNeighborThreshold=0.5f).
    neighbor_threshold: float = 0.5
    # Edit-distance rejection ratio in fuzzy_substring_search_* (was 0.8f).
    fuzzy_reject_ratio: float = 0.8
    # Confidence-weight lower bound in weighted edit distance (was 0.2f).
    confidence_weight_min: float = 0.2
    # Linear scaling factor for hotword length bonus (replaces log2).
    bonus_length_scale: float = 0.5


@dataclass
class RuntimeConfig:
    thread_num: int = 0
    warmup: int = 0
    simulate_streaming: bool = False


@dataclass
class AutotuneConfig:
    n_trials: int = 100
    sampler: str = "nsga2"
    random_seed: int = 0
    cer_baseline: float = 14.20
    precision_floor: float = 0.0  # 0 = no floor; 95 = conservative hold-out guard
    study_name: str = "default"
    study_db: str = "runtime/libtorch/configs/default.study.db"
    tuned_config_out: str = "runtime/libtorch/configs/default.tuned.yaml"
    pareto_out: str = "runtime/libtorch/configs/default.pareto.jsonl"
    eval_metrics_out: str = "runtime/libtorch/configs/default.eval.txt"


@dataclass
class DecoderConfig:
    name: str = "default"
    paths: PathConfig = field(default_factory=PathConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    hotword: HotwordConfig = field(default_factory=HotwordConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    autotune: AutotuneConfig = field(default_factory=AutotuneConfig)

    # --- I/O --------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "DecoderConfig":
        return _from_dict(cls, _load_yaml(path))

    def to_yaml(self, path: str) -> None:
        _dump_yaml(_to_dict(self), path)

    def clone(self) -> "DecoderConfig":
        return copy.deepcopy(self)

    # --- dotted-path access (for search-space overrides) ------------------
    def get_dotted(self, key: str) -> Any:
        obj: Any = self
        for part in key.split("."):
            obj = getattr(obj, part)
        return obj

    def set_dotted(self, key: str, value: Any) -> None:
        parts = key.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        # coerce to the dataclass field's declared type if known
        hints = typing.get_type_hints(type(obj))
        leaf_type = hints.get(parts[-1])
        if leaf_type is not None:
            value = _coerce(value, leaf_type)
        setattr(obj, parts[-1], value)

    # --- argv translation -------------------------------------------------
    def to_decoder_args(self, *, repo_root: str, result_path: str,
                        wav_scp: Optional[str] = None) -> List[str]:
        """Produce the `decoder_main` argv (excluding the binary itself)."""
        p = self.paths
        h = self.hotword
        d = self.decode
        r = self.runtime
        model_dir = _expand(p.model_dir, repo_root)
        testset_dir = _expand(p.testset_dir, repo_root)
        pinyin_dict_dir = _expand(p.pinyin_dict_dir, repo_root)

        wav_scp = wav_scp or os.path.join(testset_dir, "wav.scp")

        args: List[str] = [
            "--chunk_size", str(d.chunk_size),
            "--num_left_chunks", str(d.num_left_chunks),
            "--ctc_weight", str(d.ctc_weight),
            "--rescoring_weight", str(d.rescoring_weight),
            "--reverse_weight", str(d.reverse_weight),
            "--nbest", str(d.nbest),
            "--beam", str(d.beam),
            "--lattice_beam", str(d.lattice_beam),
            "--acoustic_scale", str(d.acoustic_scale),
            "--blank_skip_thresh", str(d.blank_skip_thresh),
            "--length_penalty", str(d.length_penalty),
            "--max_active", str(d.max_active),
            "--min_active", str(d.min_active),
            "--model_path", os.path.join(model_dir, "final.zip"),
            "--unit_path", os.path.join(model_dir, "units.txt"),
            "--wav_scp", wav_scp,
            "--result", result_path,
            "--thread_num", str(r.thread_num or os.cpu_count() or 1),
            "--warmup", str(r.warmup),
            f"--simulate_streaming={'true' if r.simulate_streaming else 'false'}",
        ]

        # hotword pathway: only emit flags when the user actually configured one
        if h.hotword_path:
            hp = h.hotword_path
            if not os.path.isabs(hp):
                hp = os.path.join(testset_dir, hp)
            args += [
                "--hotword_path", hp,
                "--pinyin_dict_path", pinyin_dict_dir,
                "--fuzzy_threshold", str(h.fuzzy_threshold),
                "--fuzzy_threshold_en", str(h.fuzzy_threshold_en),
                "--max_append_path", str(h.max_append_path),
                "--bonus_weight", str(h.bonus_weight),
                "--confidence_floor", str(h.confidence_floor),
                "--neighbor_threshold", str(h.neighbor_threshold),
                "--fuzzy_reject_ratio", str(h.fuzzy_reject_ratio),
                "--confidence_weight_min", str(h.confidence_weight_min),
                "--bonus_length_scale", str(h.bonus_length_scale),
                f"--enable_hotword_cache={'true' if h.enable_hotword_cache else 'false'}",
            ]
            if h.confusion_matrix_path:
                # Repo-relative resolution: the matrix is a per-model artifact,
                # not a per-test-set one. Absolute paths pass through unchanged.
                cm = _expand(h.confusion_matrix_path, repo_root)
                args += ["--confusion_matrix_path", cm]
        return args


# --- helpers -----------------------------------------------------------------

def _expand(path: str, repo_root: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    return path


def _to_dict(node: Any) -> Any:
    if is_dataclass(node):
        return {fld.name: _to_dict(getattr(node, fld.name)) for fld in fields(node)}
    if isinstance(node, list):
        return [_to_dict(x) for x in node]
    return node


def _from_dict(cls: Any, data: Any):
    if data is None:
        return cls() if isinstance(cls, type) and is_dataclass(cls) else None
    if isinstance(cls, type) and is_dataclass(cls):
        if not isinstance(data, dict):
            raise TypeError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for fld in fields(cls):
            if fld.name in data:
                resolved = hints.get(fld.name, fld.type)
                kwargs[fld.name] = _from_dict(resolved, data[fld.name])
        return cls(**kwargs)
    return _coerce(data, cls)


_PRIMITIVES = (int, float, str, bool)


def _coerce(value: Any, declared) -> Any:
    """Best-effort coerce a YAML scalar to the dataclass field's declared type.

    Handles the cases that actually show up in our config: plain `int`/`float`/
    `bool`/`str`, and the forward-reference strings produced when a module uses
    `from __future__ import annotations` (which is us).
    """
    if value is None:
        return None
    if isinstance(declared, str):
        # forward-ref like "int" / "float" — map by name
        mapping = {"int": int, "float": float, "bool": bool, "str": str}
        declared = mapping.get(declared, str)
    if declared is bool and isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    if declared in _PRIMITIVES and not isinstance(value, declared):
        try:
            return declared(value)
        except (TypeError, ValueError):
            return value
    return value


def merge_overrides(cfg: DecoderConfig, overrides: Dict[str, Any]) -> DecoderConfig:
    """Apply a flat `{ "decode.ctc_weight": 0.3, ... }` dict to a clone of cfg."""
    out = cfg.clone()
    for k, v in overrides.items():
        out.set_dotted(k, v)
    return out


def diff_config(a: DecoderConfig, b: DecoderConfig) -> Dict[str, tuple]:
    """Return `{ dotted.path: (a_val, b_val) }` for fields that differ."""
    out: Dict[str, tuple] = {}
    _walk_diff("", _to_dict(a), _to_dict(b), out)
    return out


def _walk_diff(prefix: str, a: Any, b: Any, out: Dict[str, tuple]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            _walk_diff(f"{prefix}.{k}" if prefix else k, a.get(k), b.get(k), out)
    elif a != b:
        out[prefix] = (a, b)
