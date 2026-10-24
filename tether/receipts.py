"""Inspectable metric arithmetic and byte-integrity receipts.

Hashes are integrity checks, not attestations of training or scientific truth.
Strict checkpoint inference replay lives in experiment.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("invalid artifact path")
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError("artifact path escapes study")
    return target


def seal_files(root: Path, paths) -> dict:
    root = Path(root)
    files = {str(p).replace("\\", "/"): sha256(safe_artifact(root, str(p))) for p in sorted(paths)}
    if not files or "receipt.json" in files:
        raise ValueError("receipt needs payload files and cannot contain itself")
    receipt = {"schema": "supermix-tether-integrity-v1", "files": files,
               "meaning": "Byte consistency only; use strict inference replay for model behavior."}
    atomic_json(root / "receipt.json", receipt)
    return receipt


def verify_files(root: Path) -> dict:
    root = Path(root)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != "supermix-tether-integrity-v1" or not receipt.get("files"):
        raise ValueError("unsupported or empty receipt")
    for name, expected in receipt["files"].items():
        path = safe_artifact(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    return {"files_verified": len(receipt["files"])}


def _probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def metrics_from_trace(trace: dict[str, np.ndarray]) -> dict[str, float]:
    views = ("factual", "relevant", "irrelevant", "damaged", "erased", "shuffled")
    try:
        target = np.asarray(trace["target"])
        relevant_target = np.asarray(trace["relevant_target"])
        arrays = {v: np.asarray(trace[v]) for v in views}
    except KeyError as error:
        raise ValueError(f"incomplete trace: {error}") from error
    shape = arrays["factual"].shape
    if len(shape) != 2 or not shape[0] or shape[1] < 2:
        raise ValueError("invalid trace logits")
    if target.shape != (shape[0],) or relevant_target.shape != target.shape:
        raise ValueError("invalid trace targets")
    if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(relevant_target.dtype, np.integer):
        raise ValueError("trace targets must be integers")
    if (target < 0).any() or (target >= shape[1]).any() or (relevant_target < 0).any() or (relevant_target >= shape[1]).any():
        raise ValueError("trace targets out of range")
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite or mismatched trace")
    predictions = {v: arrays[v].argmax(axis=1) for v in views}
    factual_correct = predictions["factual"] == target
    probabilities = {v: _probabilities(arrays[v]) for v in views[:3]}
    expected_effect = np.eye(shape[1])[relevant_target] - np.eye(shape[1])[target]
    actual_effect = probabilities["relevant"] - probabilities["factual"]
    accuracy = float(factual_correct.mean())
    erased = float((predictions["erased"] == target).mean())
    return {
        "accuracy": accuracy,
        "relevant_accuracy": float((predictions["relevant"] == relevant_target).mean()),
        "irrelevant_accuracy": float((predictions["irrelevant"] == target).mean()),
        "irrelevant_consistency": float((predictions["irrelevant"] == predictions["factual"]).mean()),
        "damaged_accuracy": float((predictions["damaged"] == target).mean()),
        "donor_erased_accuracy": erased,
        "donor_erasure_gap": accuracy - erased,
        "selected_shuffle_accuracy": float((predictions["shuffled"] == target).mean()),
        "correct_pair_rate": float((factual_correct & (predictions["relevant"] == relevant_target)).mean()),
        "effect_error": float(np.square(actual_effect - expected_effect).sum(axis=1).mean()),
        "invariance_error": float(np.square(probabilities["irrelevant"] - probabilities["factual"]).sum(axis=1).mean()),
    }


def paired_comparisons(rows: list[dict], bootstrap_samples: int = 2000) -> list[dict]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    keyed = {}
    for row in rows:
        key = (row["seed"], row["arm"])
        if key in keyed:
            raise ValueError("duplicate paired row")
        keyed[key] = row
    if not any(arm == "tether" for _, arm in keyed):
        return []
    comparisons = []
    for baseline in ("iit", "unpaired"):
        if not any(arm == baseline for _, arm in keyed):
            continue
        seeds = sorted({s for s, a in keyed if a in (baseline, "tether")})
        if any((s, a) not in keyed for s in seeds for a in (baseline, "tether")):
            raise ValueError("missing paired seed")
        first = keyed[(seeds[0], "tether")]["metrics"]
        for condition in sorted(first):
            for metric in sorted(first[condition]):
                differences = np.array([keyed[(s, "tether")]["metrics"][condition][metric] - keyed[(s, baseline)]["metrics"][condition][metric] for s in seeds])
                if not np.isfinite(differences).all():
                    raise ValueError("nonfinite paired metric")
                rng = np.random.default_rng(9041)
                means = rng.choice(differences, size=(bootstrap_samples, len(seeds)), replace=True).mean(axis=1)
                comparisons.append({"metric": f"{condition}.{metric}", "arm": "tether", "baseline": baseline, "seeds": seeds,
                                    "differences": differences.tolist(), "mean_difference": float(differences.mean()),
                                    "interval": np.quantile(means, [.025, .975]).tolist(),
                                    "interpretation": "descriptive paired seed bootstrap; not an anytime-valid or familywise guarantee"})
    return comparisons
