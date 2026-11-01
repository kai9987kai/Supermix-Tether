"""Matched training studies, validation-only selection and frozen inference replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

import numpy as np
import sklearn
import torch
from torch.nn import functional as F

from . import __version__
from .conformal import (
    adaptive_conformal_inference,
    calibration_metrics,
    conformal_quantile,
    evaluate_conformal_sets,
    evaluate_nex_conformal_sets,
    fit_mahalanobis_forensics,
    fit_temperature,
    nex_raps_prediction_sets,
    raps_nonconformity_scores,
    raps_prediction_sets,
    sanitize_specialist_posteriors,
    score_mahalanobis_forensics,
    softmax_temperature,
)
from .data import DonorBank, fit_donors, load_digits_split, make_batch
from .model import TetherReceiver, objective
from .receipts import atomic_json, metrics_from_trace, paired_comparisons, safe_artifact, seal_files, sha256, verify_files

_CALIBRATION_CACHE: dict[tuple, dict] = {}
_MAHALANOBIS_CACHE: dict[tuple, dict] = {}
_RESTORE_CACHE: dict[tuple, tuple] = {}


ARMS = ("passive", "dropout", "iit", "effect_only", "invariance_only", "unpaired", "tether")
CONDITIONS = {
    "seen_correlated": ("seen", "correlated", 1, False),
    "seen_independent": ("seen", "independent", 1, False),
    "seen_reversed": ("seen", "reversed", 1, False),
    "heldout_independent": ("heldout", "independent", 1, False),
    "heldout_reversed": ("heldout", "reversed", 1, False),
    "damage_one": ("heldout", "independent", 1, True),
    "damage_common": ("heldout", "independent", 3, True),
}
CONDITIONS_V2 = {
    "seen_correlated": ("seen", "correlated", 1, False),
    "seen_independent": ("seen", "independent", 1, False),
    "seen_reversed": ("seen", "reversed", 1, False),
    "heldout_independent": ("heldout", "independent", 1, False),
    "heldout_reversed": ("heldout", "reversed", 1, False),
    "damage_one": ("heldout", "independent", 1, True),
    "damage_two": ("heldout", "independent", 2, True),
    "damage_common": ("heldout", "independent", 3, True),
}


def _conditions(protocol: str) -> dict[str, tuple[str, str, int, bool]]:
    return CONDITIONS_V2 if protocol == "v2" else CONDITIONS


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...] = (11, 12, 13)
    arms: tuple[str, ...] = ARMS
    steps: int = 400
    donor_steps: int = 180
    batch_size: int = 128
    eval_size: int = 512
    validation_every: int = 50
    learning_rate: float = .02
    effect_weight: float = 1.
    invariance_weight: float = 1.
    threads: int = 2
    data_seed: int = 1729
    phase: str = "development"
    protocol: str = "v1"

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "arms", tuple(self.arms))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
        if not self.arms or len(set(self.arms)) != len(self.arms) or any(a not in ARMS for a in self.arms):
            raise ValueError("unknown or duplicate arms")
        for name in ("steps", "donor_steps", "batch_size", "eval_size", "validation_every", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.threads > 8:
            raise ValueError("at most eight CPU threads")
        for name in ("learning_rate", "effect_weight", "invariance_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"invalid {name}")
        if self.phase not in ("development", "confirmation"):
            raise ValueError("phase must be development or confirmation")
        if self.protocol not in ("v1", "v2"):
            raise ValueError("protocol must be v1 or v2")
        if type(self.data_seed) is not int or not 0 <= self.data_seed < 2**31:
            raise ValueError("invalid data_seed")


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "tether").rglob("*.py")) + sorted((root / "tether" / "web").glob("*"))
    if (root / "pyproject.toml").is_file():
        paths.append(root / "pyproject.toml")
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths if p.is_file()}


def _event_seed(seed: int, event: str) -> int:
    """Stable bounded substream ID; large public seeds cannot overflow data RNGs."""
    return int.from_bytes(hashlib.blake2s(f"{seed}:{event}".encode(), digest_size=4).digest(), "little")


def _batch(posteriors, split, config, seed, condition, *, validation=False):
    pairs, cue, damage, damaged = CONDITIONS_V2[condition]
    name = "validation" if validation or config.phase == "development" else "test"
    # Damage comparisons reuse the exact clean tasks and fixed corruption across
    # views. An irrelevant swap must not simultaneously repair the queried source.
    task_condition = "heldout_independent" if damaged else condition
    batch = make_batch(posteriors, split["labels"], split[name],
                       n=min(config.eval_size, 256) if validation else config.eval_size,
                       seed=_event_seed(seed, f"{name}:{task_condition}"),
                       split=name, cue=cue, pairs=pairs, damage_count=damage)
    if damaged:
        batch = dict(batch)
        batch["donors"] = batch["damaged"].clone()
        rows = torch.arange(len(batch["donors"]))[:, None]
        sources = batch["query"][:, 0, None]
        experts = batch["damaged_experts"]
        corrupted = batch["damaged"][rows, sources, experts]
        for key in ("relevant", "irrelevant"):
            batch[key] = batch[key].clone()
            batch[key][rows, sources, experts] = corrupted
        batch["manifest"] = {**batch["manifest"], "paired_damage": True,
                             "corruption_retained_across_views": True}
    return batch


@torch.no_grad()
def _trace(model, batch):
    model.eval()
    traces = {name: model(batch[key], batch["query"], batch["context"]).cpu().numpy()
              for name, key in (("factual", "donors"), ("relevant", "relevant"), ("irrelevant", "irrelevant"), ("damaged", "damaged"))}
    traces["erased"] = model(torch.full_like(batch["donors"], .1), batch["query"], batch["context"]).cpu().numpy()
    shuffled = batch["donors"].clone()
    index = torch.arange(len(shuffled))
    source = batch["query"][:, 0]
    shuffled[index, source] = shuffled[index, source].roll(1, dims=0)
    traces["shuffled"] = model(shuffled, batch["query"], batch["context"]).cpu().numpy()
    traces["target"] = batch["target"].cpu().numpy()
    traces["relevant_target"] = batch["relevant_target"].cpu().numpy()
    return traces


def evaluate_model(model, batch) -> dict:
    return metrics_from_trace(_trace(model, batch))


def _train_receiver(posteriors, split, config, seed, arm):
    forensic = config.protocol == "v2"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TetherReceiver(forensic=forensic)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_hash = _state_digest(model.state_dict())
    validation = _batch(posteriors, split, config, seed, "seen_independent", validation=True)
    best_score, best_state, best_step = float("inf"), None, 0
    history = []
    start = time.perf_counter()
    for step in range(config.steps):
        batch = make_batch(posteriors, split["labels"], split["train"], n=config.batch_size,
                           seed=_event_seed(seed, f"train:{step}"), split="train", cue="correlated", pairs="seen")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, details = objective(model, batch, arm, effect_weight=config.effect_weight,
                                  invariance_weight=config.invariance_weight, protocol=config.protocol)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss: seed={seed} arm={arm} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if (step + 1) % config.validation_every == 0 or step + 1 == config.steps:
            model.eval()
            with torch.no_grad():
                validation_loss = float(F.cross_entropy(model(validation["donors"], validation["query"], validation["context"]), validation["target"]))
            history.append({"step": step + 1, "train": details, "validation_seen_independent_ce": validation_loss})
            if validation_loss < best_score:
                best_score, best_step = validation_loss, step + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_step": best_step, "validation_loss": best_score, "history": history,
                   "training_seconds": elapsed, "initial_state_sha256": initial_hash,
                   "receiver_parameters": sum(p.numel() for p in model.parameters()),
                   "receiver_training_views": 4 * config.steps * config.batch_size,
                   "training_steps": config.steps, "protocol": config.protocol,
                   "selection": "minimum CE on seen-combination independent-cue validation images only"}


def _state_digest(state) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _decision(rows, comparisons, phase, protocol="v1"):
    by_metric = {(r["baseline"], r["metric"]): r for r in comparisons}
    primary = by_metric.get(("iit", "heldout_independent.accuracy"))
    clean = by_metric.get(("iit", "seen_correlated.accuracy"))
    mechanistic = by_metric.get(("unpaired", "heldout_independent.accuracy"))
    tether_rows = [r for r in rows if r["arm"] == "tether"]
    has_unpaired = any(r["arm"] == "unpaired" for r in rows)
    checks = {
        "primary_interval_above_zero": bool(primary and primary["interval"][0] > 0),
        "clean_mean_noninferior_at_two_points": bool(clean and clean["mean_difference"] >= -.02),
        "irrelevant_consistency_at_least_95_percent": bool(tether_rows and np.mean([r["metrics"]["heldout_independent"]["irrelevant_consistency"] for r in tether_rows]) >= .95),
        "positive_donor_utility_above_ten_points": bool(tether_rows and np.mean([r["metrics"]["heldout_independent"]["donor_erasure_gap"] for r in tether_rows]) > .10),
    }
    if has_unpaired or protocol == "v2":
        checks["pairing_beats_unpaired_control"] = bool(mechanistic and mechanistic["interval"][0] > 0)
    if protocol == "v2":
        d2_iit = by_metric.get(("iit", "damage_two.accuracy"))
        d2_unp = by_metric.get(("unpaired", "damage_two.accuracy"))
        checks["zero_shot_damage_two_beats_iit"] = bool(d2_iit and d2_iit["interval"][0] > 0)
        checks["zero_shot_damage_two_beats_unpaired"] = bool(d2_unp and d2_unp["interval"][0] > 0)
    return {"phase": phase, "promoted": False, "checks": checks,
            "candidate_supported_in_this_study": bool(all(checks.values())),
            "primary": "tether minus iit heldout_independent accuracy",
            "scope": "Finite same-dataset study. Seed bootstrap is descriptive. No automatic promotion, general capability or global novelty claim.",
            "status": "supported_for_further_study" if all(checks.values()) else "hypothesis_not_supported_by_all_predefined_checks"}


def run_study(config: StudyConfig, output_dir: Path, progress: Callable | None = None) -> dict:
    output_dir = Path(output_dir).resolve()
    # Exclusive directory creation is the reservation; concurrent runs cannot
    # share checkpoints, temporary writes or partial progress by accident.
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    manifest = source_manifest()
    conditions = _conditions(config.protocol)
    atomic_json(output_dir / "status.json", {"status": "running", "started_at": started, "config": asdict(config)})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    rows, payloads = [], []
    try:
        split = load_digits_split(seed=config.data_seed)
        atomic_json(output_dir / "split.json", split["manifest"])
        payloads.append("split.json")
        for seed in config.seeds:
            if progress:
                progress({"stage": "donors", "seed": seed})
            bank, posteriors, donor_metrics = fit_donors(split, seed=_event_seed(seed, "donors"), steps=config.donor_steps)
            bank.eval()
            for parameter in bank.parameters():
                parameter.requires_grad_(False)
            for arm in config.arms:
                if progress:
                    progress({"stage": "receiver", "seed": seed, "arm": arm, "completed_rows": len(rows)})
                model, training = _train_receiver(posteriors.detach(), split, config, seed, arm)
                checkpoint_name = f"checkpoints/seed-{seed}-{arm}.pt"
                checkpoint = {"schema": "supermix-tether-checkpoint-v1", "version": __version__, "seed": seed, "arm": arm,
                              "bank_config": bank.config, "bank_state": bank.state_dict(), "model_state": model.state_dict(),
                              "config": asdict(config), "dataset_manifest": split["manifest"], "source_manifest": manifest,
                              "training": training}
                _atomic_checkpoint(output_dir / checkpoint_name, checkpoint)
                metrics, arrays = {}, {}
                for condition in conditions:
                    trace = _trace(model, _batch(posteriors, split, config, seed, condition))
                    metrics[condition] = metrics_from_trace(trace)
                    arrays.update({f"{condition}__{key}": value for key, value in trace.items()})
                trace_name = f"traces/seed-{seed}-{arm}.npz"
                (output_dir / "traces").mkdir(exist_ok=True)
                np.savez_compressed(output_dir / trace_name, **arrays)
                row = {"seed": seed, "arm": arm, "metrics": metrics, "training": training, "donors": donor_metrics,
                       "checkpoint": checkpoint_name, "checkpoint_sha256": sha256(output_dir / checkpoint_name),
                       "trace": trace_name, "trace_sha256": sha256(output_dir / trace_name)}
                rows.append(row)
                payloads.extend((checkpoint_name, trace_name))
                atomic_json(output_dir / "progress.json", {"status": "running", "rows": rows, "config": asdict(config)})
        if manifest != source_manifest():
            raise RuntimeError("source changed during study; preserve partial artifacts and start a fresh run")
        comparisons = paired_comparisons(rows)
        summary = {"schema": "supermix-tether-study-v1", "status": "complete", "config": asdict(config), "rows": rows,
                   "comparisons": comparisons, "decision": _decision(rows, comparisons, config.phase, config.protocol),
                   "provenance": {"version": __version__, "source_manifest": manifest, "dataset_manifest": split["manifest"],
                                  "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                                  "sklearn": sklearn.__version__, "platform": platform.platform(), "device": "cpu",
                                  "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
                                  "evaluation": f"{config.phase} ({config.protocol}): {'validation' if config.phase == 'development' else 'test'} images; within sklearn digits, synthetic tasks/posterior interventions; donors share lineage.",
                                  "budget": "Four receiver views per training step. IIT and Tether receive identical labelled views; extra effect/invariance arithmetic is not FLOP matched."}}
        atomic_json(output_dir / "summary.json", summary)
        payloads.append("summary.json")
        seal_files(output_dir, payloads)
        atomic_json(output_dir / "status.json", {"status": "complete", "rows": len(rows)})
        return summary
    except BaseException as error:
        atomic_json(output_dir / "status.json", {"status": "failed", "error": str(error), "completed_rows": len(rows)})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def _restore(study_dir: Path, row: dict, config: StudyConfig):
    checkpoint = torch.load(safe_artifact(study_dir, row["checkpoint"]), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "supermix-tether-checkpoint-v1" or checkpoint["seed"] != row["seed"] or checkpoint["arm"] != row["arm"]:
        raise ValueError("checkpoint identity mismatch")
    split = load_digits_split(seed=config.data_seed)
    if checkpoint["dataset_manifest"] != split["manifest"]:
        raise ValueError("dataset split drift")
    bank = DonorBank(config=checkpoint["bank_config"])
    bank.load_state_dict(checkpoint["bank_state"])
    bank.eval()
    rel_weight = checkpoint["model_state"].get("reliability.0.weight")
    forensic = config.protocol == "v2" or (rel_weight is not None and rel_weight.shape[-1] == 20)
    with torch.random.fork_rng(devices=[]):
        model = TetherReceiver(forensic=forensic)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        posteriors = bank(split["images"]).softmax(-1)
    return model, split, posteriors, checkpoint


def replay_study(study_dir: Path, strict: bool = False) -> dict:
    study_dir = Path(study_dir).resolve()
    result = verify_files(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "supermix-tether-study-v1" or summary.get("status") != "complete":
        raise ValueError("unsupported or incomplete study")
    config = StudyConfig(**summary["config"])
    conditions = _conditions(config.protocol)
    working_tree_matches = source_manifest() == summary["provenance"]["source_manifest"]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    metric_checks, inference_checks, recovered = 0, 0, []
    try:
        expected_keys = {(s, a) for s in config.seeds for a in config.arms}
        if {(r["seed"], r["arm"]) for r in summary["rows"]} != expected_keys or len(summary["rows"]) != len(expected_keys):
            raise ValueError("study row identity mismatch")
        for row in summary["rows"]:
            if sha256(safe_artifact(study_dir, row["checkpoint"])) != row["checkpoint_sha256"] or sha256(safe_artifact(study_dir, row["trace"])) != row["trace_sha256"]:
                raise ValueError("row artifact hash mismatch")
            loaded = np.load(safe_artifact(study_dir, row["trace"]), allow_pickle=False)
            metrics = {}
            restored = _restore(study_dir, row, config) if strict else None
            for condition in conditions:
                trace = {key: loaded[f"{condition}__{key}"] for key in ("target", "relevant_target", "factual", "relevant", "irrelevant", "damaged", "erased", "shuffled")}
                metrics[condition] = metrics_from_trace(trace)
                for key, actual in metrics[condition].items():
                    if not np.isclose(actual, row["metrics"][condition][key], rtol=0., atol=1e-12):
                        raise ValueError(f"metric mismatch: {row['seed']} {row['arm']} {condition}.{key}")
                    metric_checks += 1
                if strict:
                    model, split, posteriors, checkpoint = restored
                    if checkpoint["source_manifest"] != summary["provenance"]["source_manifest"]:
                        raise ValueError("checkpoint source mismatch")
                    actual_trace = _trace(model, _batch(posteriors, split, config, row["seed"], condition))
                    for view, actual in actual_trace.items():
                        if not np.allclose(actual, trace[view], atol=2e-6, rtol=1e-6):
                            raise ValueError(f"inference mismatch: {condition}.{view}")
                        if view not in ("target", "relevant_target"):
                            inference_checks += 1
            recovered.append({"seed": row["seed"], "arm": row["arm"], "metrics": metrics})
            loaded.close()
        comparisons = paired_comparisons(recovered)
        if comparisons != summary["comparisons"]:
            raise ValueError("paired comparison mismatch")
        if _decision(recovered, comparisons, config.phase, config.protocol) != summary["decision"]:
            raise ValueError("decision mismatch")
        return {**result, "status": "verified", "protocol": config.protocol,
                "metric_checks": metric_checks, "inference_checks": inference_checks,
                "strict": strict, "source_manifest_matches_working_tree": working_tree_matches,
                "meaning": "Saved metric arithmetic and checkpoint inference verified; training and scientific claims require separate validation."}
    finally:
        torch.set_num_threads(previous_threads)


def _mahalanobis_cached(study_dir: Path, seed: int, split: dict, posteriors: torch.Tensor) -> dict:
    key = (str(study_dir.resolve()), int(seed))
    if key in _MAHALANOBIS_CACHE:
        return _MAHALANOBIS_CACHE[key]
    val_post = posteriors[split["validation"]].detach().cpu().numpy()
    det = fit_mahalanobis_forensics(val_post)
    if len(_MAHALANOBIS_CACHE) < 64:
        _MAHALANOBIS_CACHE[key] = det
    return det


def _apply_tamper_tensor(clean_posteriors: torch.Tensor, source: int,
                         tamper_mode: str, corrupted_class: int,
                         confidence: float) -> tuple[torch.Tensor, list[int]]:
    """Construct flat-tail (`corrupt_*`) or dark-knowledge-mimicking (`stealth_*`) specialist corruption."""
    p = clean_posteriors.clone()
    dmg: list[int] = []
    c_val = max(0.5, min(0.999, float(confidence)))
    c_idx = int(corrupted_class) % 10
    if tamper_mode in ("corrupt_1", "corrupt_2", "corrupt_3"):
        cnt = 1 if tamper_mode == "corrupt_1" else (2 if tamper_mode == "corrupt_2" else 3)
        dmg = list(range(cnt))
        other_p = (1.0 - c_val) / 9.0
        c_vec = torch.full((10,), other_p, dtype=torch.float32)
        c_vec[c_idx] = c_val
        for e in dmg:
            p[0, source, e] = c_vec
    elif tamper_mode in ("stealth_1", "stealth_2", "stealth_3"):
        cnt = 1 if tamper_mode == "stealth_1" else (2 if tamper_mode == "stealth_2" else 3)
        dmg = list(range(cnt))
        rem = 1.0 - c_val
        decoy = (c_idx + 5) % 10
        base_other = (rem * 0.12) / 8.0
        c_vec = torch.full((10,), base_other, dtype=torch.float32)
        c_vec[decoy] = rem * 0.88
        c_vec[c_idx] = c_val
        for e in dmg:
            p[0, source, e] = c_vec
    return p, dmg


def _calibrate_cached(study_dir: Path, row: dict, config: StudyConfig,
                      model: TetherReceiver, split: dict, posteriors: torch.Tensor) -> dict:
    key = (str(study_dir.resolve()), row["seed"], row["arm"], row.get("checkpoint_sha256", ""))
    if key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[key]
    val_batch = _batch(posteriors, split, config, row["seed"], "seen_independent", validation=True)
    with torch.no_grad():
        val_detail = model.inspect(val_batch["donors"], val_batch["query"], val_batch["context"])
        val_logits = val_detail["logits"].cpu().numpy()
        val_conflict = val_detail["queried_conflict_jsd"].cpu().numpy()
    val_targets = val_batch["target"].cpu().numpy()
    temp_fit = fit_temperature(val_logits, val_targets)
    t_star = temp_fit["temperature"]
    val_probs = softmax_temperature(val_logits, temperature=t_star)
    val_scores = raps_nonconformity_scores(val_probs, val_targets, k_reg=1, lam_reg=0.02)
    val_conflict_q95 = float(np.quantile(val_conflict, 0.95))
    entry = {
        "temperature": t_star,
        "val_scores": val_scores,
        "val_conflict_q95": val_conflict_q95,
        "pre_scaling": temp_fit["pre_scaling"],
        "post_scaling": temp_fit["post_scaling"],
    }
    if len(_CALIBRATION_CACHE) < 256:
        _CALIBRATION_CACHE[key] = entry
    return entry


def inspect_example(study_dir: Path, index: int = 0, arm: str = "tether",
                    mode: str = "factual", seed: int | None = None,
                    alpha: float = 0.05) -> dict:
    study_dir = Path(study_dir).resolve()
    modes = ("factual", "relevant", "irrelevant", "damaged", "damage_two", "damage_common", "erased", "context_shift")
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    config = StudyConfig(**summary["config"])
    if type(index) is not int or not 0 <= index < config.eval_size or arm not in config.arms or mode not in modes:
        raise ValueError("invalid example index, arm or mode")
    if seed is not None and (type(seed) is not int or seed not in config.seeds):
        raise ValueError("invalid study seed")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    row = next(r for r in summary["rows"] if r["arm"] == arm and (seed is None or r["seed"] == seed))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        model, split, posteriors, _ = _restore(study_dir, row, config)
        cal = _calibrate_cached(study_dir, row, config, model, split, posteriors)
        q_hat = conformal_quantile(cal["val_scores"], alpha=float(alpha))
        batch = _batch(posteriors, split, config, row["seed"], "heldout_independent")
        clean_donors = batch["donors"][index:index + 1].clone()
        clean_context = batch["context"][index:index + 1].clone()
        donors = batch["donors"].clone()
        image_ids = batch["image_ids"].clone()
        target = batch["target"]
        context = batch["context"].clone()
        damaged_experts: list[int] = []
        if mode in ("relevant", "irrelevant", "damaged"):
            donors = batch[mode]
            if mode == "relevant":
                target = batch["relevant_target"]
                image_ids[torch.arange(len(image_ids)), batch["query"][:, 0]] = batch["replacement_ids"]
            elif mode == "irrelevant":
                image_ids[torch.arange(len(image_ids)), batch["irrelevant_source"]] = batch["irrelevant_replacement_ids"]
            elif mode == "damaged":
                damaged_experts = batch["damaged_experts"][index].tolist()
        elif mode == "damage_two":
            d2_batch = _batch(posteriors, split, config, row["seed"], "damage_two")
            donors = d2_batch["donors"]
            damaged_experts = d2_batch["damaged_experts"][index].tolist()
        elif mode == "damage_common":
            d3_batch = _batch(posteriors, split, config, row["seed"], "damage_common")
            donors = d3_batch["donors"]
            damaged_experts = [0, 1, 2]
        elif mode == "erased":
            donors.fill_(.1)
        elif mode == "context_shift":
            context = F.one_hot((batch["target"] + 1) % 10, 10).float()

        q_slice = batch["query"][index:index + 1]
        d_slice = donors[index:index + 1]
        c_slice = context[index:index + 1]
        with torch.no_grad():
            detail = model.inspect(d_slice, q_slice, c_slice)
            logits_np = detail["logits"].cpu().numpy()
            probabilities = detail["logits"].softmax(-1)[0].tolist()
            patching = model.interchange_intervention(
                d_slice, q_slice, c_slice,
                clean_donors, q_slice, clean_context,
                target=target[index:index + 1],
            )
            shapley = model.shapley_interactions(d_slice, q_slice, c_slice)
        attribution = model.attribute(d_slice, q_slice, c_slice)
        ig = model.integrated_gradients(d_slice, q_slice, c_slice, steps=128)
        cal_probs = softmax_temperature(logits_np, temperature=cal["temperature"])
        ep_trust = float(detail["epistemic_trust"][0])
        conflict_val = float(detail["queried_conflict_jsd"][0])
        raps = raps_prediction_sets(
            cal_probs, q_hat, k_reg=1, lam_reg=0.02,
            epistemic_trust=np.array([ep_trust]) if model.forensic else None,
            epistemic_threshold=model.epistemic_threshold,
        )
        nex_raps = nex_raps_prediction_sets(
            logits_np, temperature=cal["temperature"], q_hat=q_hat,
            conflict_jsd=np.array([conflict_val]), cal_conflict_q95=cal["val_conflict_q95"],
            k_reg=1, lam_reg=0.02,
            epistemic_trust=np.array([ep_trust]) if model.forensic else None,
            epistemic_threshold=model.epistemic_threshold,
        )
        m_det = _mahalanobis_cached(study_dir, row["seed"], split, posteriors)
        m_scores = score_mahalanobis_forensics(d_slice.cpu().numpy(), m_det)
        patch_summary = {
            node: {
                "prediction": int(info["prediction"][0]),
                "restoration": float(info["restoration"][0]),
                "prob_shift": float(info["prob_shift"][0]),
                "kl_to_cf": float(info["kl_to_cf"][0]),
            }
            for node, info in patching["patches"].items()
        }
        source, operation = batch["query"][index].tolist()
        return {"index": index, "arm": arm, "mode": mode, "seed": row["seed"], "protocol": config.protocol,
                "images": split["images"][image_ids[index]].tolist(),
                "query": {"source": source, "operation": operation, "label": ["identity", "next digit modulo ten", "mirror (9 - digit)"][operation]},
                "target": int(target[index]), "prediction": int(np.argmax(probabilities)), "probabilities": probabilities,
                "calibrated_probabilities": cal_probs[0].tolist(),
                "temperature": cal["temperature"],
                "conformal_alpha": float(alpha),
                "conformal_q_hat": float(q_hat),
                "conformal_set": raps["prediction_sets"][0],
                "nex_conformal_set": nex_raps["prediction_sets"][0],
                "source_weights": detail["source_weights"][0].tolist(), "expert_weights": detail["expert_weights"][0].tolist(),
                "trust_score": detail["trust_score"][0].tolist(), "tail_entropy": detail["tail_entropy"][0].tolist(),
                "mahalanobis_distance": m_scores["mahalanobis_distance"][0].tolist(),
                "hybrid_trust": m_scores["hybrid_trust"][0].tolist(),
                "expert_margin": detail["expert_margin"][0].tolist(), "damaged_experts": damaged_experts,
                "epistemic_trust": ep_trust,
                "aleatoric_uncertainty": float(detail["queried_aleatoric"][0]),
                "epistemic_conflict_jsd": conflict_val,
                "forensic_spoof_risk": float(detail["forensic_spoof_risk"][0]),
                "abstain": bool(detail["abstain"][0]),
                "selective_prediction": int(detail["selective_prediction"][0]),
                "source_saliency": attribution["source_saliency"][0].tolist(),
                "expert_saliency": attribution["expert_saliency"][0].tolist(),
                "ig_source_saliency": ig["source_saliency"][0].tolist(),
                "ig_expert_saliency": ig["expert_saliency"][0].tolist(),
                "ig_completeness_error": float(ig["completeness_error"][0]),
                "shapley_values": shapley["shapley_values"][0].tolist(),
                "shapley_interaction_matrix": shapley["interaction_matrix"][0].tolist(),
                "shapley_efficiency_error": float(shapley["efficiency_error"][0]),
                "coalition_payoffs": {k: round(float(v[0]), 4) for k, v in shapley["coalition_payoffs"].items()},
                "interchange_patching": patch_summary,
                "context": context[index].tolist(), "note": "Learned weights are routing summaries, not causal explanations. Compare controlled interventions."}
    finally:
        torch.set_num_threads(previous_threads)


def live_predict(study_dir: Path, pixels: list[float] | None = None,
                 digit: int = 3, source: int = 0, operation: int = 0,
                 tamper_mode: str = "none", corrupted_class: int = 4,
                 confidence: float = 0.991, arm: str = "tether",
                 seed: int | None = None, alpha: float = 0.05,
                 defense: bool = False) -> dict:
    """Run live interactive inference and tamper simulation on real/custom digit inputs."""
    study_dir = Path(study_dir).resolve()
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    config = StudyConfig(**summary["config"])
    row = next(r for r in summary["rows"] if r["arm"] == arm and (seed is None or r["seed"] == seed))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        model, split, all_posteriors, ckpt = _restore(study_dir, row, config)
        cal = _calibrate_cached(study_dir, row, config, model, split, all_posteriors)
        q_hat = conformal_quantile(cal["val_scores"], alpha=float(alpha))
        m_det = _mahalanobis_cached(study_dir, row["seed"], split, all_posteriors)
        bank = DonorBank(config=ckpt["bank_config"])
        bank.load_state_dict(ckpt["bank_state"])
        bank.eval()

        if pixels is not None and len(pixels) == 64:
            queried_image = torch.tensor(pixels, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            cand = split["test"][split["labels"][split["test"]] == digit]
            chosen_id = cand[0] if len(cand) > 0 else split["test"][0]
            queried_image = split["images"][chosen_id]

        bg_images = [split["images"][split["test"][(i + 1) * 10]] for i in range(2)]
        all_images = [None] * 3
        all_images[source] = queried_image
        bg_idx = 0
        for i in range(3):
            if all_images[i] is None:
                all_images[i] = bg_images[bg_idx]
                bg_idx += 1
        stacked_images = torch.stack(all_images)

        with torch.no_grad():
            specialist_logits = bank(stacked_images)
            clean_posteriors = specialist_logits.softmax(-1).unsqueeze(0)

        raw_posteriors, damaged_experts = _apply_tamper_tensor(
            clean_posteriors, source, tamper_mode, corrupted_class, confidence
        )
        sanitized_info = sanitize_specialist_posteriors(raw_posteriors.cpu().numpy(), m_det)
        sanitized_posteriors = torch.tensor(sanitized_info["sanitized_posteriors"], dtype=torch.float32)
        active_posteriors = sanitized_posteriors if defense else raw_posteriors

        query_t = torch.tensor([[source, operation]], dtype=torch.long)
        context_t = torch.full((1, 10), 0.1, dtype=torch.float32)
        with torch.no_grad():
            detail = model.inspect(active_posteriors, query_t, context_t)
            sanitized_detail = model.inspect(sanitized_posteriors, query_t, context_t)
            clean_detail = model.inspect(clean_posteriors, query_t, context_t)
            logits_np = detail["logits"].cpu().numpy()
            probabilities = detail["logits"].softmax(-1)[0].tolist()
            clean_target = clean_detail["logits"].argmax(-1)
            patching = model.interchange_intervention(
                active_posteriors, query_t, context_t,
                clean_posteriors, query_t, context_t,
                target=clean_target,
            )
            shapley = model.shapley_interactions(active_posteriors, query_t, context_t, target=clean_target)

        attribution = model.attribute(active_posteriors, query_t, context_t)
        ig = model.integrated_gradients(active_posteriors, query_t, context_t, steps=128)
        cal_probs = softmax_temperature(logits_np, temperature=cal["temperature"])
        ep_trust = float(detail["epistemic_trust"][0])
        hybrid_queried_trust = float(sanitized_info["hybrid_trust"][0, source].max())
        effective_trust = hybrid_queried_trust if defense else ep_trust
        abstain_flag = (effective_trust < model.epistemic_threshold) if (model.forensic or defense) else False
        pred_class = int(np.argmax(probabilities))
        sel_pred = -1 if abstain_flag else pred_class
        conflict_val = float(detail["queried_conflict_jsd"][0])

        raps = raps_prediction_sets(
            cal_probs, q_hat, k_reg=1, lam_reg=0.02,
            epistemic_trust=np.array([effective_trust]) if (model.forensic or defense) else None,
            epistemic_threshold=model.epistemic_threshold,
        )
        nex_raps = nex_raps_prediction_sets(
            logits_np, temperature=cal["temperature"], q_hat=q_hat,
            conflict_jsd=np.array([conflict_val]), cal_conflict_q95=cal["val_conflict_q95"],
            k_reg=1, lam_reg=0.02,
            epistemic_trust=np.array([effective_trust]) if (model.forensic or defense) else None,
            epistemic_threshold=model.epistemic_threshold,
        )
        patch_summary = {
            node: {
                "prediction": int(info["prediction"][0]),
                "restoration": float(info["restoration"][0]),
                "prob_shift": float(info["prob_shift"][0]),
                "kl_to_cf": float(info["kl_to_cf"][0]),
            }
            for node, info in patching["patches"].items()
        }
        operation_names = ["identity", "next digit modulo ten", "mirror (9 - digit)"]
        san_abstain = bool(hybrid_queried_trust < model.epistemic_threshold)
        san_pred = int(sanitized_detail["logits"].argmax(-1)[0])

        return {
            "source": source,
            "operation": operation,
            "operation_label": operation_names[operation],
            "tamper_mode": tamper_mode,
            "defense": bool(defense),
            "damaged_experts": damaged_experts,
            "prediction": pred_class,
            "clean_prediction": int(clean_target[0]),
            "sanitized_prediction": -1 if san_abstain else san_pred,
            "sanitized_abstain": san_abstain,
            "probabilities": probabilities,
            "calibrated_probabilities": cal_probs[0].tolist(),
            "temperature": cal["temperature"],
            "conformal_alpha": float(alpha),
            "conformal_q_hat": float(q_hat),
            "conformal_set": raps["prediction_sets"][0],
            "nex_conformal_set": nex_raps["prediction_sets"][0],
            "source_weights": detail["source_weights"][0].tolist(),
            "expert_weights": detail["expert_weights"][0].tolist(),
            "trust_score": detail["trust_score"][0].tolist(),
            "tail_entropy": detail["tail_entropy"][0].tolist(),
            "mahalanobis_distance": sanitized_info["mahalanobis_distance"][0, source].tolist(),
            "hybrid_trust": sanitized_info["hybrid_trust"][0, source].tolist(),
            "flagged_specialists": sanitized_info["flagged_specialists"][0, source].tolist(),
            "epistemic_trust": ep_trust,
            "hybrid_epistemic_trust": hybrid_queried_trust,
            "aleatoric_uncertainty": float(detail["queried_aleatoric"][0]),
            "epistemic_conflict_jsd": conflict_val,
            "forensic_spoof_risk": float(1.0 - effective_trust),
            "abstain": bool(abstain_flag),
            "selective_prediction": int(sel_pred),
            "source_saliency": attribution["source_saliency"][0].tolist(),
            "expert_saliency": attribution["expert_saliency"][0].tolist(),
            "ig_source_saliency": ig["source_saliency"][0].tolist(),
            "ig_expert_saliency": ig["expert_saliency"][0].tolist(),
            "ig_completeness_error": float(ig["completeness_error"][0]),
            "shapley_values": shapley["shapley_values"][0].tolist(),
            "shapley_interaction_matrix": shapley["interaction_matrix"][0].tolist(),
            "shapley_efficiency_error": float(shapley["efficiency_error"][0]),
            "coalition_payoffs": {k: round(float(v[0]), 4) for k, v in shapley["coalition_payoffs"].items()},
            "specialist_predictions": raw_posteriors[0, source].argmax(-1).tolist(),
            "specialist_confidences": raw_posteriors[0, source].max(-1).values.tolist(),
            "interchange_patching": patch_summary,
            "images": [img.tolist() for img in all_images],
        }
    finally:
        torch.set_num_threads(previous_threads)


def live_arena(study_dir: Path, pixels: list[float] | None = None,
               digit: int = 3, source: int = 0, operation: int = 0,
               tamper_mode: str = "clean", corrupted_class: int = 4,
               confidence: float = 0.991, seed: int | None = None,
               alpha: float = 0.05, defense: bool = False) -> dict:
    """Compare all 7 receiver arms side-by-side and run a live adversarial phase-transition sweep."""
    study_dir = Path(study_dir).resolve()
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    config = StudyConfig(**summary["config"])
    chosen_seed = seed if seed is not None else config.seeds[0]
    if chosen_seed not in config.seeds:
        raise ValueError("invalid study seed")
    if tamper_mode == "none":
        tamper_mode = "clean"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        seed_rows = {r["arm"]: r for r in summary["rows"] if r["seed"] == chosen_seed}
        first_row = next(iter(seed_rows.values()))
        _, split, all_posteriors, first_ckpt = _restore(study_dir, first_row, config)
        m_det = _mahalanobis_cached(study_dir, chosen_seed, split, all_posteriors)
        bank = DonorBank(config=first_ckpt["bank_config"])
        bank.load_state_dict(first_ckpt["bank_state"])
        bank.eval()

        if pixels is not None and len(pixels) == 64:
            queried_image = torch.tensor(pixels, dtype=torch.float32).clamp(0.0, 1.0)
        else:
            cand = split["test"][split["labels"][split["test"]] == digit]
            chosen_id = cand[0] if len(cand) > 0 else split["test"][0]
            queried_image = split["images"][chosen_id]

        bg_images = [split["images"][split["test"][(i + 1) * 10]] for i in range(2)]
        all_images = [None] * 3
        all_images[source] = queried_image
        bg_idx = 0
        for i in range(3):
            if all_images[i] is None:
                all_images[i] = bg_images[bg_idx]
                bg_idx += 1
        stacked_images = torch.stack(all_images)

        with torch.no_grad():
            specialist_logits = bank(stacked_images)
            clean_posteriors = specialist_logits.softmax(-1).unsqueeze(0)

        def _make_tampered(tm: str, conf_val: float) -> tuple[torch.Tensor, list[int]]:
            raw_p, dmg = _apply_tamper_tensor(clean_posteriors, source, tm, corrupted_class, conf_val)
            if defense:
                san = sanitize_specialist_posteriors(raw_p.cpu().numpy(), m_det)
                return torch.tensor(san["sanitized_posteriors"], dtype=torch.float32), dmg
            return raw_p, dmg

        raw_posteriors, damaged_experts = _apply_tamper_tensor(
            clean_posteriors, source, tamper_mode, corrupted_class, confidence
        )
        san_info = sanitize_specialist_posteriors(raw_posteriors.cpu().numpy(), m_det)
        posteriors = torch.tensor(san_info["sanitized_posteriors"], dtype=torch.float32) if defense else raw_posteriors
        intact_experts = [e for e in range(3) if e not in damaged_experts]
        query_t = torch.tensor([[source, operation]], dtype=torch.long)
        context_t = torch.full((1, 10), 0.1, dtype=torch.float32)
        hybrid_queried_trust = float(san_info["hybrid_trust"][0, source].max())

        # Infer baseline digit from clean specialist consensus + task transform
        clean_digit = int(clean_posteriors[0, source].mean(0).argmax().item())
        expected_target = [clean_digit, (clean_digit + 1) % 10, 9 - clean_digit][operation]

        arm_results = []
        models_by_arm = {}
        for arm_name in config.arms:
            if arm_name not in seed_rows:
                continue
            row = seed_rows[arm_name]
            model, _, _, _ = _restore(study_dir, row, config)
            models_by_arm[arm_name] = model
            cal = _calibrate_cached(study_dir, row, config, model, split, all_posteriors)
            q_hat = conformal_quantile(cal["val_scores"], alpha=float(alpha))
            with torch.no_grad():
                detail = model.inspect(posteriors, query_t, context_t)
                clean_detail = model.inspect(clean_posteriors, query_t, context_t)
                logits_np = detail["logits"].cpu().numpy()
                probs = detail["logits"].softmax(-1)[0].tolist()
                shapley = model.shapley_interactions(posteriors, query_t, context_t, target=torch.tensor([expected_target]))
            cal_probs = softmax_temperature(logits_np, temperature=cal["temperature"])
            ep_trust = hybrid_queried_trust if defense else float(detail["epistemic_trust"][0])
            conflict_val = float(detail["queried_conflict_jsd"][0])
            raps = raps_prediction_sets(
                cal_probs, q_hat, k_reg=1, lam_reg=0.02,
                epistemic_trust=np.array([ep_trust]) if (model.forensic or defense) else None,
                epistemic_threshold=model.epistemic_threshold,
            )
            nex_raps = nex_raps_prediction_sets(
                logits_np, temperature=cal["temperature"], q_hat=q_hat,
                conflict_jsd=np.array([conflict_val]), cal_conflict_q95=cal["val_conflict_q95"],
                k_reg=1, lam_reg=0.02,
                epistemic_trust=np.array([ep_trust]) if (model.forensic or defense) else None,
                epistemic_threshold=model.epistemic_threshold,
            )
            ew = detail["expert_weights"][0, source].tolist()
            intact_share = sum(ew[e] for e in intact_experts) if intact_experts else 0.0
            pred = int(np.argmax(probs))
            abstain = (ep_trust < model.epistemic_threshold) if (model.forensic or defense) else bool(detail["abstain"][0])
            sel_pred = -1 if abstain else pred
            if tamper_mode in ("corrupt_3", "stealth_3"):
                survived = abstain
            else:
                survived = (not abstain) and (pred == expected_target)
            arm_results.append({
                "arm": arm_name,
                "prediction": pred,
                "clean_prediction": int(clean_detail["logits"].argmax(-1)[0]),
                "selective_prediction": sel_pred,
                "abstain": bool(abstain),
                "confidence": float(max(probs)),
                "target_probability": float(probs[expected_target]),
                "probabilities": probs,
                "conformal_set": raps["prediction_sets"][0],
                "nex_conformal_set": nex_raps["prediction_sets"][0],
                "conformal_q_hat": float(q_hat),
                "temperature": cal["temperature"],
                "queried_source_weight": float(detail["source_weights"][0, source]),
                "expert_weights": ew,
                "intact_weight_share": float(intact_share),
                "epistemic_trust": ep_trust,
                "shapley_values": [round(float(x), 4) for x in shapley["shapley_values"][0].tolist()],
                "survived": bool(survived),
            })

        # Build live adversarial confidence & regime sweep for key arms
        sweep_arms = [a for a in ("tether", "iit", "unpaired", "passive") if a in models_by_arm]
        conf_grid = [0.55, 0.70, 0.85, 0.92, 0.96, 0.991]
        active_sweep_mode = (
            tamper_mode
            if tamper_mode in ("corrupt_1", "corrupt_2", "corrupt_3", "stealth_1", "stealth_2", "stealth_3")
            else "corrupt_2"
        )
        sweep_tensors = torch.cat([_make_tampered(active_sweep_mode, c)[0] for c in conf_grid], dim=0)
        q_sweep = query_t.expand(len(conf_grid), -1)
        c_sweep = context_t.expand(len(conf_grid), -1)
        sweep_curves = {}
        with torch.no_grad():
            for arm_name in sweep_arms:
                m = models_by_arm[arm_name]
                det = m.inspect(sweep_tensors, q_sweep, c_sweep)
                pr = det["logits"].softmax(-1)
                preds = pr.argmax(-1).tolist()
                target_probs = pr[:, expected_target].tolist()
                sweep_curves[arm_name] = {
                    "target_probabilities": [round(float(x), 4) for x in target_probs],
                    "predictions": [int(x) for x in preds],
                    "epistemic_trust": [round(float(x), 4) for x in det["epistemic_trust"].tolist()],
                }

        return {
            "seed": chosen_seed,
            "source": source,
            "operation": operation,
            "tamper_mode": tamper_mode,
            "defense": bool(defense),
            "corrupted_class": corrupted_class,
            "confidence": confidence,
            "conformal_alpha": float(alpha),
            "clean_digit": clean_digit,
            "expected_target": expected_target,
            "damaged_experts": damaged_experts,
            "mahalanobis_distance": [round(float(x), 3) for x in san_info["mahalanobis_distance"][0, source].tolist()],
            "hybrid_trust": [round(float(x), 4) for x in san_info["hybrid_trust"][0, source].tolist()],
            "arms": arm_results,
            "sweep": {
                "mode": active_sweep_mode,
                "confidences": conf_grid,
                "curves": sweep_curves,
            },
        }
    finally:
        torch.set_num_threads(previous_threads)


def calibrate_study(study_dir: Path, alpha: float = 0.05, seed: int | None = None) -> dict:
    """Evaluate Split Conformal RAPS sets, temperature scaling, and ECE across a saved study."""
    study_dir = Path(study_dir).resolve()
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "supermix-tether-study-v1" or summary.get("status") != "complete":
        raise ValueError("unsupported or incomplete study")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    config = StudyConfig(**summary["config"])
    if seed is not None and seed not in config.seeds:
        raise ValueError("invalid study seed")
    conditions = _conditions(config.protocol)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        rows_out = []
        for row in summary["rows"]:
            if seed is not None and row["seed"] != seed:
                continue
            model, split, posteriors, _ = _restore(study_dir, row, config)
            cal = _calibrate_cached(study_dir, row, config, model, split, posteriors)
            q_hat = conformal_quantile(cal["val_scores"], alpha=float(alpha))
            loaded = np.load(safe_artifact(study_dir, row["trace"]), allow_pickle=False)
            cond_metrics = {}
            for cond in conditions:
                logits = loaded[f"{cond}__factual"]
                targets = loaded[f"{cond}__target"]
                pre_cal = calibration_metrics(logits, targets, temperature=1.0)
                post_cal = calibration_metrics(logits, targets, temperature=cal["temperature"])
                cal_probs = softmax_temperature(logits, temperature=cal["temperature"])
                # For common-mode or forensic receiver, evaluate epistemic trust on the batch
                if model.forensic and cond in ("damage_common", "damage_two", "damage_one", "heldout_independent"):
                    batch = _batch(posteriors, split, config, row["seed"], cond)
                    with torch.no_grad():
                        insp = model.inspect(batch["donors"], batch["query"], batch["context"])
                        ep_trust = insp["epistemic_trust"].cpu().numpy()
                        conflict_jsd = insp["queried_conflict_jsd"].cpu().numpy()
                else:
                    ep_trust = None
                    conflict_jsd = np.zeros(len(targets), dtype=np.float64)
                conf_eval = evaluate_conformal_sets(
                    cal_probs, targets, q_hat, k_reg=1, lam_reg=0.02,
                    epistemic_trust=ep_trust, epistemic_threshold=model.epistemic_threshold,
                )
                nex_eval = evaluate_nex_conformal_sets(
                    logits, targets, temperature=cal["temperature"], q_hat=q_hat,
                    conflict_jsd=conflict_jsd, cal_conflict_q95=cal["val_conflict_q95"],
                    k_reg=1, lam_reg=0.02,
                    epistemic_trust=ep_trust, epistemic_threshold=model.epistemic_threshold,
                )
                aci_eval = adaptive_conformal_inference(
                    cal["val_scores"], cal_probs, targets, alpha=float(alpha),
                    gamma=0.05, k_reg=1, lam_reg=0.02,
                    epistemic_trust=ep_trust, epistemic_threshold=model.epistemic_threshold,
                )
                cond_metrics[cond] = {
                    "ece_pre": pre_cal["ece"],
                    "ece_post": post_cal["ece"],
                    "brier_post": post_cal["brier"],
                    "nll_post": post_cal["nll"],
                    **conf_eval,
                    "nex_marginal_coverage": nex_eval["marginal_coverage"],
                    "nex_mean_set_size": nex_eval["mean_set_size"],
                    "aci_marginal_coverage": aci_eval["marginal_coverage"],
                    "aci_mean_set_size": aci_eval["mean_set_size"],
                }
            loaded.close()
            rows_out.append({
                "seed": row["seed"],
                "arm": row["arm"],
                "temperature": cal["temperature"],
                "q_hat": float(q_hat),
                "conditions": cond_metrics,
            })

        # Aggregate means per arm across evaluated seeds
        by_arm = {}
        for arm_name in config.arms:
            arm_rows = [r for r in rows_out if r["arm"] == arm_name]
            if not arm_rows:
                continue
            agg_conds = {}
            for cond in conditions:
                keys = arm_rows[0]["conditions"][cond].keys()
                agg_conds[cond] = {
                    k: round(float(np.mean([r["conditions"][cond][k] for r in arm_rows])), 5)
                    for k in keys
                }
            by_arm[arm_name] = {
                "mean_temperature": round(float(np.mean([r["temperature"] for r in arm_rows])), 5),
                "mean_q_hat": round(float(np.mean([r["q_hat"] for r in arm_rows])), 5),
                "seeds": [r["seed"] for r in arm_rows],
                "conditions": agg_conds,
            }

        return {
            "schema": "supermix-tether-conformal-v1",
            "protocol": config.protocol,
            "alpha": float(alpha),
            "target_coverage": round(1.0 - float(alpha), 4),
            "summary_by_arm": by_arm,
        }
    finally:
        torch.set_num_threads(previous_threads)
