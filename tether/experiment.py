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
from .data import DonorBank, fit_donors, load_digits_split, make_batch
from .model import TetherReceiver, objective
from .receipts import atomic_json, metrics_from_trace, paired_comparisons, safe_artifact, seal_files, sha256, verify_files


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
    pairs, cue, damage, damaged = CONDITIONS[condition]
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TetherReceiver()
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
        loss, details = objective(model, batch, arm, effect_weight=config.effect_weight, invariance_weight=config.invariance_weight)
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
                   "training_steps": config.steps, "selection": "minimum CE on seen-combination independent-cue validation images only"}


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


def _decision(rows, comparisons, phase):
    by_metric = {(r["baseline"], r["metric"]): r for r in comparisons}
    primary = by_metric.get(("iit", "heldout_independent.accuracy"))
    clean = by_metric.get(("iit", "seen_correlated.accuracy"))
    mechanistic = by_metric.get(("unpaired", "heldout_independent.accuracy"))
    tether_rows = [r for r in rows if r["arm"] == "tether"]
    checks = {
        "primary_interval_above_zero": bool(primary and primary["interval"][0] > 0),
        "pairing_beats_unpaired_control": bool(mechanistic and mechanistic["interval"][0] > 0),
        "clean_mean_noninferior_at_two_points": bool(clean and clean["mean_difference"] >= -.02),
        "irrelevant_consistency_at_least_95_percent": bool(tether_rows and np.mean([r["metrics"]["heldout_independent"]["irrelevant_consistency"] for r in tether_rows]) >= .95),
        "positive_donor_utility_above_ten_points": bool(tether_rows and np.mean([r["metrics"]["heldout_independent"]["donor_erasure_gap"] for r in tether_rows]) > .10),
    }
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
                for condition in CONDITIONS:
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
                   "comparisons": comparisons, "decision": _decision(rows, comparisons, config.phase),
                   "provenance": {"version": __version__, "source_manifest": manifest, "dataset_manifest": split["manifest"],
                                  "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                                  "sklearn": sklearn.__version__, "platform": platform.platform(), "device": "cpu",
                                  "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
                                  "evaluation": f"{config.phase}: {'validation' if config.phase == 'development' else 'test'} images; within sklearn digits, synthetic tasks/posterior interventions; donors share lineage.",
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
    with torch.random.fork_rng(devices=[]):
        model = TetherReceiver()
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
    if strict and source_manifest() != summary["provenance"]["source_manifest"]:
        raise ValueError("source fingerprint changed; use the recorded source revision for strict replay")
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
            for condition in CONDITIONS:
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
        if _decision(recovered, comparisons, config.phase) != summary["decision"]:
            raise ValueError("decision mismatch")
        return {**result, "status": "verified", "metric_checks": metric_checks, "inference_checks": inference_checks,
                "strict": strict, "meaning": "Saved metric arithmetic and checkpoint inference verified; training and scientific claims require separate validation."}
    finally:
        torch.set_num_threads(previous_threads)


def inspect_example(study_dir: Path, index: int = 0, arm: str = "tether", mode: str = "factual") -> dict:
    study_dir = Path(study_dir).resolve()
    modes = ("factual", "relevant", "irrelevant", "damaged", "erased", "context_shift")
    summary = json.loads((study_dir / "summary.json").read_text(encoding="utf-8"))
    config = StudyConfig(**summary["config"])
    if source_manifest() != summary["provenance"]["source_manifest"]:
        raise ValueError("source changed since this study; restore its recorded revision before inspecting saved tasks")
    if type(index) is not int or not 0 <= index < config.eval_size or arm not in config.arms or mode not in modes:
        raise ValueError("invalid example index, arm or mode")
    row = next(r for r in summary["rows"] if r["arm"] == arm)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        model, split, posteriors, _ = _restore(study_dir, row, config)
        batch = _batch(posteriors, split, config, row["seed"], "heldout_independent")
        donors = batch["donors"].clone()
        image_ids = batch["image_ids"].clone()
        target = batch["target"]
        context = batch["context"].clone()
        if mode in ("relevant", "irrelevant", "damaged"):
            donors = batch[mode]
            if mode == "relevant":
                target = batch["relevant_target"]
                image_ids[torch.arange(len(image_ids)), batch["query"][:, 0]] = batch["replacement_ids"]
            elif mode == "irrelevant":
                image_ids[torch.arange(len(image_ids)), batch["irrelevant_source"]] = batch["irrelevant_replacement_ids"]
        elif mode == "erased":
            donors.fill_(.1)
        elif mode == "context_shift":
            context = F.one_hot((batch["target"] + 1) % 10, 10).float()
        with torch.no_grad():
            detail = model.inspect(donors[index:index + 1], batch["query"][index:index + 1], context[index:index + 1])
            probabilities = detail["logits"].softmax(-1)[0].tolist()
        source, operation = batch["query"][index].tolist()
        return {"index": index, "arm": arm, "mode": mode, "seed": row["seed"],
                "images": split["images"][image_ids[index]].tolist(),
                "query": {"source": source, "operation": operation, "label": ["identity", "next digit modulo ten", "mirror (9 - digit)"][operation]},
                "target": int(target[index]), "prediction": int(np.argmax(probabilities)), "probabilities": probabilities,
                "source_weights": detail["source_weights"][0].tolist(), "expert_weights": detail["expert_weights"][0].tolist(),
                "context": context[index].tolist(), "note": "Learned weights are routing summaries, not causal explanations. Compare controlled interventions."}
    finally:
        torch.set_num_threads(previous_threads)
