"""Real digit data, frozen visual specialists and reproducible task interventions.

Image identities are split before tasks are sampled. Donors share the same data
lineage; their heterogeneous architectures do not make their errors independent.
Corruptions use a cyclic shift of a prediction, never the task's true label.
Consequently a corruption can accidentally correct an already wrong donor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn import functional as F


HELDOUT_PAIRS = ((0, 2), (1, 1), (2, 0))
SEEN_PAIRS = tuple((source, operation) for source in range(3) for operation in range(3)
                   if (source, operation) not in HELDOUT_PAIRS)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _seed(seed: Any) -> int:
    return _integer(seed, "seed", 0, 2**32 - 1)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _long_vector(value: Any, name: str, *, upper: int | None = None) -> Tensor:
    if isinstance(value, Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense CPU tensor")
        if value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{name} must contain integers")
        result = value.to(dtype=torch.long)
    else:
        array = np.asarray(value)
        if array.dtype.kind not in "iu" or array.dtype.kind == "b":
            raise TypeError(f"{name} must contain integers")
        result = torch.as_tensor(array.astype(np.int64, copy=False))
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if (result < 0).any() or (upper is not None and (result >= upper).any()):
        raise ValueError(f"{name} contains out-of-range values")
    return result


def load_digits_split(seed: int = 1729) -> dict[str, Any]:
    """Load sklearn digits and stratify image IDs into approximately 60/20/20.

    Dividing original integer pixels by 16 requires no fit on held-out images.
    This is a within-dataset image split, not the UCI writer-disjoint protocol.
    """
    seed = _seed(seed)
    digits = load_digits()
    labels = np.asarray(digits.target, dtype=np.int64)
    ids = np.arange(len(labels), dtype=np.int64)
    train, remaining = train_test_split(ids, test_size=0.4, random_state=seed, stratify=labels)
    validation, test = train_test_split(remaining, test_size=0.5, random_state=seed,
                                      stratify=labels[remaining])
    images = np.asarray(digits.data, dtype=np.float32) / np.float32(16)
    partitions = {name: np.sort(values) for name, values in
                  (("train", train), ("validation", validation), ("test", test))}
    dataset_hash = hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": "sklearn.datasets.load_digits",
        "images": len(labels),
        "features": 64,
        "classes": 10,
        "normalization": "original pixel / 16; no fitted statistics",
        "split_seed": seed,
        "split_method": "stratified 60/20/20 image identities before task construction",
        "writer_disjoint": False,
        "dataset_sha256": dataset_hash,
        "images_sha256": _digest(images),
        "labels_sha256": _digest(labels),
        "split_ids": {name: values.tolist() for name, values in partitions.items()},
        "split_sha256": {name: _digest(values) for name, values in partitions.items()},
        "heldout_pairs": [list(pair) for pair in HELDOUT_PAIRS],
    }
    return {"images": torch.from_numpy(images), "labels": torch.from_numpy(labels),
            **{name: torch.from_numpy(values) for name, values in partitions.items()},
            "manifest": manifest}


def task_targets(digits: Tensor, operations: Tensor) -> Tensor:
    """Apply identity, successor modulo ten, or 9-minus-digit to literal labels."""
    digits = _long_vector(digits, "digits", upper=10)
    operations = _long_vector(operations, "operations", upper=3)
    if digits.shape != operations.shape:
        raise ValueError("digits and operations must have the same shape")
    return torch.where(operations == 0, digits,
                       torch.where(operations == 1, (digits + 1) % 10, 9 - digits))


def make_batch(posteriors: Tensor, labels: Tensor, indices: Tensor | Sequence[int],
               n: int, seed: int, split: str = "train", cue: str = "correlated",
               pairs: str = "seen", damage_count: int = 1) -> dict[str, Any]:
    """Sample aligned factual, targeted, irrelevant and corrupted views.

    Every source/replacement image stays inside the supplied identity pool.
    Context is generated once from the original target and kept for all views.
    Damage selects experts and fabricated classes without looking at labels.
    For common-mode damage all three experts receive the same fabricated class.
    """
    n = _integer(n, "n", 1, 100_000)
    seed = _seed(seed)
    damage_count = _integer(damage_count, "damage_count", 1, 3)
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    if cue not in ("correlated", "independent", "reversed"):
        raise ValueError("cue must be correlated, independent or reversed")
    if pairs not in ("seen", "heldout", "all"):
        raise ValueError("pairs must be seen, heldout or all")
    labels = _long_vector(labels, "labels", upper=10)
    indices = _long_vector(indices, "indices", upper=len(labels))
    if indices.unique().numel() != indices.numel():
        raise ValueError("indices must contain distinct image identities")
    if not isinstance(posteriors, Tensor) or posteriors.shape != (len(labels), 3, 10):
        raise ValueError("posteriors must have shape [images,3,10]")
    if posteriors.device.type != "cpu" or posteriors.layout != torch.strided or not posteriors.is_floating_point():
        raise ValueError("posteriors must be a dense floating CPU tensor")
    if (not torch.isfinite(posteriors).all() or (posteriors < 0).any()
            or not torch.allclose(posteriors.sum(-1), torch.ones_like(posteriors.sum(-1)), atol=1e-5, rtol=1e-5)):
        raise ValueError("posteriors must contain finite probability distributions")
    if labels[indices].unique().numel() < 2:
        raise ValueError("the identity pool needs at least two digit classes for relevant swaps")

    random = torch.Generator(device="cpu").manual_seed(seed)
    image_ids = indices[torch.randint(len(indices), (n, 3), generator=random)]
    allowed = SEEN_PAIRS if pairs == "seen" else HELDOUT_PAIRS if pairs == "heldout" else tuple(
        (source, operation) for source in range(3) for operation in range(3))
    query = torch.tensor(allowed, dtype=torch.long)[torch.randint(len(allowed), (n,), generator=random)]
    row_ids = torch.arange(n)
    sources, operations = query.unbind(-1)
    donors = posteriors.detach()[image_ids].to(torch.float32).clone()

    # Use a separate deterministic stream so neither label-derived replacement
    # candidates nor cue construction can affect expert/class corruption choices.
    damage_random = torch.Generator(device="cpu").manual_seed(seed ^ 0x5A17D04B)
    damaged_experts = torch.rand((n, 3), generator=damage_random).argsort(-1)[:, :damage_count]
    damaged = donors.clone()
    selected = donors[row_ids, sources]
    if damage_count == 3:
        fabricated = ((selected[:, 0].argmax(-1) + 1) % 10)[:, None].expand(-1, 3)
    else:
        fabricated = (selected.gather(1, damaged_experts[:, :, None].expand(-1, -1, 10)).argmax(-1) + 1) % 10
    corrupted = torch.full((n, damage_count, 10), 0.001, dtype=torch.float32)
    corrupted.scatter_(-1, fabricated[:, :, None], 0.991)
    damaged[row_ids[:, None], sources[:, None], damaged_experts] = corrupted

    irrelevant_sources = (sources + torch.randint(1, 3, (n,), generator=random)) % 3
    original_digits = labels[image_ids[row_ids, sources]]
    # Cache eligible pools by digit; all are restricted to the declared split.
    alternatives = {int(digit): indices[labels[indices] != digit].tolist() for digit in labels[indices].unique()}
    alternative_lengths = {digit: len(pool) for digit, pool in alternatives.items()}
    original_list = original_digits.tolist()
    nuisance_list = labels[image_ids[row_ids, irrelevant_sources]].tolist()
    replacement_list = [0] * n
    irrelevant_list = [0] * n
    for row in range(n):
        digit = original_list[row]
        replacement_list[row] = alternatives[digit][int(torch.randint(alternative_lengths[digit], (), generator=random))]
        nuisance_digit = nuisance_list[row]
        irrelevant_list[row] = alternatives[nuisance_digit][int(torch.randint(alternative_lengths[nuisance_digit], (), generator=random))]
    replacement_ids = torch.tensor(replacement_list, dtype=torch.long)
    irrelevant_replacement_ids = torch.tensor(irrelevant_list, dtype=torch.long)
    relevant = donors.clone()
    relevant[row_ids, sources] = posteriors.detach()[replacement_ids].to(torch.float32)
    irrelevant = donors.clone()
    irrelevant[row_ids, irrelevant_sources] = posteriors.detach()[irrelevant_replacement_ids].to(torch.float32)
    target = task_targets(original_digits, operations)
    relevant_target = task_targets(labels[replacement_ids], operations)
    if cue == "independent":
        proxy = torch.randint(10, (n,), generator=random)
    else:
        wrong = (target + torch.randint(1, 10, (n,), generator=random)) % 10
        proxy = wrong if cue == "reversed" else torch.where(
            torch.rand(n, generator=random) < 0.97, target, wrong)
    context = F.one_hot(proxy, num_classes=10).to(torch.float32)
    return {
        "donors": donors, "relevant": relevant, "irrelevant": irrelevant, "damaged": damaged,
        "query": query, "context": context, "target": target, "relevant_target": relevant_target,
        "image_ids": image_ids, "replacement_ids": replacement_ids,
        "irrelevant_source": irrelevant_sources, "irrelevant_replacement_ids": irrelevant_replacement_ids,
        "damaged_experts": damaged_experts,
        "manifest": {
            "schema_version": 1, "seed": seed, "split": split, "n": n, "cue": cue,
            "pairs": pairs, "allowed_pairs": [list(pair) for pair in allowed],
            "damage_count": damage_count, "indices_sha256": _digest(indices.numpy()),
            "context_correlated_probability": 0.97 if cue == "correlated" else None,
            "context_retained_across_views": True,
            "corruption": "prediction argmax plus one modulo ten; no true label used",
            "common_mode": damage_count == 3,
            "corruption_can_accidentally_correct_donor_error": True,
        },
    }


class DonorBank(nn.Module):
    """Three reconstructable image classifiers; no arbitrary pickle objects needed."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, schema_version: int = 1,
                 input_dim: int = 64, num_classes: int = 10,
                 hidden_sizes: Sequence[int] = (32, 48, 64),
                 activations: Sequence[str] = ("relu", "tanh", "gelu"),
                 initialization_seed: int = 0) -> None:
        super().__init__()
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError("config must be a mapping")
            known = {"schema_version", "input_dim", "num_classes", "hidden_sizes", "activations", "initialization_seed"}
            if set(config) != known:
                raise ValueError("unrecognized or incomplete donor config")
            schema_version, input_dim, num_classes = config["schema_version"], config["input_dim"], config["num_classes"]
            hidden_sizes, activations = config["hidden_sizes"], config["activations"]
            initialization_seed = config["initialization_seed"]
        if schema_version != 1 or input_dim != 64 or num_classes != 10:
            raise ValueError("unsupported donor schema or digit dimensions")
        if isinstance(hidden_sizes, (str, bytes)) or len(hidden_sizes) != 3:
            raise ValueError("three hidden sizes are required")
        sizes = [_integer(size, "hidden size", 1, 512) for size in hidden_sizes]
        if isinstance(activations, (str, bytes)) or len(activations) != 3 or any(
                activation not in ("relu", "tanh", "gelu") for activation in activations):
            raise ValueError("three supported activation names are required")
        initialization_seed = _seed(initialization_seed)
        self.config = {"schema_version": 1, "input_dim": 64, "num_classes": 10,
                       "hidden_sizes": sizes, "activations": list(activations),
                       "initialization_seed": initialization_seed}
        activation_classes = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        # Initialization is deterministic and cannot alter a caller's receiver RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(64, width), activation_classes[activation](), nn.Linear(width, 10))
                for width, activation in zip(sizes, activations)
            ])

    def forward(self, images: Tensor) -> Tensor:
        if (not isinstance(images, Tensor) or images.layout != torch.strided
                or images.device.type != "cpu" or images.dtype != torch.float32
                or images.ndim != 2 or images.shape[1] != 64 or images.shape[0] < 1
                or not torch.isfinite(images).all()):
            raise ValueError("images must be a finite float32 CPU tensor [positive batch,64]")
        return torch.stack([expert(images) for expert in self.experts], dim=1)


def fit_donors(split: Mapping[str, Any], seed: int, steps: int = 180) -> tuple[DonorBank, Tensor, dict[str, Any]]:
    """Fit the three donors using train identities only, then freeze their bank.

    Validation accuracy is reported without selecting checkpoints. Test outputs
    are cached for later frozen receiver evaluation, but no test score is read.
    """
    seed = _seed(seed)
    steps = _integer(steps, "steps", 1, 20_000)
    if not isinstance(split, Mapping) or any(key not in split for key in ("images", "labels", "train", "validation", "test")):
        raise ValueError("split must contain images, labels and three image identity partitions")
    images = split["images"]
    labels = _long_vector(split["labels"], "labels", upper=10)
    if (not isinstance(images, Tensor) or images.shape != (len(labels), 64)
            or images.dtype != torch.float32 or images.device.type != "cpu"
            or not torch.isfinite(images).all()):
        raise ValueError("images must be finite float32 CPU digit rows")
    partitions = {name: _long_vector(split[name], name, upper=len(labels))
                  for name in ("train", "validation", "test")}
    joined = torch.cat(list(partitions.values()))
    if joined.unique().numel() != joined.numel():
        raise ValueError("training, validation and test identities must be disjoint and unique")
    train_ids, validation_ids = partitions["train"], partitions["validation"]
    train_images, train_labels = images[train_ids], labels[train_ids]
    random = torch.Generator(device="cpu").manual_seed(seed ^ 0x13AE7301)
    losses: list[float] = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(2, previous_threads))
        with torch.random.fork_rng(devices=[]):
            bank = DonorBank(initialization_seed=seed)
            bank.train()
            optimizer = torch.optim.AdamW(bank.parameters(), lr=0.01, weight_decay=0.0001)
            for _ in range(steps):
                chosen = torch.randint(len(train_ids), (min(256, len(train_ids)),), generator=random)
                logits = bank(train_images[chosen])
                targets = train_labels[chosen, None].expand(-1, 3)
                loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            bank.eval()
            bank.requires_grad_(False)
            with torch.no_grad():
                posteriors = bank(images).softmax(-1).detach()
                prediction = posteriors.argmax(-1)
                training_accuracy = (prediction[train_ids] == labels[train_ids, None]).float().mean(0).tolist()
                validation_accuracy = (prediction[validation_ids] == labels[validation_ids, None]).float().mean(0).tolist()
    finally:
        torch.set_num_threads(previous_threads)
    return bank, posteriors, {
        "schema_version": 1, "seed": seed, "steps": steps,
        "training_ids": train_ids.tolist(), "validation_ids": validation_ids.tolist(),
        "train_accuracy": training_accuracy, "validation_accuracy": validation_accuracy,
        "loss_history": losses, "config": dict(bank.config),
        "parameter_counts": [sum(parameter.numel() for parameter in expert.parameters()) for expert in bank.experts],
        "checkpoint_selection": "final fixed training step; no test selection",
        "shared_training_lineage": True,
    }
