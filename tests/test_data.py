"""Behavioral checks for split isolation and controlled donor interventions."""

import copy
import importlib
import io
import json
import random

import numpy as np
import pytest
import torch

try:
    data = importlib.import_module("tether.data")
except ModuleNotFoundError:
    data = None


def require_data():
    assert data is not None, "The tether.data implementation is missing"
    return data


def posterior_fixture():
    labels = torch.arange(40, dtype=torch.long) % 10
    # Each label is independently recognizable; expert-specific confidence is real.
    posterior = torch.full((40, 3, 10), 0.01)
    posterior.scatter_(2, labels[:, None, None].expand(-1, 3, 1), 0.91)
    return posterior, labels


def make_literal_batch(**options):
    module = require_data()
    posterior, labels = posterior_fixture()
    return module.make_batch(posterior, labels, torch.arange(10, 30), 180, 73, **options)


def test_literal_task_targets_and_invalid_operations():
    module = require_data()
    result = module.task_targets(torch.tensor([0, 9, 4, 2, 8, 3]), torch.tensor([0, 1, 2, 1, 2, 0]))
    assert result.tolist() == [0, 0, 5, 3, 1, 3]
    with pytest.raises(ValueError):
        module.task_targets(torch.tensor([0]), torch.tensor([3]))
    with pytest.raises(ValueError):
        module.task_targets(torch.tensor([10]), torch.tensor([0]))
    with pytest.raises((TypeError, ValueError)):
        module.task_targets(torch.tensor([1.5]), torch.tensor([0]))


def test_image_ids_split_before_tasks_with_replayable_manifest():
    module = require_data()
    split = module.load_digits_split()
    assert split["images"].shape == (1797, 64)
    assert split["images"].dtype == torch.float32
    assert split["labels"].dtype == torch.long
    assert 0 <= split["images"].min() <= split["images"].max() <= 1
    ids = [split[name].tolist() for name in ("train", "validation", "test")]
    assert [len(group) for group in ids] == [1078, 359, 360]
    assert len(set(sum(ids, []))) == 1797
    for name in ("train", "validation", "test"):
        counts = torch.bincount(split["labels"][split[name]], minlength=10)
        assert counts.min() >= (100 if name == "train" else 30)
    assert json.loads(json.dumps(split["manifest"])) == split["manifest"]
    repeated = module.load_digits_split()
    assert repeated["manifest"] == split["manifest"]
    assert all(torch.equal(repeated[name], split[name]) for name in ("train", "validation", "test"))


def test_relevant_and_irrelevant_swaps_preserve_context_and_split():
    batch = make_literal_batch(pairs="all")
    posterior, labels = posterior_fixture()
    assert batch["donors"].shape == (180, 3, 3, 10)
    assert set(batch["image_ids"].flatten().tolist()) <= set(range(10, 30))
    assert set(batch["replacement_ids"].tolist()) <= set(range(10, 30))
    for row in range(180):
        source, operation = batch["query"][row].tolist()
        digit = int(labels[batch["image_ids"][row, source]])
        replacement = int(labels[batch["replacement_ids"][row]])
        assert digit != replacement
        assert int(batch["target"][row]) == [digit, (digit + 1) % 10, 9 - digit][operation]
        assert int(batch["relevant_target"][row]) == [replacement, (replacement + 1) % 10, 9 - replacement][operation]
        assert torch.equal(batch["relevant"][row, source], posterior[batch["replacement_ids"][row]])
        other_sources = [index for index in range(3) if index != source]
        assert torch.equal(batch["relevant"][row, other_sources], batch["donors"][row, other_sources])
        assert torch.equal(batch["irrelevant"][row, source], batch["donors"][row, source])
        assert sum(not torch.equal(batch["irrelevant"][row, s], batch["donors"][row, s]) for s in range(3)) == 1
    assert batch["context"].shape == (180, 10)


def test_seen_and_heldout_query_pairs_are_disjoint():
    seen = make_literal_batch(pairs="seen")["query"].tolist()
    heldout = make_literal_batch(pairs="heldout")["query"].tolist()
    excluded = {(0, 2), (1, 1), (2, 0)}
    assert {tuple(row) for row in seen}.isdisjoint(excluded)
    assert {tuple(row) for row in heldout} == excluded
    assert len({tuple(row) for row in seen}) == 6


def test_cue_modes_are_literal_and_sampling_is_reproducible():
    wrong = make_literal_batch(cue="reversed")
    assert torch.all(wrong["context"].argmax(-1) != wrong["target"])
    independent = make_literal_batch(cue="independent")
    accuracy = (independent["context"].argmax(-1) == independent["target"]).float().mean()
    assert 0.02 < accuracy < 0.25
    correlated = make_literal_batch(cue="correlated")
    assert (correlated["context"].argmax(-1) == correlated["target"]).float().mean() > 0.9
    same = make_literal_batch(cue="correlated")
    assert all(torch.equal(value, same[key]) for key, value in correlated.items() if isinstance(value, torch.Tensor))


def test_corruption_is_label_blind_and_common_mode_can_replace_three_experts():
    module = require_data()
    posterior, labels = posterior_fixture()
    for count in (1, 2, 3):
        batch = module.make_batch(posterior, labels, torch.arange(40), 30, 9, damage_count=count)
        source = batch["query"][:, 0]
        changed = (batch["damaged"] != batch["donors"]).any(-1)
        assert torch.all(changed.sum((1, 2)) == count)
        assert torch.all(changed[torch.arange(30), source].sum(-1) == count)
        if count == 3:
            selected = batch["damaged"][torch.arange(30), source]
            assert torch.equal(selected[:, 0], selected[:, 1])
            assert torch.equal(selected[:, 0], selected[:, 2])
        assert torch.allclose(batch["damaged"].sum(-1), torch.ones(30, 3, 3))
    changed_labels = (labels + 3) % 10
    original = module.make_batch(posterior, labels, torch.arange(40), 30, 9)
    relabelled = module.make_batch(posterior, changed_labels, torch.arange(40), 30, 9)
    assert torch.equal(original["image_ids"], relabelled["image_ids"])
    assert torch.equal(original["damaged"], relabelled["damaged"])


def test_local_generators_leave_global_rngs_unchanged():
    module = require_data()
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    python_state = random.getstate()
    module.load_digits_split()
    make_literal_batch()
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after_numpy = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, after_numpy))
    assert random.getstate() == python_state


@pytest.mark.parametrize("options", [
    {"n": 0}, {"n": True}, {"n": 2.5}, {"indices": []},
    {"indices": [-1, 2]}, {"indices": [40]}, {"indices": [1.2, 2.0]},
    {"pairs": "unknown"}, {"cue": "unknown"}, {"damage_count": 0}, {"damage_count": 4},
])
def test_invalid_batch_requests_fail_before_sampling(options):
    module = require_data()
    posterior, labels = posterior_fixture()
    values = dict(posteriors=posterior, labels=labels, indices=torch.arange(40), n=8, seed=4)
    values.update(options)
    with pytest.raises((ValueError, TypeError)):
        module.make_batch(**values)


def test_invalid_posteriors_and_impossible_swap_are_rejected():
    module = require_data()
    posterior, labels = posterior_fixture()
    for invalid in (posterior[:, :, :9], posterior * float("nan"), posterior * 2):
        with pytest.raises(ValueError):
            module.make_batch(invalid, labels, torch.arange(40), 4, 3)
    with pytest.raises(ValueError):
        module.make_batch(posterior, labels, torch.tensor([0, 10, 20]), 4, 3)


def test_donors_train_only_training_images_freeze_and_restore_exactly():
    module = require_data()
    split = module.load_digits_split()
    torch_state = torch.random.get_rng_state().clone()
    bank, posterior, metrics = module.fit_donors(split, seed=11, steps=90)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert posterior.shape == (1797, 3, 10)
    assert not posterior.requires_grad
    assert torch.allclose(posterior.sum(-1), torch.ones(1797, 3), atol=1e-6)
    assert all(not parameter.requires_grad for parameter in bank.parameters())
    assert not bank.training
    assert all(value >= 0.8 for value in metrics["validation_accuracy"])
    assert "test_accuracy" not in metrics
    assert metrics["training_ids"] == split["train"].tolist()
    json.dumps(metrics)
    json.dumps(bank.config)
    clone = module.DonorBank(config=bank.config)
    buffer = io.BytesIO()
    torch.save(bank.state_dict(), buffer)
    buffer.seek(0)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    clone.eval()
    with torch.no_grad():
        assert torch.equal(bank(split["images"][:12]), clone(split["images"][:12]))
    # Changing validation/test pixels cannot change trained weights.
    altered = dict(split)
    altered["images"] = split["images"].clone()
    altered["images"][torch.cat((split["validation"], split["test"]))] = 0
    second, _, _ = module.fit_donors(altered, seed=11, steps=90)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in bank.state_dict().items())
