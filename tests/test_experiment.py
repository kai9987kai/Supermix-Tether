"""End-to-end tests of a tiny real training run and restored inference."""

import json

import pytest


def test_config_rejects_unreproducible_or_empty_study():
    from tether.experiment import StudyConfig

    for options in ({"seeds": ()}, {"seeds": (1, 1)}, {"steps": 0}, {"eval_size": -1}, {"arms": ("unknown",)}, {"phase": "promoted"}):
        with pytest.raises(ValueError):
            StudyConfig(**options)


def test_real_training_checkpoint_and_strict_inference_replay(tmp_path):
    from tether.experiment import StudyConfig, inspect_example, replay_study, run_study

    config = StudyConfig(seeds=(7,), arms=("iit", "tether"), steps=2,
                         donor_steps=2, batch_size=8, eval_size=12,
                         validation_every=1, threads=1)
    study = tmp_path / "study"
    result = run_study(config, study)
    assert result["status"] == "complete"
    assert len(result["rows"]) == 2
    assert result["decision"]["promoted"] is False
    receipt = replay_study(study, strict=True)
    assert receipt["inference_checks"] == 2 * 7 * 6
    import torch
    before_inspection = torch.random.get_rng_state().clone()
    example = inspect_example(study, 0, "tether", "factual")
    assert torch.equal(before_inspection, torch.random.get_rng_state())
    assert len(example["images"]) == 3
    assert all(len(x) == 64 for x in example["images"])
    assert len(example["probabilities"]) == 10
    assert sum(example["probabilities"]) == pytest.approx(1., abs=1e-5)
    with pytest.raises(ValueError):
        inspect_example(study, -1, "tether", "factual")
    with pytest.raises(FileExistsError):
        run_study(config, study)


def test_recomputed_outer_digest_does_not_hide_metric_falsification(tmp_path):
    from tether.experiment import StudyConfig, replay_study, run_study
    from tether.receipts import atomic_json, seal_files

    study = tmp_path / "study"
    run_study(StudyConfig(seeds=(8,), arms=("iit", "tether"), steps=1,
                         donor_steps=1, batch_size=4, eval_size=8, threads=1), study)
    summary = json.loads((study / "summary.json").read_text(encoding="utf-8"))
    summary["rows"][0]["metrics"]["heldout_independent"]["accuracy"] = 1.234
    atomic_json(study / "summary.json", summary)
    paths = json.loads((study / "receipt.json").read_text(encoding="utf-8"))["files"].keys()
    seal_files(study, paths)
    with pytest.raises(ValueError, match="metric"):
        replay_study(study, strict=False)


def test_largest_supported_seed_runs_without_child_rng_overflow(tmp_path):
    from tether.experiment import StudyConfig, run_study

    result = run_study(StudyConfig(seeds=(2**31 - 1,), arms=("iit",), steps=1,
                                  donor_steps=1, batch_size=4, eval_size=4, threads=1), tmp_path / "large-seed")
    assert result["rows"][0]["seed"] == 2**31 - 1


def test_damage_conditions_pair_tasks_and_preserve_the_irrelevant_intervention():
    import torch
    from tether.data import load_digits_split
    from tether.experiment import StudyConfig, _batch

    split = load_digits_split()
    posteriors = torch.nn.functional.one_hot(split["labels"], 10).float()[:, None].expand(-1, 3, -1).clone()
    config = StudyConfig(eval_size=24)
    clean = _batch(posteriors, split, config, 11, "heldout_independent")
    damaged = _batch(posteriors, split, config, 11, "damage_one")
    common = _batch(posteriors, split, config, 11, "damage_common")
    assert torch.equal(clean["image_ids"], damaged["image_ids"])
    assert torch.equal(clean["query"], common["query"])
    assert torch.equal(clean["context"], common["context"])
    rows = torch.arange(24)
    selected = damaged["query"][:, 0]
    assert torch.equal(damaged["donors"][rows, selected], damaged["irrelevant"][rows, selected])
    assert not torch.equal(clean["donors"], damaged["donors"])


def test_existing_empty_directory_cannot_be_reserved_twice(tmp_path):
    from tether.experiment import StudyConfig, run_study

    destination = tmp_path / "reserved"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        run_study(StudyConfig(steps=1, donor_steps=1), destination)


def test_v2_study_runs_evaluates_damage_two_and_replays_strictly(tmp_path):
    from tether.experiment import StudyConfig, inspect_example, replay_study, run_study

    config = StudyConfig(seeds=(9,), arms=("iit", "unpaired", "tether"), steps=2,
                         donor_steps=2, batch_size=8, eval_size=12,
                         validation_every=1, threads=1, protocol="v2")
    study = tmp_path / "v2-study"
    result = run_study(config, study)
    assert result["status"] == "complete"
    assert result["config"]["protocol"] == "v2"
    assert all("damage_two" in row["metrics"] for row in result["rows"])
    assert "zero_shot_damage_two_beats_iit" in result["decision"]["checks"]
    assert "zero_shot_damage_two_beats_unpaired" in result["decision"]["checks"]
    receipt = replay_study(study, strict=True)
    assert receipt["protocol"] == "v2"
    assert receipt["inference_checks"] == 3 * 8 * 6
    example = inspect_example(study, 0, "tether", "damage_two", seed=9)
    assert example["protocol"] == "v2"
    assert example["seed"] == 9
    assert len(example["damaged_experts"]) == 2
    assert len(example["trust_score"]) == 3
