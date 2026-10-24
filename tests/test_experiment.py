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
    example = inspect_example(study, 0, "tether", "factual")
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
