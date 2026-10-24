"""Catch metric falsification, unsafe paths and corrupted saved artifacts."""

import json

import numpy as np
import pytest


def test_metric_replay_uses_predictions_and_literal_targets():
    from tether.receipts import metrics_from_trace

    # Two correct factual decisions, one correct counterfactual, no correct erased answers.
    logits = np.array([[9., 0., 0.], [0., 9., 0.]])
    trace = {
        "target": np.array([0, 1]), "relevant_target": np.array([1, 2]),
        "factual": logits, "relevant": np.array([[0., 9., 0.], [9., 0., 0.]]),
        "irrelevant": logits.copy(), "damaged": logits.copy(),
        "erased": np.array([[0., 9., 0.], [9., 0., 0.]]),
        "shuffled": np.array([[0., 9., 0.], [9., 0., 0.]]),
    }
    metrics = metrics_from_trace(trace)
    assert metrics["accuracy"] == 1.
    assert metrics["relevant_accuracy"] == .5
    assert metrics["irrelevant_consistency"] == 1.
    assert metrics["donor_erasure_gap"] == 1.
    trace["target"] = np.array([2, 2])
    assert metrics_from_trace(trace)["accuracy"] == 0.


def test_paired_statistics_pair_by_seed_not_row_order():
    from tether.receipts import paired_comparisons

    rows = [
        {"seed": 1, "arm": "iit", "metrics": {"heldout_independent": {"accuracy": .2}}},
        {"seed": 2, "arm": "tether", "metrics": {"heldout_independent": {"accuracy": .6}}},
        {"seed": 2, "arm": "iit", "metrics": {"heldout_independent": {"accuracy": .4}}},
        {"seed": 1, "arm": "tether", "metrics": {"heldout_independent": {"accuracy": .4}}},
    ]
    result = paired_comparisons(rows, bootstrap_samples=100)[0]
    assert result["mean_difference"] == pytest.approx(.2)
    assert result["seeds"] == [1, 2]
    assert result["interval"] == pytest.approx([.2, .2])
    rows.pop()
    with pytest.raises(ValueError, match="paired"):
        paired_comparisons(rows)


def test_receipt_rejects_byte_tampering_and_traversal(tmp_path):
    from tether.receipts import atomic_json, seal_files, verify_files, safe_artifact

    atomic_json(tmp_path / "summary.json", {"answer": 3})
    seal_files(tmp_path, ["summary.json"])
    assert verify_files(tmp_path)["files_verified"] == 1
    (tmp_path / "summary.json").write_text('{"answer":4}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_files(tmp_path)
    with pytest.raises(ValueError, match="path"):
        safe_artifact(tmp_path, "../outside.pt")


def test_nonfinite_trace_is_rejected():
    from tether.receipts import metrics_from_trace

    with pytest.raises(ValueError):
        metrics_from_trace({"target": np.array([0]), "factual": np.array([[float("nan"), 0.]])})


def test_unpaired_control_gets_a_separate_seed_matched_comparison():
    from tether.receipts import paired_comparisons
    rows = [{"seed": 1, "arm": arm, "metrics": {"heldout_independent": {"accuracy": accuracy}}}
            for arm, accuracy in (("iit", .5), ("unpaired", .7), ("tether", .6))]
    comparisons = paired_comparisons(rows, bootstrap_samples=20)
    differences = {item["baseline"]: item["mean_difference"] for item in comparisons}
    assert differences == pytest.approx({"iit": .1, "unpaired": -.1})
