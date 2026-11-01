"""Tests for v0.4 Conformal RAPS, Axiomatic Integrated Gradients, Causal Patching & Live Arena."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tether.conformal import (
    adaptive_conformal_inference,
    calibration_metrics,
    conformal_quantile,
    evaluate_conformal_sets,
    evaluate_nex_conformal_sets,
    evaluate_weighted_conformal_sets,
    fit_mahalanobis_forensics,
    fit_temperature,
    helmholtz_free_energy_forensics,
    nex_raps_prediction_sets,
    raps_nonconformity_scores,
    raps_prediction_sets,
    sanitize_specialist_posteriors,
    score_mahalanobis_forensics,
    softmax_temperature,
    weighted_conformal_prediction_sets,
)
from tether.experiment import calibrate_study, inspect_example, live_arena, live_predict
from tether.model import TetherReceiver


def test_axiomatic_integrated_gradients_completeness_and_baseline():
    torch.manual_seed(42)
    model = TetherReceiver(forensic=True)
    model.eval()
    gen = torch.Generator().manual_seed(101)
    donors = (torch.randn(3, 3, 3, 10, generator=gen) * 2.2).softmax(-1)
    query = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
    context = torch.full((3, 10), 0.1)

    ig = model.integrated_gradients(donors, query, context, steps=64)
    assert ig["attributions"].shape == (3, 3, 3, 10)
    assert ig["source_saliency"].shape == (3, 3)
    assert ig["expert_saliency"].shape == (3, 3, 3)
    # Completeness axiom: sum(IG) == F(P) - F(P_0) within numerical quadrature tolerance
    assert (ig["completeness_error"] < 0.025).all()

    # Baseline identity axiom: when P == P_0 (uniform 0.1), attributions must be identically zero
    uniform = torch.full((2, 3, 3, 10), 0.1)
    ig_zero = model.integrated_gradients(uniform, query[:2], context[:2], steps=16)
    assert torch.allclose(ig_zero["attributions"], torch.zeros_like(ig_zero["attributions"]), atol=1e-7)
    assert torch.allclose(ig_zero["completeness_error"], torch.zeros(2), atol=1e-7)


def test_causal_interchange_intervention_patching():
    torch.manual_seed(19)
    model = TetherReceiver(forensic=True)
    model.eval()
    # Base input has digit 1 on source 0; counterfactual has digit 7 on source 0
    base_donors = torch.full((2, 3, 3, 10), 0.01)
    base_donors[:, :, :, 1] = 0.91
    cf_donors = torch.full((2, 3, 3, 10), 0.01)
    cf_donors[:, :, :, 7] = 0.91
    query = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    context = torch.full((2, 10), 0.1)

    res = model.interchange_intervention(base_donors, query, context, cf_donors)
    patches = res["patches"]
    assert set(patches) == {
        "expert_weights",
        "source_posteriors",
        "source_weights",
        "pooled_posterior",
        "task_logits",
        "shortcut_logits",
    }
    # Patching source_posteriors, pooled_posterior, or task_logits on a pure donor change restores 100% of cf logits
    for node in ("source_posteriors", "pooled_posterior", "task_logits"):
        assert torch.allclose(patches[node]["restoration"], torch.ones(2), atol=1e-4)
        assert torch.allclose(patches[node]["kl_to_cf"], torch.zeros(2), atol=1e-5)
    # Because context did not change between base and cf, patching shortcut_logits restores 0%
    assert torch.allclose(patches["shortcut_logits"]["restoration"], torch.zeros(2), atol=1e-5)


def test_tri_part_uncertainty_decomposition_separates_conflict_from_common_mode_spoof():
    model = TetherReceiver(forensic=True)
    model.eval()
    # Natural clean posteriors agreeing on class 3 with realistic runner-up tail (concentrated on 8 and 5)
    intact = torch.tensor([0.001, 0.001, 0.002, 0.85, 0.001, 0.04, 0.001, 0.001, 0.102, 0.001])
    spoof = torch.tensor([0.001] * 9 + [0.991])  # class 9 spoof with flat uniform tail

    donors_clean = intact.view(1, 1, 1, 10).expand(1, 3, 3, 10).clone()
    donors_partial = donors_clean.clone()
    donors_partial[0, 0, 0] = spoof  # 1/3 corrupted on source 0
    donors_partial[0, 0, 1] = spoof  # 2/3 corrupted on source 0
    donors_common = donors_clean.clone()
    donors_common[0, 0, :] = spoof  # 3/3 common-mode corrupted on source 0

    query = torch.tensor([[0, 0]], dtype=torch.long)
    context = torch.full((1, 10), 0.1)

    d_clean = model.inspect(donors_clean, query, context)
    d_partial = model.inspect(donors_partial, query, context)
    d_common = model.inspect(donors_common, query, context)

    # Clean: zero conflict JSD, low spoof risk
    assert d_clean["queried_conflict_jsd"][0].item() < 1e-5
    assert d_clean["forensic_spoof_risk"][0].item() < 0.50

    # Partial 2/3 corruption: high inter-specialist conflict JSD, but intact expert keeps spoof risk < 0.50
    assert d_partial["queried_conflict_jsd"][0].item() > 0.15
    assert d_partial["forensic_spoof_risk"][0].item() < 0.50
    assert d_partial["abstain"][0].item() is False

    # Common-mode 3/3 spoofing: JSD is 0.0 (all 3 agree!), but forensic spoof risk hits 1.0!
    assert d_common["queried_conflict_jsd"][0].item() < 1e-5
    assert d_common["forensic_spoof_risk"][0].item() > 0.99
    assert d_common["abstain"][0].item() is True


def test_raps_conformal_prediction_and_temperature_calibration():
    rng = np.random.default_rng(123)
    n_cal, n_test, k = 200, 200, 10
    targets_cal = rng.integers(0, k, size=n_cal)
    targets_test = rng.integers(0, k, size=n_test)

    # Create overconfident uncalibrated logits (80% top-1 accuracy, true class is rank 2 on the remaining 20%, scaled by 4x)
    pred_cal = targets_cal.copy()
    pred_cal[:40] = (pred_cal[:40] + 1) % k
    logits_cal = rng.normal(0, 0.3, size=(n_cal, k))
    logits_cal[np.arange(n_cal), targets_cal] += 2.2
    logits_cal[np.arange(n_cal), pred_cal] += 3.0
    logits_cal *= 4.0

    pred_test = targets_test.copy()
    pred_test[:40] = (pred_test[:40] + 1) % k
    logits_test = rng.normal(0, 0.3, size=(n_test, k))
    logits_test[np.arange(n_test), targets_test] += 2.2
    logits_test[np.arange(n_test), pred_test] += 3.0
    logits_test *= 4.0

    temp_res = fit_temperature(logits_cal, targets_cal)
    assert temp_res["temperature"] > 1.5
    assert temp_res["post_scaling"]["nll"] < temp_res["pre_scaling"]["nll"]
    assert temp_res["post_scaling"]["ece"] < temp_res["pre_scaling"]["ece"]

    probs_cal = softmax_temperature(logits_cal, temperature=temp_res["temperature"])
    probs_test = softmax_temperature(logits_test, temperature=temp_res["temperature"])

    scores = raps_nonconformity_scores(probs_cal, targets_cal, k_reg=1, lam_reg=0.02)
    q_hat = conformal_quantile(scores, alpha=0.10)
    eval_res = evaluate_conformal_sets(probs_test, targets_test, q_hat, k_reg=1, lam_reg=0.02)
    assert eval_res["marginal_coverage"] >= 0.85
    assert 1.0 <= eval_res["mean_set_size"] < 3.0

    # Verify epistemic abstention gates conformal sets to empty sets
    low_trust = np.zeros(n_test)
    abstain_res = raps_prediction_sets(probs_test, q_hat, epistemic_trust=low_trust, epistemic_threshold=0.10)
    assert abstain_res["abstention_rate"] == 1.0
    assert abstain_res["mean_set_size"] == 0.0
    assert all(len(s) == 0 for s in abstain_res["prediction_sets"])


def test_v2_confirmation_conformal_and_live_arena():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")

    # 1. Calibrate study on seed 101
    cal = calibrate_study(study_path, alpha=0.05, seed=101)
    assert cal["schema"] == "supermix-tether-conformal-v1"
    tether_summary = cal["summary_by_arm"]["tether"]
    assert tether_summary["conditions"]["heldout_independent"]["marginal_coverage"] >= 0.94
    assert tether_summary["conditions"]["heldout_independent"]["mean_set_size"] < 1.25
    assert tether_summary["conditions"]["damage_two"]["marginal_coverage"] >= 0.93
    assert tether_summary["conditions"]["damage_common"]["abstention_rate"] == 1.0

    # 2. Live arena on 2/3 specialist corruption
    arena = live_arena(
        study_path, digit=3, source=0, operation=0,
        tamper_mode="corrupt_2", corrupted_class=8, confidence=0.991, seed=101,
    )
    by_arm = {a["arm"]: a for a in arena["arms"]}
    assert by_arm["tether"]["survived"] is True
    assert by_arm["tether"]["prediction"] == 3
    assert by_arm["tether"]["intact_weight_share"] > 0.90
    assert 3 in by_arm["tether"]["conformal_set"]
    assert "tether" in arena["sweep"]["curves"]
    assert len(arena["sweep"]["curves"]["tether"]["target_probabilities"]) == 6

    # 3. Inspect example includes IG, Shapley values, and circuit patching
    ex = inspect_example(study_path, index=0, arm="tether", mode="damage_two", seed=101)
    assert "conformal_set" in ex
    assert "nex_conformal_set" in ex
    assert "ig_expert_saliency" in ex
    assert "shapley_values" in ex
    assert len(ex["shapley_values"]) == 3
    assert ex["shapley_efficiency_error"] < 1e-4
    assert "interchange_patching" in ex
    assert ex["ig_completeness_error"] < 0.05


def test_cooperative_shapley_values_and_shapley_taylor_interactions():
    torch.manual_seed(77)
    model = TetherReceiver(forensic=True)
    model.eval()
    gen = torch.Generator().manual_seed(2026)
    donors = (torch.randn(4, 3, 3, 10, generator=gen) * 2.5).softmax(-1)
    query = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 2]], dtype=torch.long)
    context = torch.full((4, 10), 0.1)

    sh = model.shapley_interactions(donors, query, context)
    assert sh["shapley_values"].shape == (4, 3)
    assert sh["interaction_matrix"].shape == (4, 3, 3)
    # Exact Shapley Efficiency: sum_e phi_e == v({0,1,2}) - v(empty)
    assert (sh["efficiency_error"] < 1e-5).all()
    # Exact Order-2 Shapley-Taylor Efficiency: sum_{i,j} I^(2)_{i,j} == v({0,1,2}) - v(empty)
    assert (sh["taylor_efficiency_error"] < 1e-5).all()

    # Baseline identity: when queried-source specialists equal uniform 0.1, all Shapley values are 0
    uniform = torch.full((2, 3, 3, 10), 0.1)
    sh_zero = model.shapley_interactions(uniform, query[:2], context[:2])
    assert torch.allclose(sh_zero["shapley_values"], torch.zeros((2, 3)), atol=1e-6)
    assert torch.allclose(sh_zero["interaction_matrix"], torch.zeros((2, 3, 3)), atol=1e-6)


def test_mahalanobis_forensics_and_stealth_spoof_sanitization():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")

    # 1. Under stealth_3 (3/3 dark-knowledge tail-mimicking spoof), scalar tail entropy is tricked without defense
    res_undefended = live_predict(
        study_path, digit=3, source=0, operation=0,
        tamper_mode="stealth_3", corrupted_class=8, confidence=0.96, seed=101, defense=False,
    )
    assert res_undefended["epistemic_trust"] > 0.50  # 1D tail entropy is fooled by decoy runner-up spike
    assert res_undefended["abstain"] is False
    # But Mahalanobis detector flags all 3 stealth-spoofed specialists
    assert all(res_undefended["flagged_specialists"])
    assert res_undefended["sanitized_abstain"] is True

    # 2. With Mahalanobis defense enabled, stealth_3 is safely gated to abstention and empty conformal set
    res_defended = live_predict(
        study_path, digit=3, source=0, operation=0,
        tamper_mode="stealth_3", corrupted_class=8, confidence=0.96, seed=101, defense=True,
    )
    assert res_defended["abstain"] is True
    assert res_defended["selective_prediction"] == -1
    assert res_defended["conformal_set"] == []

    # 3. Under stealth_2 (2/3 stealth tail-mimicry), Mahalanobis defense sanitizes the 2 spoofed specialists
    # and restores the clean digit prediction (3) even for vulnerable arms in live_arena
    arena_def = live_arena(
        study_path, digit=3, source=0, operation=0,
        tamper_mode="stealth_2", corrupted_class=8, confidence=0.96, seed=101, defense=True,
    )
    by_arm = {a["arm"]: a for a in arena_def["arms"]}
    assert by_arm["tether"]["survived"] is True
    assert by_arm["tether"]["prediction"] == 3
    assert by_arm["iit"]["survived"] is True
    assert by_arm["iit"]["prediction"] == 3


def test_adaptive_and_non_exchangeable_conformal_under_shift():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")

    cal = calibrate_study(study_path, alpha=0.05, seed=101)
    iit_d2 = cal["summary_by_arm"]["iit"]["conditions"]["damage_two"]
    # Standard exchangeable RAPS miscovers on damage_two for IIT (~80% coverage),
    # whereas Non-Exchangeable RAPS and Online ACI substantially boost coverage under shift
    assert iit_d2["nex_marginal_coverage"] > iit_d2["marginal_coverage"] + 0.08
    assert iit_d2["aci_marginal_coverage"] > iit_d2["marginal_coverage"] + 0.08
    assert iit_d2["aci_marginal_coverage"] >= 0.90
    assert iit_d2["wcp_marginal_coverage"] > iit_d2["marginal_coverage"] + 0.08


def test_williams_beer_pid_conservation_and_regime_shift():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")

    ex_clean = inspect_example(study_path, index=0, arm="tether", mode="factual", seed=101)
    pid_clean = ex_clean["pid"]
    assert pid_clean["conservation_error"] < 1e-5
    assert pid_clean["redundancy"] >= 0.0
    assert pid_clean["synergy"] >= 0.0
    assert all(u >= 0.0 for u in pid_clean["unique"])
    # On clean inputs where all 3 specialists agree, Shared Redundancy R dominates
    assert pid_clean["normalized_shares"]["redundancy"] > 0.65

    # Under 2-of-3 specialist corruption (damage_two), Shared Redundancy R drops to 0
    # and the lone intact specialist's Unique Information U_intact dominates
    ex_d2 = inspect_example(study_path, index=0, arm="tether", mode="damage_two", seed=101)
    pid_d2 = ex_d2["pid"]
    assert pid_d2["conservation_error"] < 1e-5
    assert pid_d2["redundancy"] == pytest.approx(0.0, abs=1e-5)
    assert max(pid_d2["normalized_shares"]["unique"]) > 0.80


def test_causal_mediation_and_critical_breakdown_threshold():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")

    # Under adversarial context proxy cue + 2/3 corruption, Tether maintains high visual dominance
    pred_tether = live_predict(
        study_path, digit=3, source=0, operation=0,
        tamper_mode="corrupt_2", corrupted_class=8, confidence=0.98,
        seed=101, arm="tether", context_mode="adversarial",
    )
    assert pred_tether["prediction"] == 3
    assert pred_tether["causal_mediation"]["visual_dominance_ratio"] > 0.80
    assert pred_tether["critical_corruption_threshold"] > 0.95
    assert 3 in pred_tether["wcp_conformal_set"]

    # Helmholtz free-energy forensics separates flat-tail corrupted experts (N_eff ~ 9) from intact expert
    pr = pred_tether["spectral_participation_ratio"]
    dmg = set(pred_tether["damaged_experts"])
    for e_idx in range(3):
        if e_idx in dmg:
            assert pr[e_idx] > 8.5
        else:
            assert pr[e_idx] < 8.5

