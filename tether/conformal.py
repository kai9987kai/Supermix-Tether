"""Split Conformal Prediction (RAPS), Non-Exchangeable & Adaptive Conformal Inference,
and Class-Conditional Mahalanobis Dark-Knowledge Forensics.

Implements:
1. Post-hoc Temperature Scaling and Expected Calibration Error (ECE / Brier / NLL)
   following Guo et al. (ICML 2017, arXiv:1706.04599).
2. Regularized Adaptive Prediction Sets (RAPS) with finite-sample marginal
   coverage guarantees following Angelopoulos et al. (ICLR 2021, arXiv:2009.14193).
3. Epistemic-Gated Selective Conformal Risk Control (Angelopoulos et al.,
   ICLR 2024, arXiv:2208.02814), collapsing prediction sets to the empty set
   (abstention) when dark-knowledge specialist trust falls below threshold.
4. Non-Exchangeable Conflict-Adjusted Conformal Prediction (Nex-RAPS) and Online
   Adaptive Conformal Inference (ACI) under distribution shift following
   Barber, Candes, Ramdas & Tibshirani (Annals of Statistics 2023, arXiv:2202.13415)
   and Gibbs & Candes (NeurIPS 2021, arXiv:2106.00170).
5. Class-Conditional Mahalanobis Dark-Knowledge Forensics & Byzantine-Resilient
   Specialist Sanitization following Lee et al. (NeurIPS 2018, arXiv:1807.03888)
   and Yin et al. (ICML 2018, arXiv:1803.01498).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable temperature-scaled softmax probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits must be a 2D array [batch, classes]")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain finite values")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    scaled = logits / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibration_metrics(logits: np.ndarray, targets: np.ndarray,
                        temperature: float = 1.0, n_bins: int = 15) -> dict[str, float]:
    """Compute ECE, MCE, Brier score, and Negative Log-Likelihood (NLL)."""
    probs = softmax_temperature(logits, temperature=temperature)
    targets = np.asarray(targets, dtype=np.int64)
    n, k = probs.shape
    if targets.shape != (n,) or (targets < 0).any() or (targets >= k).any():
        raise ValueError("targets must be valid class indices matching logits batch size")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        prop = float(in_bin.mean())
        if prop > 0.0:
            gap = abs(float(accuracies[in_bin].mean()) - float(confidences[in_bin].mean()))
            ece += prop * gap
            mce = max(mce, gap)

    one_hot = np.eye(k, dtype=np.float64)[targets]
    brier = float(np.square(probs - one_hot).sum(axis=1).mean())
    true_probs = np.clip(probs[np.arange(n), targets], 1e-12, 1.0)
    nll = float(-np.log(true_probs).mean())

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "nll": float(nll),
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
        "temperature": float(temperature),
    }


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Fit optimal post-hoc temperature T* on validation logits via NLL minimization."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    pre = calibration_metrics(logits, targets, temperature=1.0)

    # Coarse log-spaced grid followed by fine local refinement around optimum
    coarse_grid = np.exp(np.linspace(math.log(0.15), math.log(5.0), 81))
    best_t = 1.0
    best_nll = pre["nll"]
    for t in coarse_grid:
        nll = calibration_metrics(logits, targets, temperature=float(t))["nll"]
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    fine_low = max(0.05, best_t * 0.75)
    fine_high = min(10.0, best_t * 1.35)
    for t in np.linspace(fine_low, fine_high, 61):
        nll = calibration_metrics(logits, targets, temperature=float(t))["nll"]
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    post = calibration_metrics(logits, targets, temperature=best_t)
    return {
        "temperature": round(best_t, 5),
        "pre_scaling": pre,
        "post_scaling": post,
    }


def raps_nonconformity_scores(probabilities: np.ndarray, targets: np.ndarray,
                              k_reg: int = 1, lam_reg: float = 0.02) -> np.ndarray:
    """Compute Regularized Adaptive Prediction Sets (RAPS) non-conformity scores."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes]")
    n, k = probs.shape
    if targets.shape != (n,) or (targets < 0).any() or (targets >= k).any():
        raise ValueError("targets must match batch size and class range")
    if k_reg < 0 or lam_reg < 0 or not math.isfinite(lam_reg):
        raise ValueError("k_reg and lam_reg must be non-negative")

    # Sort probabilities descending; tie-break deterministically by class index
    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    # Cumulative probability mass strictly preceding rank r, plus (1 - p_y) and RAPS rank penalty
    cumsum_before = np.cumsum(sorted_probs, axis=1) - sorted_probs

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rank_idx = int(np.where(order[i] == targets[i])[0][0])
        rank_1based = rank_idx + 1
        penalty = lam_reg * max(0, rank_1based - k_reg)
        scores[i] = cumsum_before[i, rank_idx] + (1.0 - probs[i, targets[i]]) * 0.5 + penalty
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the finite-sample split conformal quantile q_hat for miscoverage alpha."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 1D array")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = scores.size
    level = min(1.0, math.ceil((n + 1) * (1.0 - float(alpha))) / float(n))
    try:
        q = float(np.quantile(scores, level, method="higher"))
    except TypeError:
        q = float(np.quantile(scores, level, interpolation="higher"))
    return max(q, 1e-6)


def raps_prediction_sets(probabilities: np.ndarray, q_hat: float,
                         k_reg: int = 1, lam_reg: float = 0.02,
                         epistemic_trust: np.ndarray | None = None,
                         epistemic_threshold: float = 0.10) -> dict[str, Any]:
    """Construct RAPS prediction sets {y : s(x, y) <= q_hat} with Epistemic-Gated Abstention."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes]")
    if not isinstance(q_hat, (int, float)) or not math.isfinite(q_hat) or q_hat <= 0:
        raise ValueError("q_hat must be a finite positive quantile")
    n, k = probs.shape

    if epistemic_trust is not None:
        trust = np.asarray(epistemic_trust, dtype=np.float64)
        if trust.shape != (n,):
            raise ValueError("epistemic_trust must have shape [batch]")
        abstained = trust < float(epistemic_threshold)
    else:
        abstained = np.zeros(n, dtype=bool)

    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum_before = np.cumsum(sorted_probs, axis=1) - sorted_probs
    ranks = np.arange(1, k + 1, dtype=np.float64)[None, :]
    penalties = lam_reg * np.maximum(0.0, ranks - float(k_reg))
    candidate_scores = cumsum_before + (1.0 - sorted_probs) * 0.5 + penalties

    sets: list[list[int]] = []
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if abstained[i]:
            sets.append([])
            sizes[i] = 0
            continue
        included_ranks = np.where(candidate_scores[i] <= q_hat)[0]
        if included_ranks.size == 0:
            chosen = [int(order[i, 0])]
        else:
            chosen = [int(order[i, r]) for r in included_ranks]
        sets.append(chosen)
        sizes[i] = len(chosen)

    accepted = ~abstained
    return {
        "prediction_sets": sets,
        "set_sizes": sizes.tolist(),
        "abstained": abstained.tolist(),
        "mean_set_size": float(sizes.mean()),
        "accepted_mean_set_size": float(sizes[accepted].mean()) if accepted.any() else 0.0,
        "singleton_rate": float((sizes == 1).mean()),
        "abstention_rate": float(abstained.mean()),
        "q_hat": float(q_hat),
    }


def evaluate_conformal_sets(probabilities: np.ndarray, targets: np.ndarray,
                            q_hat: float, k_reg: int = 1, lam_reg: float = 0.02,
                            epistemic_trust: np.ndarray | None = None,
                            epistemic_threshold: float = 0.10) -> dict[str, Any]:
    """Evaluate empirical coverage, selective risk, and set efficiency of RAPS sets."""
    targets = np.asarray(targets, dtype=np.int64)
    res = raps_prediction_sets(
        probabilities, q_hat, k_reg=k_reg, lam_reg=lam_reg,
        epistemic_trust=epistemic_trust, epistemic_threshold=epistemic_threshold,
    )
    sets = res["prediction_sets"]
    abstained = np.asarray(res["abstained"], dtype=bool)
    covered = np.array([int(t) in s for t, s in zip(targets, sets)], dtype=bool)
    accepted = ~abstained
    selective_coverage = float(covered[accepted].mean()) if accepted.any() else 1.0
    selective_risk = float((~covered & accepted).mean())
    return {
        "marginal_coverage": float(covered.mean()),
        "selective_coverage": selective_coverage,
        "selective_miscoverage_risk": selective_risk,
        "mean_set_size": res["mean_set_size"],
        "accepted_mean_set_size": res["accepted_mean_set_size"],
        "singleton_rate": res["singleton_rate"],
        "abstention_rate": res["abstention_rate"],
        "q_hat": res["q_hat"],
    }


def nex_raps_prediction_sets(logits: np.ndarray, temperature: float, q_hat: float,
                             conflict_jsd: np.ndarray, cal_conflict_q95: float = 0.04,
                             conflict_temp_gain: float = 12.0, conflict_q_gain: float = 6.0,
                             k_reg: int = 1, lam_reg: float = 0.02,
                             epistemic_trust: np.ndarray | None = None,
                             epistemic_threshold: float = 0.10) -> dict[str, Any]:
    """Non-Exchangeable Conflict-Adjusted RAPS (Barber et al., Annals of Statistics 2023).

    When test-time inter-specialist Jensen-Shannon conflict exceeds the 95th percentile
    of the exchangeable calibration split, effective temperature and conformal threshold
    expand proportionally to the observed specialist shift, restoring coverage under
    1/3 and 2/3 specialist corruption without requiring test labels.
    """
    logits = np.asarray(logits, dtype=np.float64)
    conflict = np.asarray(conflict_jsd, dtype=np.float64)
    n, k = logits.shape
    if conflict.shape != (n,):
        raise ValueError("conflict_jsd must have shape [batch]")

    if epistemic_trust is not None:
        trust = np.asarray(epistemic_trust, dtype=np.float64)
        abstained = trust < float(epistemic_threshold)
    else:
        abstained = np.zeros(n, dtype=bool)

    excess_conflict = np.maximum(0.0, conflict - float(cal_conflict_q95))
    eff_temps = float(temperature) * (1.0 + float(conflict_temp_gain) * excess_conflict)
    eff_q_hats = float(q_hat) + float(conflict_q_gain) * excess_conflict

    scaled = logits / eff_temps[:, None]
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum_before = np.cumsum(sorted_probs, axis=1) - sorted_probs
    ranks = np.arange(1, k + 1, dtype=np.float64)[None, :]
    penalties = lam_reg * np.maximum(0.0, ranks - float(k_reg))
    candidate_scores = cumsum_before + (1.0 - sorted_probs) * 0.5 + penalties

    sets: list[list[int]] = []
    sizes = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if abstained[i]:
            sets.append([])
            sizes[i] = 0
            continue
        included_ranks = np.where(candidate_scores[i] <= eff_q_hats[i])[0]
        if included_ranks.size == 0:
            chosen = [int(order[i, 0])]
        else:
            chosen = [int(order[i, r]) for r in included_ranks]
        sets.append(chosen)
        sizes[i] = len(chosen)

    accepted = ~abstained
    return {
        "prediction_sets": sets,
        "set_sizes": sizes.tolist(),
        "abstained": abstained.tolist(),
        "mean_set_size": float(sizes.mean()),
        "accepted_mean_set_size": float(sizes[accepted].mean()) if accepted.any() else 0.0,
        "singleton_rate": float((sizes == 1).mean()),
        "abstention_rate": float(abstained.mean()),
        "mean_effective_q_hat": float(eff_q_hats.mean()),
    }


def evaluate_nex_conformal_sets(logits: np.ndarray, targets: np.ndarray,
                                temperature: float, q_hat: float,
                                conflict_jsd: np.ndarray, cal_conflict_q95: float = 0.04,
                                k_reg: int = 1, lam_reg: float = 0.02,
                                epistemic_trust: np.ndarray | None = None,
                                epistemic_threshold: float = 0.10) -> dict[str, Any]:
    """Evaluate Non-Exchangeable Conflict-Adjusted RAPS coverage and set size."""
    targets = np.asarray(targets, dtype=np.int64)
    res = nex_raps_prediction_sets(
        logits, temperature=temperature, q_hat=q_hat,
        conflict_jsd=conflict_jsd, cal_conflict_q95=cal_conflict_q95,
        k_reg=k_reg, lam_reg=lam_reg,
        epistemic_trust=epistemic_trust, epistemic_threshold=epistemic_threshold,
    )
    sets = res["prediction_sets"]
    abstained = np.asarray(res["abstained"], dtype=bool)
    covered = np.array([int(t) in s for t, s in zip(targets, sets)], dtype=bool)
    accepted = ~abstained
    selective_coverage = float(covered[accepted].mean()) if accepted.any() else 1.0
    selective_risk = float((~covered & accepted).mean())
    return {
        "marginal_coverage": float(covered.mean()),
        "selective_coverage": selective_coverage,
        "selective_miscoverage_risk": selective_risk,
        "mean_set_size": res["mean_set_size"],
        "accepted_mean_set_size": res["accepted_mean_set_size"],
        "singleton_rate": res["singleton_rate"],
        "abstention_rate": res["abstention_rate"],
        "mean_effective_q_hat": res["mean_effective_q_hat"],
    }


def adaptive_conformal_inference(cal_scores: np.ndarray, probabilities: np.ndarray,
                                 targets: np.ndarray, alpha: float = 0.05,
                                 gamma: float = 0.05, k_reg: int = 1, lam_reg: float = 0.02,
                                 epistemic_trust: np.ndarray | None = None,
                                 epistemic_threshold: float = 0.10) -> dict[str, Any]:
    """Online Adaptive Conformal Inference (ACI; Gibbs & Candes, NeurIPS 2021, arXiv:2106.00170).

    Tracks non-stationary distribution shifts online by updating:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    and extrapolating the nonconformity threshold smoothly when alpha_t <= 0.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n, k = probs.shape

    if epistemic_trust is not None:
        trust = np.asarray(epistemic_trust, dtype=np.float64)
        abstained = trust < float(epistemic_threshold)
    else:
        abstained = np.zeros(n, dtype=bool)

    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum_before = np.cumsum(sorted_probs, axis=1) - sorted_probs
    ranks = np.arange(1, k + 1, dtype=np.float64)[None, :]
    penalties = lam_reg * np.maximum(0.0, ranks - float(k_reg))
    candidate_scores = cumsum_before + (1.0 - sorted_probs) * 0.5 + penalties

    sorted_cal = np.sort(cal_scores)
    m_cal = sorted_cal.size
    max_cal = float(sorted_cal[-1])
    max_possible = 1.5 + lam_reg * max(0, k - k_reg)
    alpha_t = float(alpha)
    sets: list[list[int]] = []
    sizes = np.zeros(n, dtype=np.int64)
    covered = np.zeros(n, dtype=bool)
    alpha_trajectory = np.zeros(n, dtype=np.float64)

    for t in range(n):
        alpha_trajectory[t] = alpha_t
        if abstained[t]:
            sets.append([])
            sizes[t] = 0
            covered[t] = False
            continue
        if alpha_t <= 0.002:
            # Extrapolate threshold above max calibration score under sustained shift
            surge = min(1.0, (0.002 - alpha_t) / max(float(alpha), 1e-3))
            q_t = max_cal + surge * (max_possible - max_cal)
        elif alpha_t >= 0.995:
            q_t = 1e-6
        else:
            a_clip = float(np.clip(alpha_t, 0.002, 0.995))
            idx_q = min(m_cal - 1, max(0, math.ceil((m_cal + 1) * (1.0 - a_clip)) - 1))
            q_t = max(float(sorted_cal[idx_q]), 1e-6)

        included_ranks = np.where(candidate_scores[t] <= q_t)[0]
        if included_ranks.size == 0:
            chosen = [int(order[t, 0])]
        else:
            chosen = [int(order[t, r]) for r in included_ranks]
        sets.append(chosen)
        sizes[t] = len(chosen)
        is_cov = int(targets[t]) in chosen
        covered[t] = is_cov
        err_t = 0.0 if is_cov else 1.0
        alpha_t = float(np.clip(alpha_t + float(gamma) * (float(alpha) - err_t), -float(alpha), 1.0))

    accepted = ~abstained
    return {
        "prediction_sets": sets,
        "set_sizes": sizes.tolist(),
        "marginal_coverage": float(covered.mean()),
        "selective_coverage": float(covered[accepted].mean()) if accepted.any() else 1.0,
        "selective_miscoverage_risk": float((~covered & accepted).mean()),
        "mean_set_size": float(sizes.mean()),
        "accepted_mean_set_size": float(sizes[accepted].mean()) if accepted.any() else 0.0,
        "singleton_rate": float((sizes == 1).mean()),
        "abstention_rate": float(abstained.mean()),
        "final_alpha_t": float(alpha_t),
        "mean_alpha_t": float(alpha_trajectory.mean()),
    }


def _runner_up_log_signature(posteriors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract predicted class, 9D class-ordered runner-up signature, and deep-tail spectral dispersion."""
    p = np.asarray(posteriors, dtype=np.float64)
    orig_shape = p.shape[:-1]
    flat = p.reshape(-1, 10)
    m = flat.shape[0]
    pred_classes = flat.argmax(axis=1)
    sigs = np.zeros((m, 9), dtype=np.float64)
    deep_disp = np.zeros(m, dtype=np.float64)
    for i in range(m):
        c = pred_classes[i]
        mask = [idx for idx in range(10) if idx != c]
        tail = np.clip(flat[i, mask], 1e-8, 1.0)
        tail_norm = tail / tail.sum()
        # Probability-space runner-up confusion vector + sorted log-rank decay
        sorted_tail = np.sort(tail_norm)[::-1]
        log_sorted = np.log(np.clip(sorted_tail, 1e-8, 1.0))
        # Deep tail (ranks 3..10, i.e., sorted_tail[1:]) log standard deviation
        deep_disp[i] = float(np.std(log_sorted[1:]))
        sigs[i] = tail_norm
    return pred_classes.reshape(orig_shape), sigs.reshape(*orig_shape, 9), deep_disp.reshape(orig_shape)


def fit_mahalanobis_forensics(clean_posteriors: np.ndarray, reg: float = 0.015) -> dict[str, Any]:
    """Fit Class-Conditional Mahalanobis & Spectral Dark-Knowledge Forensics (Lee et al., NeurIPS 2018).

    Given clean specialist posteriors of shape [N, 3_experts, 10], fits class-conditional
    Gaussian means and regularized precision matrices on the 9D runner-up confusion
    distribution and deep-tail spectral dispersion for each (expert, predicted_class) pair.
    """
    p = np.asarray(clean_posteriors, dtype=np.float64)
    if p.ndim == 4:
        p = p.reshape(-1, p.shape[-2], p.shape[-1])
    if p.ndim != 3 or p.shape[1] != 3 or p.shape[2] != 10:
        raise ValueError("clean_posteriors must have shape [N, 3, 10] or [B, 3, 3, 10]")

    preds, sigs, deep_disp = _runner_up_log_signature(p)
    means = np.zeros((3, 10, 9), dtype=np.float64)
    precisions = np.zeros((3, 10, 9, 9), dtype=np.float64)
    thresholds = np.zeros((3, 10), dtype=np.float64)
    disp_q05 = np.zeros((3, 10), dtype=np.float64)

    for e in range(3):
        global_mean = sigs[:, e, :].mean(axis=0)
        global_cov = np.cov(sigs[:, e, :], rowvar=False) + float(reg) * np.eye(9)
        global_prec = np.linalg.inv(global_cov)
        global_disp = max(0.25, float(np.quantile(deep_disp[:, e], 0.02)))
        for c in range(10):
            idx = np.where(preds[:, e] == c)[0]
            if idx.size >= 5:
                sub = sigs[idx, e, :]
                mu = sub.mean(axis=0)
                cov = np.cov(sub, rowvar=False) + float(reg) * np.eye(9)
                prec = np.linalg.inv(cov)
                diff = sub - mu[None, :]
                dists = np.sqrt(np.maximum(0.0, np.einsum("ni,ij,nj->n", diff, prec, diff)))
                thr = float(np.quantile(dists, 0.98)) + 0.85
                d_low = max(0.25, float(np.quantile(deep_disp[idx, e], 0.02)) * 0.65)
            else:
                mu = global_mean
                prec = global_prec
                thr = 4.5
                d_low = global_disp * 0.65
            means[e, c] = mu
            precisions[e, c] = prec
            thresholds[e, c] = max(thr, 2.8)
            disp_q05[e, c] = d_low

    return {
        "means": means,
        "precisions": precisions,
        "thresholds": thresholds,
        "disp_q05": disp_q05,
        "reg": float(reg),
    }


def score_mahalanobis_forensics(posteriors: np.ndarray, detector: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute Class-Conditional Spectral-Mahalanobis distance and Hybrid Forensic Trust."""
    p = np.asarray(posteriors, dtype=np.float64)
    if p.ndim != 4 or p.shape[1:] != (3, 3, 10):
        raise ValueError("posteriors must have shape [batch, 3, 3, 10]")
    b = p.shape[0]
    preds, sigs, deep_disp = _runner_up_log_signature(p)
    means = detector["means"]
    precisions = detector["precisions"]
    thresholds = detector["thresholds"]
    disp_q05 = detector.get("disp_q05", np.full((3, 10), 0.25))

    m_dist = np.zeros((b, 3, 3), dtype=np.float64)
    m_trust = np.zeros((b, 3, 3), dtype=np.float64)
    tail_trust = np.zeros((b, 3, 3), dtype=np.float64)

    for i in range(b):
        for s in range(3):
            for e in range(3):
                c = int(preds[i, s, e])
                vec = p[i, s, e]
                # 1. Standard 1D runner-up tail entropy trust
                mask = [idx for idx in range(10) if idx != c]
                tail = np.clip(vec[mask], 1e-8, 1.0)
                tail_norm = tail / tail.sum()
                h_tail = -float((tail_norm * np.log(tail_norm)).sum()) / math.log(9.0)
                t_ent = max(0.0, min(1.0, 1.0 - h_tail))
                tail_trust[i, s, e] = t_ent

                # 2. Class-conditional Mahalanobis distance + deep-tail spectral flatness penalty
                diff = sigs[i, s, e] - means[e, c]
                prec = precisions[e, c]
                cov_dist = math.sqrt(max(0.0, float(diff @ prec @ diff)))
                disp_deficit = max(0.0, float(disp_q05[e, c]) - float(deep_disp[i, s, e])) / max(float(disp_q05[e, c]), 1e-4)
                # Synthetic flat or single-spike decoy tails have disp_deficit -> 1.0
                d_val = cov_dist + 10.0 * (disp_deficit ** 2)
                m_dist[i, s, e] = d_val
                excess = max(0.0, d_val - float(thresholds[e, c]))
                m_trust[i, s, e] = math.exp(-0.65 * excess)

    hybrid_trust = tail_trust * m_trust
    return {
        "mahalanobis_distance": m_dist,
        "mahalanobis_trust": m_trust,
        "tail_entropy_trust": tail_trust,
        "hybrid_trust": hybrid_trust,
    }


def sanitize_specialist_posteriors(posteriors: np.ndarray, detector: dict[str, Any],
                                   hybrid_threshold: float = 0.15) -> dict[str, Any]:
    """Byzantine-Resilient Specialist Sanitizer (Yin et al. 2018; Lee et al. 2018).

    Detects corrupted specialists (including both flat-tail and stealth tail-mimicking
    spoofs) via Hybrid Spectral-Mahalanobis Trust and replaces flagged specialists with
    the trust-weighted consensus of surviving specialists on that source (or uniform 0.1
    when all 3 specialists on a source are compromised, triggering safe abstention).
    """
    p = np.asarray(posteriors, dtype=np.float64)
    scores = score_mahalanobis_forensics(p, detector)
    hybrid = scores["hybrid_trust"]
    flagged = hybrid < float(hybrid_threshold)
    sanitized = p.copy()
    b = p.shape[0]
    all_compromised = np.zeros((b, 3), dtype=bool)

    for i in range(b):
        for s in range(3):
            valid = np.where(~flagged[i, s])[0]
            if valid.size == 0:
                # Total sensor compromise on source s -> collapse to uniform 0.1 so receiver abstains
                sanitized[i, s, :, :] = 0.1
                all_compromised[i, s] = True
            elif valid.size < 3:
                weights = hybrid[i, s, valid]
                weights = weights / weights.sum()
                clean_consensus = (weights[:, None] * p[i, s, valid, :]).sum(axis=0)
                for e in range(3):
                    if flagged[i, s, e]:
                        sanitized[i, s, e, :] = clean_consensus

    return {
        "sanitized_posteriors": sanitized,
        "flagged_specialists": flagged,
        "all_compromised": all_compromised,
        "mahalanobis_distance": scores["mahalanobis_distance"],
        "hybrid_trust": hybrid,
        "tail_entropy_trust": scores["tail_entropy_trust"],
    }


def weighted_conformal_prediction_sets(
    cal_scores: np.ndarray,
    cal_shift_scores: np.ndarray,
    test_logits: np.ndarray,
    test_shift_scores: np.ndarray,
    temperature: float = 1.0,
    alpha: float = 0.05,
    beta: float = 6.0,
    temp_shift_gain: float = 8.0,
    k_reg: int = 1,
    lam_reg: float = 0.02,
    epistemic_trust: np.ndarray | None = None,
    epistemic_threshold: float = 0.10,
) -> dict[str, Any]:
    """Covariate-Shift Likelihood-Ratio Weighted Conformal Prediction (Tibshirani et al., NeurIPS 2019).

    Reweights calibration nonconformity scores V_i by the exponential specialist-conflict
    likelihood ratio w(x) = exp(beta * d_shift(x)) and computes the test-conditional
    weighted quantile q_WCP(x_test) using normalized weights:
        p_i(x_test) = w_i / (sum_j w_j + w(x_test)),
        p_inf(x_test) = w(x_test) / (sum_j w_j + w(x_test)).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    v_cal = np.asarray(cal_scores, dtype=np.float64).reshape(-1)
    d_cal = np.asarray(cal_shift_scores, dtype=np.float64).reshape(-1)
    logits = np.asarray(test_logits, dtype=np.float64)
    d_test = np.asarray(test_shift_scores, dtype=np.float64).reshape(-1)
    if v_cal.shape[0] != d_cal.shape[0] or v_cal.shape[0] == 0:
        raise ValueError("cal_scores and cal_shift_scores must be non-empty and match in length")
    if logits.ndim != 2 or logits.shape[0] != d_test.shape[0]:
        raise ValueError("test_logits and test_shift_scores must match in batch dimension")

    order = np.argsort(v_cal)
    v_sorted = v_cal[order]
    d_sorted = d_cal[order]
    w_cal = np.exp(np.clip(beta * d_sorted, -20.0, 20.0))
    cum_w_cal = np.cumsum(w_cal)
    sum_w_cal = float(cum_w_cal[-1])
    cal_q95 = float(np.quantile(d_cal, 0.95))

    n_test, num_classes = logits.shape
    w_test = np.exp(np.clip(beta * d_test, -20.0, 20.0))

    abstain = np.zeros(n_test, dtype=bool)
    if epistemic_trust is not None:
        trust = np.asarray(epistemic_trust, dtype=np.float64).reshape(-1)
        abstain = trust < float(epistemic_threshold)

    penalties = float(lam_reg) * np.maximum(0, np.arange(1, num_classes + 1) - int(k_reg))
    q_wcp = np.zeros(n_test, dtype=np.float64)
    t_wcp = np.zeros(n_test, dtype=np.float64)
    prediction_sets: list[list[int]] = []
    set_sizes = np.zeros(n_test, dtype=np.int64)

    for i in range(n_test):
        denom = sum_w_cal + float(w_test[i])
        idx = int(np.searchsorted(cum_w_cal, (1.0 - alpha) * denom, side="left"))
        excess_shift = max(0.0, float(d_test[i]) - cal_q95)
        if idx < len(v_sorted):
            q_base = float(v_sorted[idx])
        else:
            q_base = float(v_sorted[-1])
        q_i = q_base * (1.0 + 3.5 * excess_shift)
        t_i = max(float(temperature), 1e-3) * (1.0 + float(temp_shift_gain) * excess_shift)
        q_wcp[i] = q_i
        t_wcp[i] = t_i

        if abstain[i]:
            prediction_sets.append([])
            set_sizes[i] = 0
            continue

        row_logits = logits[i] / t_i
        row_logits = row_logits - row_logits.max()
        p = np.exp(row_logits)
        p = p / p.sum()
        cls_order = np.argsort(-p)
        p_sorted = p[cls_order]
        cum_before = np.cumsum(p_sorted) - p_sorted
        min_scores = cum_before + penalties
        included = cls_order[min_scores <= q_i].tolist()
        if not included:
            included = [int(cls_order[0])]
        prediction_sets.append([int(c) for c in included])
        set_sizes[i] = len(included)

    return {
        "prediction_sets": prediction_sets,
        "set_sizes": set_sizes,
        "abstain": abstain,
        "weighted_q_hat": q_wcp,
        "effective_temperatures": t_wcp,
    }


def evaluate_weighted_conformal_sets(
    cal_scores: np.ndarray,
    cal_shift_scores: np.ndarray,
    test_logits: np.ndarray,
    test_targets: np.ndarray,
    test_shift_scores: np.ndarray,
    temperature: float = 1.0,
    alpha: float = 0.05,
    beta: float = 6.0,
    temp_shift_gain: float = 8.0,
    k_reg: int = 1,
    lam_reg: float = 0.02,
    epistemic_trust: np.ndarray | None = None,
    epistemic_threshold: float = 0.10,
) -> dict[str, float]:
    """Evaluate Covariate-Shift Weighted Conformal Prediction (WCP) sets on a labeled cohort."""
    y = np.asarray(test_targets, dtype=np.int64).reshape(-1)
    res = weighted_conformal_prediction_sets(
        cal_scores=cal_scores,
        cal_shift_scores=cal_shift_scores,
        test_logits=test_logits,
        test_shift_scores=test_shift_scores,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        temp_shift_gain=temp_shift_gain,
        k_reg=k_reg,
        lam_reg=lam_reg,
        epistemic_trust=epistemic_trust,
        epistemic_threshold=epistemic_threshold,
    )
    sets = res["prediction_sets"]
    sizes = res["set_sizes"]
    abstain = res["abstain"]
    covered = np.array([int(y[i]) in sets[i] for i in range(len(y))], dtype=bool)
    accepted = ~abstain
    accepted_count = int(accepted.sum())
    selective_cov = float(covered[accepted].mean()) if accepted_count > 0 else 1.0
    return {
        "wcp_marginal_coverage": round(float(covered.mean()), 5),
        "wcp_selective_coverage": round(selective_cov, 5),
        "wcp_mean_set_size": round(float(sizes.mean()), 5),
        "wcp_mean_q_hat": round(float(res["weighted_q_hat"].mean()), 5),
    }


def helmholtz_free_energy_forensics(
    posteriors: np.ndarray,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute Helmholtz Free-Energy OOD & Runner-Up Spectral Concentration (Liu et al., NeurIPS 2020).

    Evaluates the runner-up log-posterior spectrum log P_{s,e,(2..10)} to compute:
      - tail_free_energy: E_tail(P_{s,e}; T) = -T * log sum_{r=2..10} exp(log P_{s,e,(r)} / T)
      - spectral_participation_ratio: Effective number of active runner-up classes
        N_eff = 1 / sum_{r=2..10} q_r^2 in [1, 9] (where q_r is normalized runner-up tail).
    """
    p = np.asarray(posteriors, dtype=np.float64)
    if p.ndim != 4 or p.shape[1:] != (3, 3, 10):
        raise ValueError("posteriors must have shape [batch, 3, 3, 10]")
    t = max(float(temperature), 1e-3)
    b = p.shape[0]
    tail_free_energy = np.zeros((b, 3, 3), dtype=np.float64)
    participation_ratio = np.zeros((b, 3, 3), dtype=np.float64)

    for i in range(b):
        for s in range(3):
            for e in range(3):
                vec = np.clip(p[i, s, e], 1e-8, 1.0)
                c_max = int(np.argmax(vec))
                mask = [idx for idx in range(10) if idx != c_max]
                tail = vec[mask]
                log_tail = np.log(tail) / t
                m = float(log_tail.max())
                lse = m + math.log(float(np.exp(log_tail - m).sum()))
                tail_free_energy[i, s, e] = -t * lse

                q = tail / tail.sum()
                ipr = float((q ** 2).sum())
                participation_ratio[i, s, e] = 1.0 / max(ipr, 1e-8)

    return {
        "tail_free_energy": tail_free_energy,
        "participation_ratio": participation_ratio,
    }


def conformal_risk_control(
    cal_probs: np.ndarray,
    cal_targets: np.ndarray,
    alpha: float = 0.05,
    k_reg: int = 1,
    lam_reg: float = 0.02,
    rank_weight: float = 0.35,
    bound_b: float = 1.0,
) -> dict[str, Any]:
    """Fit Conformal Risk Control threshold lambda_CRC for monotone bounded loss (Angelopoulos et al., ICLR 2024).

    Given calibration probabilities [n, K] and targets [n], evaluates the monotone non-increasing
    rank-weighted miscoverage loss L_i(lambda) in [0, B]:
        L_i(lambda) = 1{y_i not in C_lambda(x_i)} * ((1 - w_r) + w_r * rank(y_i) / (K - 1))
    and selects:
        lambda_CRC = inf { lambda : (sum_{i=1}^n L_i(lambda) + B) / (n + 1) <= alpha }.
    Guarantees finite-sample expected risk E[L_{n+1}(lambda_CRC)] <= alpha under exchangeability.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    probs = np.asarray(cal_probs, dtype=np.float64)
    y = np.asarray(cal_targets, dtype=np.int64).reshape(-1)
    n, k = probs.shape
    if n == 0 or y.shape[0] != n:
        raise ValueError("cal_probs and cal_targets must be non-empty and match in length")

    # Entry threshold v_i is the exact RAPS score at which true label y_i enters C_lambda(x_i)
    v_entry = raps_nonconformity_scores(probs, y, k_reg=k_reg, lam_reg=lam_reg)
    order = np.argsort(-probs, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1).astype(np.float64)

    w_r = float(np.clip(rank_weight, 0.0, 1.0))
    miss_severity = (1.0 - w_r) + w_r * (ranks / max(1, k - 1))
    miss_severity = np.clip(miss_severity, 0.0, float(bound_b))

    # Candidate thresholds are sorted v_entry values; L_i(lambda) = miss_severity[i] if lambda < v_entry[i] else 0
    sorted_idx = np.argsort(v_entry)
    v_sorted = v_entry[sorted_idx]
    sev_sorted = miss_severity[sorted_idx]
    # Suffix sums of severity for items not yet covered when lambda = v_sorted[j]
    total_sev = float(sev_sorted.sum())
    prefix_sev = np.cumsum(sev_sorted)
    remaining_sev = total_sev - prefix_sev
    crc_objective = (remaining_sev + float(bound_b)) / float(n + 1)

    valid = np.where(crc_objective <= float(alpha))[0]
    if valid.size > 0:
        lambda_crc = float(v_sorted[int(valid[0])])
        emp_risk = float(remaining_sev[int(valid[0])] / float(n))
        bound_risk = float(crc_objective[int(valid[0])])
    else:
        lambda_crc = float(v_sorted[-1])
        emp_risk = 0.0
        bound_risk = float(bound_b) / float(n + 1)

    return {
        "lambda_crc": lambda_crc,
        "alpha": float(alpha),
        "empirical_cal_risk": emp_risk,
        "certified_upper_bound": bound_risk,
        "rank_weight": w_r,
    }


def evaluate_conformal_risk_control(
    probs: np.ndarray,
    targets: np.ndarray,
    lambda_crc: float,
    k_reg: int = 1,
    lam_reg: float = 0.02,
    rank_weight: float = 0.35,
    epistemic_trust: np.ndarray | None = None,
    epistemic_threshold: float = 0.10,
) -> dict[str, Any]:
    """Evaluate Conformal Risk Control (CRC) prediction sets and monotone risk on a test cohort."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    n, k = p.shape
    sets_info = raps_prediction_sets(
        p, q_hat=float(lambda_crc), k_reg=k_reg, lam_reg=lam_reg,
        epistemic_trust=epistemic_trust, epistemic_threshold=epistemic_threshold,
    )
    sets = sets_info["prediction_sets"]
    order = np.argsort(-p, axis=1)
    w_r = float(np.clip(rank_weight, 0.0, 1.0))

    abstained = np.array([len(s) == 0 for s in sets], dtype=bool)
    covered = np.array([(not abstained[i]) and (int(y[i]) in sets[i]) for i in range(n)], dtype=bool)
    ranks = np.argmax(order == y[:, None], axis=1).astype(np.float64)
    miss_losses = (1.0 - w_r) + w_r * (ranks / max(1, k - 1))
    losses = np.where(abstained | covered, 0.0, miss_losses)

    accepted = ~abstained
    return {
        "crc_lambda": round(float(lambda_crc), 5),
        "crc_expected_loss": round(float(losses.mean()), 5),
        "crc_accepted_loss": round(float(losses[accepted].mean()), 5) if accepted.any() else 0.0,
        "crc_marginal_coverage": round(float(covered.mean()), 5),
        "crc_mean_set_size": round(float(sets_info["mean_set_size"]), 5),
        "prediction_sets": sets,
    }


def selective_risk_coverage_profile(
    probs: np.ndarray,
    targets: np.ndarray,
    epistemic_trust: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Selective Classification Risk-Coverage Curve, AURC & Excess AURC (Geifman & El-Yaniv, NeurIPS 2017).

    Ranks predictions by hybrid selective confidence s(x) = p_max(x) * tau(x) and computes:
      - aurc: Area Under the Risk-Coverage Curve in [0, 1] (lower is better).
      - oracle_aurc: Bayes-optimal lower bound AURC* given the classifier's base accuracy.
      - e_aurc: Excess AURC = AURC - AURC* >= 0.
      - risk_at_90_cov / risk_at_80_cov: Selective error rates at 90% and 80% coverage.
      - curve_coverages / curve_risks: 10-point risk-coverage trajectory for plotting.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = p.shape[0]
    if n == 0 or y.shape[0] != n:
        raise ValueError("probs and targets must be non-empty and match in length")

    preds = p.argmax(axis=1)
    p_max = p.max(axis=1)
    if epistemic_trust is not None:
        tau = np.clip(np.asarray(epistemic_trust, dtype=np.float64).reshape(-1), 0.0, 1.0)
        score = p_max * (0.25 + 0.75 * tau)
    else:
        score = p_max

    errors = (preds != y).astype(np.float64)
    # Sort descending by confidence score (tie-break by p_max descending)
    order = np.lexsort((-p_max, -score))
    sorted_errors = errors[order]

    cum_errors = np.cumsum(sorted_errors)
    counts = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_errors / counts
    aurc = float(risks.mean())

    # Oracle ranks all correct predictions (error=0) before all incorrect predictions (error=1)
    oracle_errors = np.sort(errors)
    oracle_risks = np.cumsum(oracle_errors) / counts
    oracle_aurc = float(oracle_risks.mean())
    e_aurc = max(0.0, aurc - oracle_aurc)

    idx_90 = max(0, min(n - 1, int(round(0.90 * n)) - 1))
    idx_80 = max(0, min(n - 1, int(round(0.80 * n)) - 1))

    grid_covs = [round(0.1 * k, 2) for k in range(1, 11)]
    grid_risks = []
    for c_val in grid_covs:
        idx_c = max(0, min(n - 1, int(round(c_val * n)) - 1))
        grid_risks.append(round(float(risks[idx_c]), 5))

    return {
        "aurc": round(aurc, 5),
        "oracle_aurc": round(oracle_aurc, 5),
        "e_aurc": round(e_aurc, 5),
        "risk_at_90_cov": round(float(risks[idx_90]), 5),
        "risk_at_80_cov": round(float(risks[idx_80]), 5),
        "curve_coverages": grid_covs,
        "curve_risks": grid_risks,
    }
