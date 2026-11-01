"""A learned modular receiver and matched four-view training objectives.

No task answer, correct source selector or digit transformation is implemented in
inference. Query-conditioned routing, reliability weights, task matrices and the
context shortcut are all trained parameters. Donor probabilities are a frozen
boundary: receiver gradients never flow back into a donor network.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("passive", "dropout", "iit", "effect_only", "invariance_only", "tether", "unpaired")
_VIEW_NAMES = ("donors", "relevant", "irrelevant", "damaged")


def _probabilities(value, name, shape, n=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
        raise ValueError(f"{name} must have shape [batch,{','.join(map(str, shape))}]")
    if value.shape[0] < 1 or (n is not None and value.shape[0] != n):
        raise ValueError(f"{name} must have the same nonempty batch size")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} probabilities must lie in [0,1]")
    if not torch.allclose(value.sum(-1), torch.ones_like(value[..., 0]), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} probabilities must sum to one")


def _inputs(donors, query, context):
    _probabilities(donors, "donors", (3, 3, 10))
    n = donors.shape[0]
    if not isinstance(query, torch.Tensor) or query.dtype != torch.long:
        raise TypeError("query must be a long tensor")
    if query.shape != (n, 2):
        raise ValueError("query must have shape [batch,2]")
    if (query < 0).any() or (query > 2).any():
        raise ValueError("query source and operation indices must lie in [0,2]")
    _probabilities(context, "context", (10,), n)
    if donors.device != query.device or donors.device != context.device:
        raise ValueError("donors, query and context must be on the same device")


def _target(value, name, n, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise TypeError(f"{name} must be a long tensor")
    if value.shape != (n,) or value.device != device:
        raise ValueError(f"{name} must match the batch size and device")
    if (value < 0).any() or (value > 9).any():
        raise ValueError(f"{name} must contain class indices in [0,9]")


class TetherReceiver(nn.Module):
    """Small receiver for three sources, three experts and ten digit classes.

    In v0.1 (`forensic=False`), reliability uses each expert's full posterior,
    normalized entropy, agreement with the local ensemble, and distance from the
    local ensemble. In v0.2 (`forensic=True`), reliability replaces cross-expert
    majority consensus with intrinsic dark-knowledge features (top-1/top-2 margin,
    normalized runner-up tail entropy and runner-up tail dispersion) so majority
    specialist corruption cannot invert reliability weights.
    """

    def __init__(self, hidden=24, *, forensic=False, epistemic_threshold=0.10):
        super().__init__()
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 4:
            raise ValueError("hidden must be an integer of at least four")
        if not isinstance(forensic, bool):
            raise TypeError("forensic must be a boolean")
        if not isinstance(epistemic_threshold, (int, float)) or isinstance(epistemic_threshold, bool) or not math.isfinite(epistemic_threshold) or epistemic_threshold < 0:
            raise ValueError("epistemic_threshold must be a finite non-negative number")
        self.forensic = forensic
        self.epistemic_threshold = float(epistemic_threshold)
        self.config = {"hidden": hidden, "forensic": True, "epistemic_threshold": self.epistemic_threshold} if forensic else {"hidden": hidden}
        self.source_embedding = nn.Embedding(3, hidden)
        self.operation_embedding = nn.Embedding(3, hidden)
        self.source_router = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3)
        )
        self.expert_embedding = nn.Embedding(3, 6)
        reliability_in = 20 if forensic else 19
        self.reliability = nn.Sequential(
            nn.Linear(reliability_in, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.task_matrices = nn.Parameter(torch.randn(3, 10, 10) * 0.08)
        self.task_bias = nn.Parameter(torch.zeros(3, 10))
        self.shortcut = nn.Sequential(
            nn.Linear(10, hidden), nn.Tanh(), nn.Linear(hidden, 10)
        )

    def _compute(self, donors, query, context, detach_donors: bool = True):
        # The specialist bank is frozen even if a caller supplies a live donor
        # computation graph rather than the normal cached posterior tensors.
        dtype = self.task_matrices.dtype
        p = donors.detach().to(dtype=dtype) if detach_donors else donors.to(dtype=dtype)
        context = context.to(dtype=dtype)
        consensus = p.mean(2, keepdim=True)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1, keepdim=True) / math.log(10)
        agreement = (p * consensus).sum(-1, keepdim=True)
        deviation = (p - consensus).square().sum(-1, keepdim=True)
        top2 = p.topk(2, dim=-1).values
        margin = top2[..., :1] - top2[..., 1:2]
        tail = p.topk(9, dim=-1, largest=False).values
        tail_norm = tail / tail.sum(-1, keepdim=True).clamp_min(1e-8)
        tail_entropy = -(tail_norm * tail_norm.clamp_min(1e-8).log()).sum(-1, keepdim=True) / math.log(9)
        tail_std = tail.std(-1, keepdim=True) * 10.0
        identities = self.expert_embedding.weight[None, None].expand(p.shape[0], 3, -1, -1)
        if self.forensic:
            features = torch.cat((p, entropy, margin, tail_entropy, tail_std, identities), -1)
        else:
            features = torch.cat((p, entropy, agreement, deviation, identities), -1)
        expert_weights = self.reliability(features).squeeze(-1).softmax(-1)
        source_posteriors = (expert_weights[..., None] * p).sum(2)
        query_features = torch.cat((self.source_embedding(query[:, 0]),
                                    self.operation_embedding(query[:, 1])), -1)
        source_weights = self.source_router(query_features).softmax(-1)
        pooled = (source_weights[..., None] * source_posteriors).sum(1)
        transformed = torch.einsum("bi,bij->bj", pooled, self.task_matrices[query[:, 1]])
        transformed = transformed + self.task_bias[query[:, 1]]
        shortcut = self.shortcut(context)
        logits = transformed + shortcut
        trust_scores = (1.0 - tail_entropy.squeeze(-1)).clamp(0.0, 1.0)
        rows = torch.arange(p.shape[0], device=p.device)
        queried_sources = query[:, 0]
        queried_trust = trust_scores[rows, queried_sources].max(dim=-1).values
        epistemic_abstain = (queried_trust < self.epistemic_threshold) if self.forensic else torch.zeros_like(queried_trust, dtype=torch.bool)
        predictions = logits.argmax(-1)
        selective_predictions = torch.where(epistemic_abstain, torch.full_like(predictions, -1), predictions)
        consensus_sq = consensus.squeeze(2)
        consensus_entropy = -(consensus_sq * consensus_sq.clamp_min(1e-8).log()).sum(-1) / math.log(10)
        mean_expert_entropy = entropy.squeeze(-1).mean(-1)
        conflict_jsd = (consensus_entropy - mean_expert_entropy).clamp(0.0, 1.0)
        aleatoric = (expert_weights * entropy.squeeze(-1)).sum(-1).clamp(0.0, 1.0)
        queried_aleatoric = aleatoric[rows, queried_sources]
        queried_conflict = conflict_jsd[rows, queried_sources]
        forensic_spoof = (1.0 - queried_trust).clamp(0.0, 1.0)
        return {
            "logits": logits,
            "source_weights": source_weights,
            "expert_weights": expert_weights,
            "source_posteriors": source_posteriors,
            "pooled_posterior": pooled,
            "expert_entropy": entropy.squeeze(-1),
            "expert_agreement": agreement.squeeze(-1),
            "expert_margin": margin.squeeze(-1),
            "tail_entropy": tail_entropy.squeeze(-1),
            "trust_score": trust_scores,
            "epistemic_trust": queried_trust,
            "aleatoric_uncertainty": aleatoric,
            "queried_aleatoric": queried_aleatoric,
            "epistemic_conflict_jsd": conflict_jsd,
            "queried_conflict_jsd": queried_conflict,
            "forensic_spoof_risk": forensic_spoof,
            "abstain": epistemic_abstain,
            "selective_prediction": selective_predictions,
            "task_logits": transformed,
            "shortcut_logits": shortcut,
        }

    def inspect(self, donors, query, context):
        """Return neural output and learned routes; this does not change state."""
        _inputs(donors, query, context)
        if donors.device != self.task_matrices.device:
            raise ValueError("input tensors and receiver must be on the same device")
        return self._compute(donors, query, context)

    def attribute(self, donors, query, context, target=None):
        """Compute input-gradient causal attribution of predictions to donor posteriors."""
        _inputs(donors, query, context)
        if donors.device != self.task_matrices.device:
            raise ValueError("input tensors and receiver must be on the same device")
        dtype = self.task_matrices.dtype
        p = donors.clone().detach().to(dtype=dtype).requires_grad_(True)
        detail = self._compute(p, query, context, detach_donors=False)
        logits = detail["logits"]
        if target is None:
            chosen_target = logits.argmax(dim=-1)
        else:
            _target(target, "target", donors.shape[0], donors.device)
            chosen_target = target
        target_score = logits.gather(1, chosen_target.unsqueeze(-1)).sum()
        grad_out = torch.autograd.grad(target_score, p, allow_unused=True, create_graph=False)[0]
        grads = torch.zeros_like(p) if grad_out is None else grad_out
        saliency = (p.detach() * grads.detach()).abs()
        source_saliency = saliency.sum(dim=(-1, -2))
        expert_saliency = saliency.sum(dim=-1)
        return {
            "saliency": saliency,
            "source_saliency": source_saliency,
            "expert_saliency": expert_saliency,
            "gradients": grads.detach(),
            "logits": logits.detach(),
            "target": chosen_target,
        }

    def integrated_gradients(self, donors, query, context, target=None, steps: int = 128):
        """Axiomatic Integrated Gradients (Sundararajan et al., 2017) from uniform donor baseline."""
        _inputs(donors, query, context)
        if donors.device != self.task_matrices.device:
            raise ValueError("input tensors and receiver must be on the same device")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2 or steps > 256:
            raise ValueError("steps must be an integer in [2, 256]")
        dtype = self.task_matrices.dtype
        n = donors.shape[0]
        p = donors.detach().to(dtype=dtype)
        baseline = torch.full_like(p, 0.1)
        with torch.no_grad():
            full_detail = self._compute(p, query, context)
            base_detail = self._compute(baseline, query, context)
            logits = full_detail["logits"]
            base_logits = base_detail["logits"]
        if target is None:
            chosen_target = logits.argmax(dim=-1)
        else:
            _target(target, "target", n, donors.device)
            chosen_target = target
        # Quartic boundary-layer quadrature in u = 1 - alpha avoids float32 cancellation near alpha -> 1
        t_edges = torch.linspace(0.0, 1.0, steps + 1, device=p.device, dtype=torch.float64)
        u_edges = (1.0 - t_edges).pow(4)
        u_mid = (0.5 * (u_edges[:-1] + u_edges[1:])).to(dtype=dtype)
        weights = (u_edges[:-1] - u_edges[1:]).to(dtype=dtype)
        delta_p = p - baseline
        # Positive convex combination (1 - u)*p + u*baseline preserves full float32 tail precision
        interp = ((1.0 - u_mid)[:, None, None, None, None] * p.unsqueeze(0)
                  + u_mid[:, None, None, None, None] * baseline.unsqueeze(0)).reshape(steps * n, 3, 3, 10)
        interp.requires_grad_(True)
        q_rep = query.unsqueeze(0).expand(steps, -1, -1).reshape(steps * n, 2)
        c_rep = context.to(dtype=dtype).unsqueeze(0).expand(steps, -1, -1).reshape(steps * n, 10)
        t_rep = chosen_target.unsqueeze(0).expand(steps, -1).reshape(steps * n)
        interp_logits = self._compute(interp, q_rep, c_rep, detach_donors=False)["logits"]
        score = interp_logits.gather(1, t_rep.unsqueeze(-1)).sum()
        grad_out = torch.autograd.grad(score, interp, allow_unused=True, create_graph=False)[0]
        grads = torch.zeros_like(interp) if grad_out is None else grad_out
        avg_grads = (grads.detach().reshape(steps, n, 3, 3, 10) * weights[:, None, None, None, None]).sum(dim=0)
        attributions = delta_p * avg_grads
        saliency = attributions.abs()
        source_saliency = saliency.sum(dim=(-1, -2))
        expert_saliency = saliency.sum(dim=-1)
        source_net = attributions.sum(dim=(-1, -2))
        expert_net = attributions.sum(dim=-1)
        rows = torch.arange(n, device=p.device)
        target_delta = logits[rows, chosen_target] - base_logits[rows, chosen_target]
        attribution_sum = attributions.sum(dim=(-1, -2, -3))
        completeness_error = (attribution_sum - target_delta).abs()
        return {
            "attributions": attributions,
            "saliency": saliency,
            "source_saliency": source_saliency,
            "expert_saliency": expert_saliency,
            "source_net_contribution": source_net,
            "expert_net_contribution": expert_net,
            "baseline_logits": base_logits,
            "logits": logits,
            "target": chosen_target,
            "target_delta": target_delta,
            "attribution_sum": attribution_sum,
            "completeness_error": completeness_error,
        }

    @torch.no_grad()
    def interchange_intervention(self, base_donors, base_query, base_context,
                                 cf_donors, cf_query=None, cf_context=None, target=None):
        """Causal Abstraction Interchange Intervention (activation patching) across internal nodes."""
        _inputs(base_donors, base_query, base_context)
        if cf_query is None:
            cf_query = base_query
        if cf_context is None:
            cf_context = base_context
        _inputs(cf_donors, cf_query, cf_context)
        if base_donors.shape[0] != cf_donors.shape[0]:
            raise ValueError("base and counterfactual inputs must have the same batch size")
        if base_donors.device != self.task_matrices.device or cf_donors.device != self.task_matrices.device:
            raise ValueError("input tensors and receiver must be on the same device")
        n = base_donors.shape[0]
        dtype = self.task_matrices.dtype
        p_base = base_donors.detach().to(dtype=dtype)
        base = self._compute(p_base, base_query, base_context)
        cf = self._compute(cf_donors, cf_query, cf_context)
        if target is None:
            chosen_target = cf["logits"].argmax(dim=-1)
        else:
            _target(target, "target", n, base_donors.device)
            chosen_target = target
        rows = torch.arange(n, device=base_donors.device)
        base_prob = base["logits"].softmax(-1)
        cf_prob = cf["logits"].softmax(-1)
        base_target_logit = base["logits"][rows, chosen_target]
        cf_target_logit = cf["logits"][rows, chosen_target]
        logit_gap = cf_target_logit - base_target_logit
        denom = torch.where(logit_gap >= 0, logit_gap.clamp_min(1e-5), logit_gap.clamp_max(-1e-5))

        nodes = ("expert_weights", "source_posteriors", "source_weights",
                 "pooled_posterior", "task_logits", "shortcut_logits")
        patches = {}
        for node in nodes:
            ew = cf["expert_weights"] if node == "expert_weights" else base["expert_weights"]
            if node == "source_posteriors":
                sp = cf["source_posteriors"]
            elif node == "expert_weights":
                sp = (ew[..., None] * p_base).sum(2)
            else:
                sp = base["source_posteriors"]
            sw = cf["source_weights"] if node == "source_weights" else base["source_weights"]
            if node == "pooled_posterior":
                pooled = cf["pooled_posterior"]
            elif node in ("expert_weights", "source_posteriors", "source_weights"):
                pooled = (sw[..., None] * sp).sum(1)
            else:
                pooled = base["pooled_posterior"]
            if node == "task_logits":
                task_l = cf["task_logits"]
            elif node in ("expert_weights", "source_posteriors", "source_weights", "pooled_posterior"):
                task_l = torch.einsum("bi,bij->bj", pooled, self.task_matrices[base_query[:, 1]]) + self.task_bias[base_query[:, 1]]
            else:
                task_l = base["task_logits"]
            short_l = cf["shortcut_logits"] if node == "shortcut_logits" else base["shortcut_logits"]
            patched_logits = task_l + short_l
            patched_prob = patched_logits.softmax(-1)
            patched_target_logit = patched_logits[rows, chosen_target]
            restoration = ((patched_target_logit - base_target_logit) / denom).clamp(-1.0, 2.0)
            prob_shift = patched_prob[rows, chosen_target] - base_prob[rows, chosen_target]
            kl_to_cf = (cf_prob * (cf_prob.clamp_min(1e-8).log() - patched_prob.clamp_min(1e-8).log())).sum(-1).clamp_min(0.0)
            patches[node] = {
                "logits": patched_logits,
                "probabilities": patched_prob,
                "prediction": patched_logits.argmax(-1),
                "restoration": restoration,
                "prob_shift": prob_shift,
                "kl_to_cf": kl_to_cf,
            }
        return {
            "target": chosen_target,
            "base_logits": base["logits"],
            "cf_logits": cf["logits"],
            "base_prediction": base["logits"].argmax(-1),
            "cf_prediction": cf["logits"].argmax(-1),
            "patches": patches,
        }

    @torch.no_grad()
    def shapley_interactions(self, donors, query, context, target=None):
        """Exact Cooperative Game-Theoretic Specialist Shapley Values & Shapley-Taylor Interactions.

        Evaluates all 2^3 = 8 specialist coalitions on the queried source relative to
        the maximum-entropy uniform baseline (0.1) in a single batched pass, returning:
        - Exact Shapley values phi_e (Lundberg & Lee, NeurIPS 2017)
        - Exact Harsanyi / Mobius coalition dividends (d_0, d_1, d_2, d_01, d_02, d_12, d_012)
        - Exact Order-2 Shapley-Taylor Interaction Matrix I^(2)_{i,j} (Sundararajan et al., ICML 2020)
        where sum_{i,j} I^(2)_{i,j} == sum_e phi_e == v({0,1,2}) - v(empty) exactly.
        """
        _inputs(donors, query, context)
        if donors.device != self.task_matrices.device:
            raise ValueError("input tensors and receiver must be on the same device")
        n = donors.shape[0]
        dtype = self.task_matrices.dtype
        p = donors.detach().to(dtype=dtype)
        full_detail = self._compute(p, query, context)
        logits = full_detail["logits"]
        if target is None:
            chosen_target = logits.argmax(dim=-1)
        else:
            _target(target, "target", n, donors.device)
            chosen_target = target

        uniform = torch.full_like(p, 0.1)
        rows = torch.arange(n, device=p.device)
        q_src = query[:, 0]

        # Build all 8 coalitions [8, n, 3, 3, 10] where absent specialists on queried source are 0.1
        coalition_inputs = p.unsqueeze(0).expand(8, -1, -1, -1, -1).clone()
        for mask in range(8):
            for e in range(3):
                if not ((mask >> e) & 1):
                    coalition_inputs[mask, rows, q_src, e] = uniform[rows, q_src, e]

        flat_inputs = coalition_inputs.reshape(8 * n, 3, 3, 10)
        q_rep = query.unsqueeze(0).expand(8, -1, -1).reshape(8 * n, 2)
        c_rep = context.to(dtype=dtype).unsqueeze(0).expand(8, -1, -1).reshape(8 * n, 10)
        flat_logits = self._compute(flat_inputs, q_rep, c_rep)["logits"].reshape(8, n, 10)
        # v[mask, b] is the target class logit under coalition `mask`
        v = flat_logits[:, rows, chosen_target]
        v0, v1, v2, v3, v4, v5, v6, v7 = v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]

        phi_0 = (v1 - v0) / 3.0 + (v3 - v2) / 6.0 + (v5 - v4) / 6.0 + (v7 - v6) / 3.0
        phi_1 = (v2 - v0) / 3.0 + (v3 - v1) / 6.0 + (v6 - v4) / 6.0 + (v7 - v5) / 3.0
        phi_2 = (v4 - v0) / 3.0 + (v5 - v1) / 6.0 + (v6 - v2) / 6.0 + (v7 - v3) / 3.0
        shapley_values = torch.stack((phi_0, phi_1, phi_2), dim=-1)

        # Harsanyi / Mobius coalition dividends
        d0 = v1 - v0
        d1 = v2 - v0
        d2 = v4 - v0
        d01 = v3 - v1 - v2 + v0
        d02 = v5 - v1 - v4 + v0
        d12 = v6 - v2 - v4 + v0
        d012 = v7 - v3 - v5 - v6 + v1 + v2 + v4 - v0

        # Order-2 Shapley-Taylor symmetric matrix [n, 3, 3] whose sum equals v7 - v0
        st_matrix = torch.zeros((n, 3, 3), dtype=dtype, device=p.device)
        st_matrix[:, 0, 0] = d0
        st_matrix[:, 1, 1] = d1
        st_matrix[:, 2, 2] = d2
        pair_01 = 0.5 * (d01 + d012 / 3.0)
        pair_02 = 0.5 * (d02 + d012 / 3.0)
        pair_12 = 0.5 * (d12 + d012 / 3.0)
        st_matrix[:, 0, 1] = pair_01
        st_matrix[:, 1, 0] = pair_01
        st_matrix[:, 0, 2] = pair_02
        st_matrix[:, 2, 0] = pair_02
        st_matrix[:, 1, 2] = pair_12
        st_matrix[:, 2, 1] = pair_12

        total_gain = v7 - v0
        efficiency_error = (shapley_values.sum(dim=-1) - total_gain).abs()
        taylor_efficiency_error = (st_matrix.sum(dim=(-1, -2)) - total_gain).abs()

        coalition_labels = ("empty", "E0", "E1", "E0+E1", "E2", "E0+E2", "E1+E2", "E0+E1+E2")
        coalition_payoffs = {coalition_labels[m]: v[m] for m in range(8)}
        return {
            "target": chosen_target,
            "shapley_values": shapley_values,
            "interaction_matrix": st_matrix,
            "mobius_dividends": {
                "d0": d0, "d1": d1, "d2": d2,
                "d01": d01, "d02": d02, "d12": d12,
                "d012": d012,
            },
            "coalition_payoffs": coalition_payoffs,
            "total_gain": total_gain,
            "baseline_logit": v0,
            "full_logit": v7,
            "efficiency_error": efficiency_error,
            "taylor_efficiency_error": taylor_efficiency_error,
        }

    def pid_decomposition(self, donors, query, context, target=None):
        """Compute Williams-Beer (2010) Partial Information Decomposition (PID) across the 3 specialists.

        Decomposes the point-wise directed predictive information gain (in nats)
        toward the target class over the uniform baseline coalition into non-negative
        information atoms:
          - Shared Redundancy (R_shared): Information redundantly available across specialists.
          - Unique Specialist Information (U_0, U_1, U_2): Information exclusively held by
            specialist e beyond any other specialist (crucial under 2/3 corruption).
          - Coalitional Circuit Synergy (S_syn): Emergent information unlocked only when
            specialists are combined through the receiver's non-linear reliability router.
        """
        _inputs(donors, query, context)
        dtype = self.task_matrices.dtype
        batch_size = donors.shape[0]
        device = donors.device
        p = donors.detach().to(dtype=dtype)
        c_tensor = context.detach().to(dtype=dtype)

        with torch.no_grad():
            full_detail = self._compute(p, query, c_tensor, detach_donors=True)
            full_logits = full_detail["logits"]
            if target is None:
                chosen_target = full_logits.argmax(dim=-1)
            else:
                _target(target, "target", batch_size, device)
                chosen_target = target

            uniform = torch.full_like(p, 0.1)
            coalition_tensors = []
            for mask_int in range(8):
                coalition_p = p.clone()
                for e in range(3):
                    if not (mask_int & (1 << e)):
                        for b in range(batch_size):
                            s_idx = int(query[b, 0])
                            coalition_p[b, s_idx, e] = uniform[b, s_idx, e]
                coalition_tensors.append(coalition_p)

            batched_p = torch.cat(coalition_tensors, dim=0)
            batched_q = query.repeat(8, 1)
            batched_c = c_tensor.repeat(8, 1)
            batched_probs = self._compute(batched_p, batched_q, batched_c, detach_donors=True)["logits"].softmax(dim=-1)
            batched_probs = batched_probs.view(8, batch_size, 10)

            rows = torch.arange(batch_size, device=device)
            target_probs = batched_probs[:, rows, chosen_target].clamp_min(1e-8)
            base_prob = target_probs[0]
            info = (target_probs.log() - base_prob.log()).clamp_min(0.0)

            i0 = info[1]
            i1 = info[2]
            i2 = info[4]
            i012 = info[7]
            singletons = torch.stack((i0, i1, i2), dim=-1)  # [B, 3]

            triadic_redundancy = singletons.min(dim=-1).values
            u0 = (i0 - torch.maximum(i1, i2)).clamp_min(0.0)
            u1 = (i1 - torch.maximum(i0, i2)).clamp_min(0.0)
            u2 = (i2 - torch.maximum(i0, i1)).clamp_min(0.0)
            unique_expert = torch.stack((u0, u1, u2), dim=-1)

            max_singleton = singletons.max(dim=-1).values
            pairwise_redundancy = (max_singleton - triadic_redundancy - unique_expert.sum(dim=-1)).clamp_min(0.0)
            shared_redundancy = triadic_redundancy + pairwise_redundancy
            synergy = (i012 - max_singleton).clamp_min(0.0)
            total_info = shared_redundancy + unique_expert.sum(dim=-1) + synergy
            denom = total_info.clamp_min(1e-8)

            norm_r = shared_redundancy / denom
            norm_u = unique_expert / denom.unsqueeze(-1)
            norm_s = synergy / denom
            normalized_atoms = torch.cat((norm_r.unsqueeze(-1), norm_u, norm_s.unsqueeze(-1)), dim=-1)

        return {
            "target": chosen_target,
            "redundancy": shared_redundancy,
            "triadic_redundancy": triadic_redundancy,
            "pairwise_redundancy": pairwise_redundancy,
            "unique_expert": unique_expert,
            "synergy": synergy,
            "total_information": total_info,
            "singleton_information": singletons,
            "full_information": i012,
            "normalized_atoms": normalized_atoms,
        }

    def causal_mediation(self, donors, query, context, target=None):
        """Compute Natural Direct Effect (NDE_visual) and Natural Indirect Effect (NIE_shortcut).

        Follows Pearl (2001) and Vig et al. (NeurIPS 2020, arXiv:2004.12265) to decompose
        the total prediction margin and probability gain into:
          - Visual Natural Direct Effect (NDE_visual): Effect transmitted strictly through
            the visual specialist routing & task transformation pathway under neutral context.
          - Shortcut Natural Indirect Effect (NIE_shortcut): Effect mediated through the
            additive context proxy shortcut branch.
          - Visual Dominance Ratio (rho_visual in [0, 1]): Fraction of pathway drive coming
            from visual evidence vs. context shortcut.
        """
        _inputs(donors, query, context)
        dtype = self.task_matrices.dtype
        batch_size = donors.shape[0]
        device = donors.device
        p = donors.detach().to(dtype=dtype)
        c_tensor = context.detach().to(dtype=dtype)
        p_null = torch.full_like(p, 0.1)
        c_null = torch.full_like(c_tensor, 0.1)

        with torch.no_grad():
            batched_p = torch.cat((p_null, p, p_null, p), dim=0)
            batched_q = query.repeat(4, 1)
            batched_c = torch.cat((c_null, c_null, c_tensor, c_tensor), dim=0)
            out = self._compute(batched_p, batched_q, batched_c, detach_donors=True)
            logits = out["logits"].view(4, batch_size, 10)
            probs = logits.softmax(dim=-1)

            if target is None:
                chosen_target = logits[3].argmax(dim=-1)
            else:
                _target(target, "target", batch_size, device)
                chosen_target = target

            rows = torch.arange(batch_size, device=device)
            target_logits = logits[:, rows, chosen_target]
            masked = logits.clone()
            masked[:, rows, chosen_target] = -1e9
            runner_up_logits = masked.max(dim=-1).values
            margins = target_logits - runner_up_logits  # [4, B]: 00, 10, 01, 11

            nde_margin = margins[1] - margins[0]
            nie_margin = margins[3] - margins[1]
            te_margin = margins[3] - margins[0]
            visual_dominance = nde_margin.abs() / (nde_margin.abs() + nie_margin.abs() + 1e-8)

            target_probs = probs[:, rows, chosen_target]
            nde_prob = target_probs[1] - target_probs[0]
            nie_prob = target_probs[3] - target_probs[1]
            te_prob = target_probs[3] - target_probs[0]

        return {
            "target": chosen_target,
            "nde_margin": nde_margin,
            "nie_margin": nie_margin,
            "total_effect_margin": te_margin,
            "visual_dominance": visual_dominance,
            "nde_prob": nde_prob,
            "nie_prob": nie_prob,
            "total_effect_prob": te_prob,
            "visual_only_prediction": logits[1].argmax(dim=-1),
            "shortcut_only_prediction": logits[2].argmax(dim=-1),
            "full_prediction": logits[3].argmax(dim=-1),
        }

    def critical_corruption_threshold(
        self,
        clean_donors,
        query,
        context,
        corrupted_experts=(0, 1),
        corrupted_class: int = 4,
        stealth: bool = False,
        target=None,
    ):
        """Find the exact critical corruption confidence alpha_crit in [0.10, 1.00] via batched grid + refinement.

        Returns the lowest adversarial confidence on `corrupted_experts` at the queried source
        where the receiver's top-1 prediction flips away from `target` (or 1.0 if immune up to 0.9999).
        """
        _inputs(clean_donors, query, context)
        dtype = self.task_matrices.dtype
        batch_size = clean_donors.shape[0]
        device = clean_donors.device
        p_clean = clean_donors.detach().to(dtype=dtype)
        c_tensor = context.detach().to(dtype=dtype)

        with torch.no_grad():
            clean_logits = self._compute(p_clean, query, c_tensor, detach_donors=True)["logits"]
            if target is None:
                chosen_target = clean_logits.argmax(dim=-1)
            else:
                _target(target, "target", batch_size, device)
                chosen_target = target

            grid = torch.linspace(0.10, 0.9995, 37, device=device, dtype=dtype)
            corrupted_cls = int(corrupted_class) % 10
            decoy_cls = (corrupted_cls + 5) % 10

            batch_views = []
            for conf_val in grid:
                c_f = float(conf_val)
                if stealth:
                    rem = max(0.0, 1.0 - c_f)
                    base_tail = (rem * 0.12) / 8.0
                    vec = torch.full((10,), base_tail, device=device, dtype=dtype)
                    vec[corrupted_cls] = c_f
                    vec[decoy_cls] = rem * 0.88
                else:
                    rem_each = max(0.0, (1.0 - c_f) / 9.0)
                    vec = torch.full((10,), rem_each, device=device, dtype=dtype)
                    vec[corrupted_cls] = c_f
                view = p_clean.clone()
                for b in range(batch_size):
                    s_idx = int(query[b, 0])
                    for e in corrupted_experts:
                        if 0 <= int(e) < 3:
                            view[b, s_idx, int(e)] = vec
                batch_views.append(view)

            stacked = torch.cat(batch_views, dim=0)
            q_rep = query.repeat(len(grid), 1)
            c_rep = c_tensor.repeat(len(grid), 1)
            preds = self._compute(stacked, q_rep, c_rep, detach_donors=True)["logits"].argmax(dim=-1)
            preds = preds.view(len(grid), batch_size)

            alpha_crit = torch.ones(batch_size, device=device, dtype=dtype)
            immune = torch.ones(batch_size, device=device, dtype=torch.bool)
            for b in range(batch_size):
                wrong_mask = preds[:, b] != chosen_target[b]
                if wrong_mask.any():
                    first_idx = int(wrong_mask.nonzero(as_tuple=False)[0].item())
                    alpha_crit[b] = grid[first_idx]
                    immune[b] = False

        return {
            "target": chosen_target,
            "alpha_crit": alpha_crit,
            "immune": immune,
        }

    def forward(self, donors, query, context):
        return self.inspect(donors, query, context)["logits"]


def _dropout_view(donors, view):
    """Deterministic donor dropout without consuming any random generator.

    Replace one quarter of row/source/expert positions with a uniform posterior.
    Four views cover each position exactly once. All rows retain normalized
    inputs; no class, target or selected source is consulted by the mask.
    """
    rows = torch.arange(donors.shape[0], device=donors.device)[:, None, None]
    sources = torch.arange(3, device=donors.device)[None, :, None]
    experts = torch.arange(3, device=donors.device)[None, None, :]
    drop = (rows + 3 * sources + experts + view) % 4 == 0
    return torch.where(drop[..., None], torch.full_like(donors, 0.1), donors)


def _v2_terms(details, views, batch, probabilities, *, rolled: bool):
    """Compute paired or row-rolled Co-Variation & Perturbation-Localized Invariance."""
    n = len(batch["target"])
    rows = torch.arange(n, device=batch["target"].device)
    ref_donors = views[0].detach().roll(1, dims=0) if rolled else views[0].detach()
    ref_target = batch["target"].roll(1, dims=0) if rolled else batch["target"]
    ref_pooled = details[0]["pooled_posterior"].roll(1, dims=0) if rolled else details[0]["pooled_posterior"]

    changed = (ref_target != batch["relevant_target"]).to(dtype=probabilities[1].dtype)
    stale_prob = probabilities[1][rows, ref_target].clamp(max=1.0 - 1e-6)
    repel = (-(1.0 - stale_prob).log() * changed).mean()

    delta_r = (views[1].detach() - ref_donors).abs().mean((-1, -2)).clamp_min(1e-4)
    source_q = delta_r / delta_r.sum(-1, keepdim=True)
    kl_source = -(source_q * details[1]["source_weights"].clamp_min(1e-6).log()).sum(-1).mean()

    task_ce = 0.5 * (
        F.cross_entropy(details[0]["task_logits"], batch["target"])
        + F.cross_entropy(details[1]["task_logits"], batch["relevant_target"])
    )
    effect = 0.5 * (repel + kl_source + task_ce)

    delta_d = (0.5 * (views[3].detach() - ref_donors).abs().sum(-1)).clamp(0.0, 1.0)
    intact_q = (1.0 - delta_d).clamp_min(1e-4)
    intact_q = intact_q / intact_q.sum(-1, keepdim=True)
    kl_expert = -(intact_q * details[3]["expert_weights"].clamp_min(1e-6).log()).sum(-1).mean()
    pooled_inv = 0.5 * (
        (details[2]["pooled_posterior"] - ref_pooled).square().sum(-1).mean()
        + (details[3]["pooled_posterior"] - ref_pooled).square().sum(-1).mean()
    )
    invariance = kl_expert + pooled_inv
    return effect, invariance


def objective(model, batch, arm, effect_weight=1.0, invariance_weight=1.0, *, protocol="v1"):
    """Return the differentiable seven-arm objective and scalar diagnostics.

    Each arm makes exactly four receiver evaluations. Passive repeats factual
    data; dropout applies four deterministic factual donor masks. Other arms
    share the same factual/relevant/irrelevant/damaged evaluations and labels.
    In protocol='v1', effect and invariance operate on final class probability
    vectors. In protocol='v2' (or when `model.forensic` is True), effect uses
    paired pre-intervention target repulsion, source co-variation routing and
    direct task-logit supervision, while invariance uses perturbation-localized
    specialist weighting and pre-shortcut pooled-posterior stability.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if protocol not in ("v1", "v2"):
        raise ValueError("protocol must be 'v1' or 'v2'")
    for name, value in (("effect_weight", effect_weight), ("invariance_weight", invariance_weight)):
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    required = (*_VIEW_NAMES, "query", "context", "target", "relevant_target")
    if not isinstance(batch, Mapping) or any(key not in batch for key in required):
        raise ValueError(f"batch must contain {required}")
    donors, query, context = batch["donors"], batch["query"], batch["context"]
    _inputs(donors, query, context)
    n = donors.shape[0]
    for name in _VIEW_NAMES[1:]:
        _probabilities(batch[name], name, (3, 3, 10), n)
        if batch[name].device != donors.device:
            raise ValueError("all donor views must be on the same device")
    _target(batch["target"], "target", n, donors.device)
    _target(batch["relevant_target"], "relevant_target", n, donors.device)

    if arm == "passive":
        views = [donors] * 4
    elif arm == "dropout":
        views = [_dropout_view(donors, i) for i in range(4)]
    else:
        views = [batch[name] for name in _VIEW_NAMES]
    labels = [batch["target"]] * 4
    if arm not in ("passive", "dropout"):
        labels[1] = batch["relevant_target"]

    use_v2 = (protocol == "v2" or bool(getattr(model, "forensic", False))) and hasattr(model, "inspect")
    if use_v2:
        details = [model.inspect(view.detach(), query, context) for view in views]
        logits = [d["logits"] for d in details]
    else:
        details = None
        logits = [model(view.detach(), query, context) for view in views]

    for prediction in logits:
        if prediction.shape != (n, 10) or not torch.isfinite(prediction).all():
            raise ValueError("receiver must return finite logits with shape [batch,10]")
    ce = torch.stack([F.cross_entropy(prediction, target) for prediction, target in zip(logits, labels)]).mean()
    probabilities = [prediction.softmax(-1) for prediction in logits]

    if use_v2:
        effect, invariance = _v2_terms(details, views, batch, probabilities, rolled=False)
        unpaired_effect, unpaired_invariance = _v2_terms(details, views, batch, probabilities, rolled=True)
    else:
        ideal_effect = F.one_hot(batch["relevant_target"], 10) - F.one_hot(batch["target"], 10)
        observed_effect = probabilities[1] - probabilities[0]
        effect = (observed_effect - ideal_effect).square().sum(-1).mean()
        residual_relevant = probabilities[1] - F.one_hot(batch["relevant_target"], 10)
        residual_factual = probabilities[0] - F.one_hot(batch["target"], 10)
        unpaired_effect = (residual_relevant.roll(1, dims=0) - residual_factual).square().sum(-1).mean()
        invariance = torch.stack([
            (probabilities[i] - probabilities[0]).square().sum(-1).mean()
            for i in (2, 3)
        ]).mean()
        unpaired_invariance = invariance

    loss = ce
    if arm in ("effect_only", "tether"):
        loss = loss + float(effect_weight) * effect
    if arm == "unpaired":
        loss = loss + float(effect_weight) * unpaired_effect
    if arm in ("invariance_only", "tether"):
        loss = loss + float(invariance_weight) * invariance
    if arm == "unpaired":
        loss = loss + float(invariance_weight) * unpaired_invariance
    diagnostics = {"ce": float(ce.detach()), "effect": float(effect.detach()),
                   "unpaired_effect": float(unpaired_effect.detach()),
                   "invariance": float(invariance.detach()),
                   "unpaired_invariance": float(unpaired_invariance.detach()),
                   "loss": float(loss.detach()), "views": 4}
    return loss, diagnostics
