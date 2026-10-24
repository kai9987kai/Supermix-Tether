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
