"""Receiver and objective contracts, with independent probability counterexamples."""

import importlib
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.fixture(autouse=True)
def two_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model_api():
    return importlib.import_module("tether.model")


def literal_batch(n=2):
    # A fully normalized posterior, with literal expected counterfactual effects.
    factual = torch.tensor([0.6, 0.2] + [0.025] * 8)
    relevant = torch.tensor([0.2, 0.6] + [0.025] * 8)
    return {
        "donors": factual.expand(n, 3, 3, 10).clone(),
        "relevant": relevant.expand(n, 3, 3, 10).clone(),
        "irrelevant": factual.expand(n, 3, 3, 10).clone(),
        "damaged": factual.expand(n, 3, 3, 10).clone(),
        "query": torch.tensor([[0, 0]] * n),
        "context": torch.full((n, 10), 0.1),
        "target": torch.zeros(n, dtype=torch.long),
        "relevant_target": torch.ones(n, dtype=torch.long),
    }


class FirstPosterior(nn.Module):
    """An actual transparent classifier: first posterior is its output distribution."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def forward(self, donors, query, context):
        self.seen.append(donors.detach().clone())
        return donors[:, 0, 0].log() * self.scale


def test_receiver_contract_available():
    try:
        module = model_api()
    except ModuleNotFoundError:
        module = None
    assert module is not None, "Task 2 receiver/objective implementation is missing"
    assert callable(module.TetherReceiver) and callable(module.objective)


def test_inspection_normalizes_routes_and_matches_forward():
    torch.manual_seed(43)
    model = model_api().TetherReceiver()
    batch = literal_batch()
    result = model.inspect(batch["donors"], batch["query"], batch["context"])
    assert result["source_weights"].shape == (2, 3)
    assert result["expert_weights"].shape == (2, 3, 3)
    assert result["logits"].shape == (2, 10)
    torch.testing.assert_close(result["source_weights"].sum(-1), torch.ones(2))
    torch.testing.assert_close(result["expert_weights"].sum(-1), torch.ones(2, 3))
    assert (result["source_weights"] > 0).all()
    assert (result["expert_weights"] > 0).all()
    torch.testing.assert_close(
        model(batch["donors"], batch["query"], batch["context"]), result["logits"]
    )


def test_external_seed_controls_initialization_and_state_dict_roundtrip():
    torch.manual_seed(11)
    first = model_api().TetherReceiver()
    torch.manual_seed(11)
    second = model_api().TetherReceiver()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(12)
    third = model_api().TetherReceiver()
    third.load_state_dict(first.state_dict())
    batch = literal_batch()
    torch.testing.assert_close(
        first(batch["donors"], batch["query"], batch["context"]),
        third(batch["donors"], batch["query"], batch["context"]), rtol=0, atol=0,
    )


@pytest.mark.parametrize("field,mutation", [
    ("donors", lambda x: x[:, :2]),
    ("donors", lambda x: x.fill_(float("nan"))),
    ("donors", lambda x: x.fill_(0.2)),
    ("donors", lambda x: x.fill_(-0.1)),
    ("query", lambda x: x.float()),
    ("query", lambda x: x.fill_(3)),
    ("query", lambda x: x.fill_(-1)),
    ("context", lambda x: x[:, :9]),
    ("context", lambda x: x.fill_(float("inf"))),
    ("context", lambda x: x.fill_(0.2)),
])
def test_public_boundary_rejects_invalid_inputs(field, mutation):
    batch = literal_batch()
    batch[field] = mutation(batch[field].clone())
    with pytest.raises((ValueError, TypeError)):
        model_api().TetherReceiver()(batch["donors"], batch["query"], batch["context"])


def test_public_boundary_rejects_empty_batch():
    batch = literal_batch(0)
    with pytest.raises(ValueError):
        model_api().TetherReceiver()(batch["donors"], batch["query"], batch["context"])


def test_literal_relevant_effect_distinguishes_wrong_change():
    objective = model_api().objective
    batch = literal_batch(1)
    _, correct_direction = objective(FirstPosterior(), batch, "effect_only")
    # Change is [-.4,+.4], ideal change [-1,+1]: two squared errors of .6.
    assert correct_direction["effect"] == pytest.approx(0.72, abs=1e-6)
    batch["relevant"] = batch["donors"].clone()
    _, no_effect = objective(FirstPosterior(), batch, "effect_only")
    assert no_effect["effect"] == pytest.approx(2.0, abs=1e-6)
    assert no_effect["effect"] > correct_direction["effect"]


def test_literal_irrelevant_and_damage_sensitivity_are_both_penalized():
    batch = literal_batch(1)
    batch["irrelevant"][:] = torch.tensor([0.2, 0.6] + [0.025] * 8)
    batch["damaged"][:] = torch.tensor([0.4, 0.4] + [0.025] * 8)
    loss, info = model_api().objective(FirstPosterior(), batch, "tether")
    # Squared changes: .4^2+.4^2=.32 and .2^2+.2^2=.08; their mean is .20.
    assert info["invariance"] == pytest.approx(0.20, abs=1e-6)
    assert float(loss.detach()) == pytest.approx(info["ce"] + 0.72 + 0.20, abs=1e-6)
    batch["irrelevant"] = batch["donors"].clone()
    batch["damaged"] = batch["donors"].clone()
    _, stable = model_api().objective(FirstPosterior(), batch, "tether")
    assert stable["invariance"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("arm,extras", [
    ("passive", (0, 0)), ("dropout", (0, 0)), ("iit", (0, 0)),
    ("effect_only", (1, 0)), ("invariance_only", (0, 1)), ("tether", (1, 1)),
    ("unpaired", (1, 1)),
])
def test_each_arm_has_four_evaluations_and_correct_loss_terms(arm, extras):
    batch = literal_batch()
    batch["irrelevant"][:] = torch.tensor([0.2, 0.6] + [0.025] * 8)
    model = FirstPosterior()
    loss, info = model_api().objective(model, batch, arm, effect_weight=0.3, invariance_weight=0.7)
    assert len(model.seen) == 4
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in info.values())
    assert "unpaired_effect" in info
    effect_key = "unpaired_effect" if arm == "unpaired" else "effect"
    expected = info["ce"] + 0.3 * extras[0] * info[effect_key] + 0.7 * extras[1] * info["invariance"]
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    if arm == "passive":
        assert all(torch.equal(view, batch["donors"]) for view in model.seen)
        assert info["ce"] == pytest.approx(-math.log(0.6), abs=1e-6)
    elif arm == "dropout":
        assert any(not torch.equal(view, batch["donors"]) for view in model.seen)
        state = torch.random.get_rng_state().clone()
        repeated = FirstPosterior()
        model_api().objective(repeated, batch, arm)
        assert torch.equal(state, torch.random.get_rng_state())
        assert all(torch.equal(a, b) for a, b in zip(model.seen, repeated.seen))
    else:
        assert all(torch.equal(actual, batch[name]) for actual, name in zip(
            model.seen, ["donors", "relevant", "irrelevant", "damaged"]))
        # Correct labels have probabilities .6, .6, .2 and .6 respectively.
        assert info["ce"] == pytest.approx(-(3 * math.log(0.6) + math.log(0.2)) / 4, abs=1e-6)


def test_unpaired_effect_breaks_alignment_but_retains_marginal_supervision():
    batch = literal_batch(2)
    batch["donors"][1] = torch.tensor([0.3, 0.5] + [0.025] * 8)
    batch["relevant"][1] = torch.tensor([0.5, 0.3] + [0.025] * 8)
    batch["irrelevant"] = batch["relevant"].clone()
    batch["damaged"] = batch["donors"].clone()
    paired_model, unpaired_model = FirstPosterior(), FirstPosterior()
    paired_loss, paired = model_api().objective(paired_model, batch, "tether", 0.3, 0.7)
    unpaired_loss, unpaired = model_api().objective(unpaired_model, batch, "unpaired", 0.3, 0.7)
    # Nonidentical residuals in classes 0/1:
    # factual: (-.4,.2), (-.7,.5); relevant: (.2,-.4), (.5,-.7).
    # Matched squared residual differences: .72 and 2.88, mean 1.80.
    # Rolling relevant rows gives (.9,-.9) twice: unpaired mean 1.62.
    # Other eight residual coordinates are identical and cancel in both.
    assert paired["effect"] == pytest.approx(1.80, abs=1e-6)
    assert unpaired["unpaired_effect"] == pytest.approx(1.62, abs=1e-6)
    assert paired["unpaired_effect"] == pytest.approx(1.62, abs=1e-6)
    assert unpaired["ce"] == pytest.approx(paired["ce"], abs=1e-7)
    # Irrelevant squared changes average .20; damaged changes are zero.
    assert unpaired["invariance"] == pytest.approx(0.10, abs=1e-6)
    assert float(paired_loss.detach()) == pytest.approx(paired["ce"] + 0.61, abs=1e-6)
    assert float(unpaired_loss.detach()) == pytest.approx(unpaired["ce"] + 0.556, abs=1e-6)
    assert len(paired_model.seen) == len(unpaired_model.seen) == 4
    assert all(torch.equal(first, second) for first, second in zip(paired_model.seen, unpaired_model.seen))
    unpaired_loss.backward()
    assert unpaired_model.scale.grad is not None and torch.isfinite(unpaired_model.scale.grad)


@pytest.mark.parametrize("change", ["unknown_arm", "negative_weight", "nan_weight", "bad_target", "missing_view", "bad_view"])
def test_objective_validates_before_training(change):
    batch, arm, kwargs = literal_batch(), "tether", {}
    if change == "unknown_arm":
        arm = "invented"
    elif change == "negative_weight":
        kwargs["effect_weight"] = -1
    elif change == "nan_weight":
        kwargs["invariance_weight"] = float("nan")
    elif change == "bad_target":
        batch["target"][0] = 10
    elif change == "missing_view":
        del batch["relevant"]
    elif change == "bad_view":
        batch["damaged"][0, 0, 0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        model_api().objective(FirstPosterior(), batch, arm, **kwargs)


def test_receiver_learns_literal_tasks_and_keeps_donors_frozen():
    # Not sklearn data or task_targets: independently generated supervised fixture.
    # An untrained hard-coded source/transform would pass immediately; require a
    # substantial measured training improvement and nonzero gradients instead.
    torch.manual_seed(119)
    model = model_api().TetherReceiver()
    rng = torch.Generator().manual_seed(203)
    digits = torch.randint(10, (240, 3), generator=rng)
    query = torch.stack((torch.randint(3, (240,), generator=rng),
                         torch.randint(3, (240,), generator=rng)), -1)
    selected = digits[torch.arange(240), query[:, 0]]
    target = selected.clone()
    target[query[:, 1] == 1] = (selected[query[:, 1] == 1] + 1) % 10
    target[query[:, 1] == 2] = 9 - selected[query[:, 1] == 2]
    donors = (0.95 * F.one_hot(digits, 10).float() + 0.005).unsqueeze(2).expand(-1, -1, 3, -1).clone()
    donors.requires_grad_(True)
    context = torch.full((240, 10), 0.1)
    with torch.no_grad():
        initial = F.cross_entropy(model(donors, query, context), target).item()
    opt = torch.optim.Adam(model.parameters(), lr=0.025)
    first_gradients = None
    for _ in range(180):
        opt.zero_grad()
        loss = F.cross_entropy(model(donors, query, context), target)
        loss.backward()
        if first_gradients is None:
            first_gradients = {name: p.grad.abs().sum().item() for name, p in model.named_parameters() if p.grad is not None}
        opt.step()
    with torch.no_grad():
        logits = model(donors, query, context)
        final = F.cross_entropy(logits, target).item()
        erased = model(torch.full_like(donors, 0.1), query, context)
    assert donors.grad is None, "Receiver training must not update frozen donors"
    assert len(first_gradients) >= 8 and sum(v > 0 for v in first_gradients.values()) >= 8
    assert final < initial * 0.25
    assert (logits.argmax(-1) == target).float().mean().item() > 0.95
    assert (erased.argmax(-1) == target).float().mean().item() < 0.30


def test_objective_retains_receiver_gradient_flow():
    torch.manual_seed(8)
    model = model_api().TetherReceiver()
    loss, _ = model_api().objective(model, literal_batch(), "tether")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_forensic_receiver_exposes_dark_knowledge_diagnostics_and_distinguishes_corruption():
    torch.manual_seed(29)
    model = model_api().TetherReceiver(forensic=True)
    assert model.forensic is True
    batch = literal_batch(2)
    # Expert 0 has realistic runner-up tail structure; Expert 1 & 2 have flat fabricated tail.
    intact = torch.tensor([0.92, 0.05, 0.015, 0.005, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001])
    fabricated = torch.tensor([0.001, 0.991] + [0.001] * 8)
    batch["donors"][:, 0, 0] = intact
    batch["donors"][:, 0, 1] = fabricated
    batch["donors"][:, 0, 2] = fabricated
    info = model.inspect(batch["donors"], batch["query"], batch["context"])
    assert info["trust_score"].shape == (2, 3, 3)
    assert info["tail_entropy"].shape == (2, 3, 3)
    assert info["expert_margin"].shape == (2, 3, 3)
    # Intact specialist has non-maximal tail entropy -> positive trust score; fabricated has zero trust score.
    assert (info["trust_score"][:, 0, 0] > 0.35).all()
    assert (info["trust_score"][:, 0, 1] < 1e-4).all()
    assert (info["trust_score"][:, 0, 2] < 1e-4).all()


def test_v2_objective_distinguishes_paired_from_unpaired_perturbation_localization():
    torch.manual_seed(31)
    batch = literal_batch(2)
    # Row 0 and Row 1 have distinct donor posteriors and targets so rolling rows breaks alignment.
    batch["donors"][0, 0] = torch.tensor([0.82, 0.10] + [0.01] * 8)
    batch["donors"][1, 0] = torch.tensor([0.10, 0.10, 0.73] + [0.01] * 7)
    batch["target"] = torch.tensor([0, 2], dtype=torch.long)
    batch["relevant"] = batch["donors"].clone()
    batch["relevant"][0, 0] = torch.tensor([0.10, 0.82] + [0.01] * 8)
    batch["relevant"][1, 0] = torch.tensor([0.10, 0.10, 0.10, 0.64] + [0.01] * 6)
    batch["relevant_target"] = torch.tensor([1, 3], dtype=torch.long)
    batch["irrelevant"] = batch["donors"].clone()
    batch["damaged"] = batch["donors"].clone()
    # Corrupt only expert 2 on source 0.
    batch["damaged"][0, 0, 2] = torch.tensor([0.001, 0.991] + [0.001] * 8)
    batch["damaged"][1, 0, 2] = torch.tensor([0.001, 0.001, 0.001, 0.991] + [0.001] * 6)

    torch.manual_seed(31)
    paired_model = model_api().TetherReceiver(forensic=True)
    torch.manual_seed(31)
    unpaired_model = model_api().TetherReceiver(forensic=True)

    paired_loss, paired_info = model_api().objective(paired_model, batch, "tether", protocol="v2")
    unpaired_loss, unpaired_info = model_api().objective(unpaired_model, batch, "unpaired", protocol="v2")
    assert paired_info["effect"] != pytest.approx(paired_info["unpaired_effect"], abs=1e-4)
    assert paired_info["invariance"] != pytest.approx(paired_info["unpaired_invariance"], abs=1e-4)
    assert float(paired_loss.detach()) == pytest.approx(
        paired_info["ce"] + paired_info["effect"] + paired_info["invariance"], abs=1e-6
    )
    assert float(unpaired_loss.detach()) == pytest.approx(
        unpaired_info["ce"] + unpaired_info["unpaired_effect"] + unpaired_info["unpaired_invariance"], abs=1e-6
    )
    paired_loss.backward()
    grads = [p.grad for p in paired_model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
