import torch
import torch.nn.functional as F
from tether.model import TetherReceiver
from tether.experiment import live_predict, inspect_example


def test_epistemic_trust_and_selective_abstention():
    model = TetherReceiver(forensic=True, epistemic_threshold=0.10)
    batch_size = 4
    # Create natural peaked posteriors with structured dark knowledge
    gen = torch.Generator().manual_seed(42)
    logits = torch.randn(batch_size, 3, 3, 10, generator=gen) * 2.5
    clean_donors = logits.softmax(-1)

    query = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1]], dtype=torch.long)
    context = torch.full((batch_size, 10), 0.1)

    detail_clean = model.inspect(clean_donors, query, context)
    assert (detail_clean["epistemic_trust"] > 0.20).all()
    assert not detail_clean["abstain"].any()
    assert (detail_clean["selective_prediction"] >= 0).all()

    # Corrupt all 3 experts on source 0 with uniform tail (spoofed sensor)
    corrupted_donors = clean_donors.clone()
    conf = 0.99
    other_p = (1.0 - conf) / 9.0
    spoof = torch.full((10,), other_p)
    spoof[4] = conf
    # Apply to all experts of queried source 0 for row 0
    corrupted_donors[0, 0, :] = spoof

    detail_corrupted = model.inspect(corrupted_donors, query, context)
    # Row 0 queried source 0 has 3/3 corrupted experts with uniform tail -> tail_entropy = 1.0 -> trust = 0.0
    assert detail_corrupted["trust_score"][0, 0, 0].item() < 0.05
    assert detail_corrupted["epistemic_trust"][0].item() < 0.05
    assert detail_corrupted["abstain"][0].item() is True
    assert detail_corrupted["selective_prediction"][0].item() == -1


def test_causal_attribution():
    model = TetherReceiver(forensic=True)
    donors = torch.full((1, 3, 3, 10), 0.01)
    for s in range(3):
        for e in range(3):
            donors[0, s, e, 3] = 0.91

    query = torch.tensor([[1, 0]], dtype=torch.long)  # Query source 1
    context = torch.full((1, 10), 0.1)

    attr = model.attribute(donors, query, context)
    assert "saliency" in attr
    assert "source_saliency" in attr
    assert "expert_saliency" in attr
    assert attr["saliency"].shape == (1, 3, 3, 10)
    assert attr["source_saliency"].shape == (1, 3)
    assert attr["expert_saliency"].shape == (1, 3, 3)
    assert (attr["saliency"] >= 0).all()

    # Saliency on queried source 1 should be strictly positive
    assert attr["source_saliency"][0, 1].item() > 0.0


def test_live_predict_clean_and_tampered():
    res_clean = live_predict("outputs/v2-confirmation", digit=3, source=0, operation=0, tamper_mode="clean")
    assert res_clean["abstain"] is False
    assert res_clean["selective_prediction"] == res_clean["prediction"]
    assert res_clean["epistemic_trust"] > 0.10

    res_corrupt3 = live_predict("outputs/v2-confirmation", digit=3, source=0, operation=0, tamper_mode="corrupt_3", corrupted_class=9)
    assert res_corrupt3["abstain"] is True
    assert res_corrupt3["selective_prediction"] == -1
    assert res_corrupt3["epistemic_trust"] < 0.05
    assert len(res_corrupt3["source_saliency"]) == 3
