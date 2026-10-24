"""ONNX export and cross-framework numerical verification for Supermix Tether."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .data import DonorBank
from .model import TetherReceiver
from .receipts import sha256


class _ReceiverWrapper(nn.Module):
    """Wrapper that outputs logits, source weights, expert weights, and trust scores for ONNX."""

    def __init__(self, receiver: TetherReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def forward(self, donors: torch.Tensor, query: torch.Tensor, context: torch.Tensor):
        info = self.receiver._compute(donors, query, context)
        return info["logits"], info["source_weights"], info["expert_weights"], info["trust_score"]


def export_receiver_onnx(receiver: TetherReceiver, output_path: str | Path,
                         opset_version: int = 14) -> Path:
    """Export TetherReceiver to ONNX and verify outputs with ONNX Runtime."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _ReceiverWrapper(receiver)
    wrapper.eval()

    dummy_donors = torch.full((2, 3, 3, 10), 0.1, dtype=torch.float32)
    dummy_query = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    dummy_context = torch.full((2, 10), 0.1, dtype=torch.float32)

    with torch.no_grad():
        torch_logits, torch_sw, torch_ew, torch_ts = wrapper(dummy_donors, dummy_query, dummy_context)

    torch.onnx.export(
        wrapper,
        (dummy_donors, dummy_query, dummy_context),
        str(output_path),
        input_names=["donors", "query", "context"],
        output_names=["logits", "source_weights", "expert_weights", "trust_score"],
        dynamic_axes={
            "donors": {0: "batch_size"},
            "query": {0: "batch_size"},
            "context": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "source_weights": {0: "batch_size"},
            "expert_weights": {0: "batch_size"},
            "trust_score": {0: "batch_size"},
        },
        dynamo=False,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "donors": dummy_donors.numpy(),
        "query": dummy_query.numpy(),
        "context": dummy_context.numpy(),
    }
    ort_logits, ort_sw, ort_ew, ort_ts = session.run(None, ort_inputs)

    np.testing.assert_allclose(ort_logits, torch_logits.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ort_sw, torch_sw.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ort_ew, torch_ew.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ort_ts, torch_ts.numpy(), rtol=1e-5, atol=1e-5)

    return output_path


def export_donor_bank_onnx(bank: DonorBank, output_path: str | Path,
                           opset_version: int = 14) -> Path:
    """Export DonorBank to ONNX and verify outputs with ONNX Runtime."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank.eval()

    dummy_images = torch.rand((4, 64), dtype=torch.float32)
    with torch.no_grad():
        torch_logits = bank(dummy_images)

    torch.onnx.export(
        bank,
        (dummy_images,),
        str(output_path),
        input_names=["images"],
        output_names=["specialist_logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "specialist_logits": {0: "batch_size"},
        },
        dynamo=False,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"images": dummy_images.numpy()}
    ort_logits = session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(ort_logits, torch_logits.numpy(), rtol=1e-5, atol=1e-5)
    return output_path


def export_checkpoint(checkpoint_path: str | Path, output_dir: str | Path,
                      opset_version: int = 14) -> dict[str, Any]:
    """Export both DonorBank and TetherReceiver from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bank = DonorBank(config=ckpt["bank_config"])
    bank.load_state_dict(ckpt["bank_state"])
    bank.eval()

    rel_weight = ckpt["model_state"].get("reliability.0.weight")
    forensic = (rel_weight is not None and rel_weight.shape[-1] == 20)
    receiver = TetherReceiver(forensic=forensic)
    receiver.load_state_dict(ckpt["model_state"])
    receiver.eval()

    receiver_onnx = export_receiver_onnx(receiver, output_dir / "receiver.onnx", opset_version=opset_version)
    bank_onnx = export_donor_bank_onnx(bank, output_dir / "donors.onnx", opset_version=opset_version)

    manifest = {
        "schema": "supermix-tether-onnx-v1",
        "source_checkpoint": str(checkpoint_path.name),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "receiver_onnx": receiver_onnx.name,
        "receiver_onnx_sha256": sha256(receiver_onnx),
        "donors_onnx": bank_onnx.name,
        "donors_onnx_sha256": sha256(bank_onnx),
        "opset_version": opset_version,
        "forensic_mode": forensic,
        "verification": "ONNX Runtime output verified against PyTorch within atol=1e-5, rtol=1e-5",
    }
    manifest_path = output_dir / "onnx_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
