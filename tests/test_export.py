from pathlib import Path
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
import torch
from tether.data import DonorBank
from tether.export import export_donor_bank_onnx, export_receiver_onnx, export_checkpoint
from tether.model import TetherReceiver


def test_export_receiver_onnx_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "receiver.onnx"
        receiver = TetherReceiver(forensic=True)
        export_receiver_onnx(receiver, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 1000

        # Check ONNX model validity
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)

        # Run ONNX Runtime session and compare with PyTorch
        session = ort.InferenceSession(str(onnx_path))
        donors = np.random.uniform(0.01, 0.9, size=(2, 3, 3, 10)).astype(np.float32)
        donors = donors / donors.sum(axis=-1, keepdims=True)
        query = np.array([[0, 1], [2, 0]], dtype=np.int64)
        context = np.full((2, 10), 0.1, dtype=np.float32)

        ort_inputs = {"donors": donors, "query": query, "context": context}
        ort_outputs = session.run(None, ort_inputs)
        logits_ort = ort_outputs[0]

        with torch.no_grad():
            pt_detail = receiver.inspect(torch.tensor(donors), torch.tensor(query), torch.tensor(context))
            logits_pt = pt_detail["logits"].numpy()

        np.testing.assert_allclose(logits_ort, logits_pt, atol=1e-4, rtol=1e-4)


def test_export_donor_bank_onnx():
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "donor_bank.onnx"
        bank = DonorBank(initialization_seed=42)
        export_donor_bank_onnx(bank, onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 1000

        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)

        session = ort.InferenceSession(str(onnx_path))
        images = np.random.uniform(0.0, 1.0, size=(3, 64)).astype(np.float32)
        ort_out = session.run(None, {"images": images})[0]

        with torch.no_grad():
            pt_out = bank(torch.tensor(images)).numpy()

        np.testing.assert_allclose(ort_out, pt_out, atol=1e-4, rtol=1e-4)


def test_export_checkpoint_cli_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "exported"
        ckpt_path = Path("outputs/v2-confirmation/checkpoints/seed-101-tether.pt")
        meta = export_checkpoint(ckpt_path, out_dir)
        assert (out_dir / "receiver.onnx").exists()
        assert (out_dir / "donors.onnx").exists()
        assert (out_dir / "onnx_manifest.json").exists()
        assert meta["schema"] == "supermix-tether-onnx-v1"
