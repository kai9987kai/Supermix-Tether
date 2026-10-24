"""Behavior tests for the read-only loopback results server."""

from contextlib import contextmanager
import http.client
import importlib
import importlib.util
import json
from pathlib import Path
import threading

import pytest


def server_module():
    # Keep initial implementation absence an explicit failure, not a collection error.
    try:
        spec = importlib.util.find_spec("tether.server")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "The read-only tether.server API has not been implemented"
    return importlib.import_module("tether.server")


@contextmanager
def running_server(study):
    server = server_module().make_server(study, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port, path, method="GET", body=None, headers=None, timeout=15):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def write_summary(study):
    study.mkdir()
    (study / "summary.json").write_text(json.dumps({
        "schema": "supermix-tether-study-v1",
        "status": "complete",
        "config": {"receiver_steps": 2},
        "rows": [{"seed": 17, "arm": "tether", "metrics": {
            "shift": {"accuracy": 0.4, "donor_erasure_gap": 0.3}
        }}],
        "comparisons": [],
        "decision": {"promoted": False},
        "provenance": {"source_sha256": "abc123"},
    }), encoding="utf-8")


def test_missing_study_is_clear_json_and_static_viewer_remains_available(tmp_path):
    with running_server(tmp_path / "missing") as port:
        status, headers, body = request(port, "/api/results")
        assert status == 404
        result = json.loads(body)
        assert result["status"] == "missing"
        assert "summary" in result["error"].lower()
        assert headers["Content-Type"].startswith("application/json")
        status, headers, body = request(port, "/")
        assert status == 200
        assert b"Supermix Tether" in body


def test_results_load_real_saved_summary_and_disable_caching(tmp_path):
    study = tmp_path / "study"
    write_summary(study)
    with running_server(study) as port:
        status, headers, body = request(port, "/api/results")
        assert status == 200
        result = json.loads(body)
        assert result["schema"] == "supermix-tether-study-v1"
        assert result["rows"][0]["metrics"]["shift"]["accuracy"] == 0.4
        assert result["provenance"]["source_sha256"] == "abc123"
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("path", [
    "/../summary.json", "/%2e%2e/summary.json",
    "/%2e%2e%5csummary.json", "/server.py", "/api/../summary.json",
    "/%252e%252e/summary.json",
])
def test_static_paths_cannot_escape_web_assets(tmp_path, path):
    (tmp_path / "summary.json").write_text("SECRET STUDY FILE", encoding="utf-8")
    with running_server(tmp_path) as port:
        status, _, body = request(port, path)
        assert status == 404
        assert b"SECRET STUDY FILE" not in body


@pytest.mark.parametrize("query", [
    "index=-1", "index=nan", "index=1.5", "index=1000001",
    "arm=unknown", "mode=train", "index=0&index=1", "unexpected=value",
    "seed=-1", "seed=abc",
])
def test_invalid_example_queries_are_rejected_before_inference(tmp_path, query):
    study = tmp_path / "study"
    write_summary(study)
    with running_server(study) as port:
        status, _, body = request(port, "/api/example?" + query)
        assert status == 400
        assert "error" in json.loads(body)


def test_example_without_saved_run_does_not_attempt_inference(tmp_path):
    with running_server(tmp_path / "missing") as port:
        status, _, body = request(port, "/api/example?index=0&arm=tether&mode=factual")
        assert status == 404
        assert json.loads(body)["status"] == "missing"


def test_partial_json_reports_unavailable_instead_of_empty_success(tmp_path):
    (tmp_path / "summary.json").write_text('{"schema":', encoding="utf-8")
    with running_server(tmp_path) as port:
        status, _, body = request(port, "/api/results")
        assert status == 503
        assert json.loads(body)["status"] == "unavailable"


def test_post_cannot_modify_study(tmp_path):
    study = tmp_path / "study"
    write_summary(study)
    original = (study / "summary.json").read_bytes()
    with running_server(study) as port:
        status, _, body = request(port, "/api/results", method="POST")
        assert status == 405
        assert "error" in json.loads(body)
    assert (study / "summary.json").read_bytes() == original


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.1", "::", "example.org"])
def test_server_refuses_nonloopback_hosts(tmp_path, host):
    with pytest.raises(ValueError, match="loopback"):
        server_module().make_server(tmp_path, host=host, port=0)


def test_live_predict_endpoints():
    study_path = Path("outputs/v2-confirmation")
    if not (study_path / "summary.json").exists():
        pytest.skip("outputs/v2-confirmation not present")
    with running_server(study_path) as port:
        status, _, body = request(port, "/api/live_predict?digit=7")
        assert status == 200
        payload = json.loads(body)
        assert payload["prediction"] == 7
        assert "epistemic_trust" in payload
        assert "abstain" in payload
        assert payload["abstain"] is False

        # POST with 3/3 corruption -> safe abstention
        post_data = json.dumps({"digit": 7, "tamper_mode": "corrupt_3"}).encode("utf-8")
        status, _, body = request(port, "/api/live_predict", method="POST", body=post_data, headers={"Content-Type": "application/json"})
        assert status == 200
        post_payload = json.loads(body)
        assert post_payload["abstain"] is True
        assert post_payload["selective_prediction"] == -1


