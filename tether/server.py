"""Read-only, loopback-only HTTP viewer for saved Supermix Tether studies."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import socket
import threading
from urllib.parse import parse_qs, unquote, urlsplit

WEB_ROOT = Path(__file__).resolve().parent / "web"
ARMS = ("passive", "dropout", "iit", "effect_only", "invariance_only", "unpaired", "tether")
INFERENCE_LOCK = threading.Lock()
MODES = ("factual", "relevant", "irrelevant", "damaged", "erased", "context_shift")
ASSETS = {"/": "index.html", "/index.html": "index.html",
          "/app.js": "app.js", "/styles.css": "styles.css"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}


def _loopback_host(host: str) -> None:
    if host.lower() == "localhost":
        return
    try:
        allowed = ipaddress.ip_address(host).is_loopback
    except ValueError:
        allowed = False
    if not allowed:
        raise ValueError("The study viewer can bind only to a loopback host.")


def make_server(study_dir: str | Path, host: str = "127.0.0.1",
                port: int = 0) -> HTTPServer:
    """Create a read-only local server without starting its event loop."""
    _loopback_host(host)
    study = Path(study_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Keep experiment/CLI output readable; errors remain in HTTP JSON.
            pass

        def _respond(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; "
                             "base-uri 'none'; frame-ancestors 'none'; object-src 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload, allow_nan=False, ensure_ascii=True).encode("utf-8")
            self._respond(code, body, "application/json; charset=utf-8")

        def _summary(self):
            try:
                path = study / "summary.json"
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("Saved summary exceeds the 16 MiB viewer limit.")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Saved summary must be a JSON object.")
                # Validate serializability before emitting HTTP success headers.
                json.dumps(payload, allow_nan=False)
                return payload
            except FileNotFoundError:
                self._json(404, {"status": "missing",
                                "error": "Study summary.json is missing. Complete a saved study first."})
            except (OSError, ValueError, UnicodeError) as exc:
                self._json(503, {"status": "unavailable",
                                "error": f"Saved summary is not readable: {exc}"})
            return None

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/api/results":
                payload = self._summary()
                if payload is not None:
                    self._json(200, payload)
                return
            if path == "/api/example":
                try:
                    values = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                    if set(values) - {"index", "arm", "mode"} or any(len(v) != 1 for v in values.values()):
                        raise ValueError("Use one index, arm and mode parameter only.")
                    raw_index = values.get("index", ["0"])[0]
                    if not raw_index.isascii() or not raw_index.isdecimal():
                        raise ValueError("Example index must be a non-negative integer.")
                    index = int(raw_index)
                    if index > 1_000_000:
                        raise ValueError("Example index exceeds the viewer limit.")
                    arm = values.get("arm", ["tether"])[0]
                    mode = values.get("mode", ["factual"])[0]
                    if arm not in ARMS:
                        raise ValueError("Unknown receiver arm.")
                    if mode not in MODES:
                        raise ValueError("Unknown inspection mode.")
                except ValueError as exc:
                    self._json(400, {"status": "invalid", "error": str(exc)})
                    return
                summary = self._summary()
                if summary is None:
                    return
                if summary.get("status") != "complete":
                    self._json(409, {"status": "incomplete",
                                     "error": "Inspection requires a completed saved study."})
                    return
                try:
                    from tether.experiment import inspect_example
                    # Restoring a checkpoint uses Torch's shared RNG context;
                    # serialize saved-model inspections across browser requests.
                    with INFERENCE_LOCK:
                        payload = inspect_example(study, index, arm, mode)
                    json.dumps(payload, allow_nan=False)
                except FileNotFoundError as exc:
                    self._json(404, {"status": "missing", "error": str(exc)})
                except (ValueError, IndexError, KeyError) as exc:
                    self._json(400, {"status": "invalid", "error": str(exc)})
                except (ImportError, OSError, RuntimeError, TypeError) as exc:
                    self._json(503, {"status": "unavailable",
                                     "error": f"Saved inference is unavailable: {exc}"})
                else:
                    self._json(200, payload)
                return
            filename = ASSETS.get(path)
            if filename is None:
                self._json(404, {"status": "missing", "error": "Unknown local viewer resource."})
                return
            try:
                asset = WEB_ROOT / filename
                self._respond(200, asset.read_bytes(), CONTENT_TYPES[asset.suffix])
            except OSError:
                self._json(404, {"status": "missing", "error": "Viewer asset is unavailable."})

        def do_POST(self):
            self._json(405, {"status": "read_only", "error": "This saved-study viewer is read-only."})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    class LocalServer(ThreadingHTTPServer):
        daemon_threads = True
        if ":" in host:
            address_family = socket.AF_INET6

    return LocalServer((host, port), Handler)


def serve(study_dir: str | Path, host: str = "127.0.0.1", port: int = 8797) -> None:
    """Serve saved results until interrupted; never trains or writes study files."""
    server = make_server(study_dir, host=host, port=port)
    address = f"[{host}]" if ":" in host else host
    print(f"Supermix Tether: http://{address}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

