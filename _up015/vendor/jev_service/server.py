from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .service import Service


class Handler(BaseHTTPRequestHandler):
    service: Service

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._write(404, {"error": "not found"})
            return
        status, payload = self.service.handle(self.path, {})
        self._write(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._write(422, {"error": str(exc)})
            return
        status, response = self.service.handle(self.path, payload)
        self._write(status, response)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JEV decision service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"JEV decision service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def create_server(host: str = "127.0.0.1", port: int = 8787, service: Service | None = None) -> ThreadingHTTPServer:
    class ApplicationHandler(Handler):
        pass
    ApplicationHandler.service = service or Service()
    return ThreadingHTTPServer((host, port), ApplicationHandler)


if __name__ == "__main__":
    main()
