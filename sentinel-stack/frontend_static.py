"""Serve the built PC frontend beside Sentinel's existing inference routes."""

from pathlib import Path
from urllib.parse import unquote, urlsplit


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}


def handle_frontend_get(handler, root) -> bool:
    """Return True only when this helper has completed the HTTP response."""
    path = urlsplit(handler.path).path
    if path == "/api/config":
        engine = handler.engine
        provider = engine.provider_name if engine is not None else None
        health = {
            **handler.meta,
            "ok": engine is not None,
            "provider": provider,
            "degraded": engine.degraded if engine is not None else True,
        }
        handler._json(200, {
            "mode": "native_service",
            "source": "sentinel",
            "upstream": "/v1",
            "decision_path": "/v1/decide",
            "health_path": "/healthz",
            "meta_path": "/v1/meta",
            "provider": provider,
            "upstream_health": health,
            "version": handler.meta.get("framework"),
        })
        return True

    if path in ("/dashboard", "/healthz", "/v1", "/api") or path.startswith(("/v1/", "/api/")):
        return False

    def not_found():
        handler._json(404, {"error": "not found"})
        return True

    try:
        path = unquote(path, errors="strict")
        if not path.startswith("/") or "\\" in path or "\x00" in path:
            return not_found()
        if any(part.startswith(".") for part in path.split("/")):
            return not_found()
        relative = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        content_type = CONTENT_TYPES.get(Path(relative).suffix.lower())
        if content_type is None:
            return not_found()
        frontend = (Path(root) / "frontend-current").resolve(strict=True)
        file_path = (frontend / relative).resolve(strict=True)
        if not file_path.is_relative_to(frontend) or not file_path.is_file():
            return not_found()
        body = file_path.read_bytes()
    except (OSError, ValueError, UnicodeError):
        return not_found()

    handler.send_response(200)
    handler.send_header("content-type", content_type)
    handler.send_header("content-length", str(len(body)))
    handler.send_header("cache-control", "no-cache")
    handler.send_header("x-content-type-options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)
    return True
