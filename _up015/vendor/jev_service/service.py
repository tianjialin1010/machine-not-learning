from __future__ import annotations

from typing import Any

from .engine import DecisionEngine
from .models import DecisionRequest
from .errors import ProviderError
from . import __version__


class Service:
    def __init__(self, engine: DecisionEngine | None = None) -> None:
        self.engine = engine or DecisionEngine()

    def handle(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            if path == "/v1/decide":
                result = self.engine.decide(DecisionRequest.from_dict(payload))
                return 200, result.to_dict()
            if path == "/v1/replay":
                raw_requests = payload.get("requests", [])
                if not isinstance(raw_requests, list):
                    raise ValueError("requests must be an array")
                requests = [DecisionRequest.from_dict(item) for item in raw_requests]
                return 200, self.engine.replay(requests)
            if path == "/v1/memory/feedback":
                record_id = str(payload.get("record_id", ""))
                action = str(payload.get("action", ""))
                if not record_id:
                    raise ValueError("record_id is required")
                record = self.engine.memory.feedback(record_id, action, payload.get("content"))
                if record is None:
                    return 404, {"error": "memory record not found"}
                return 200, {"record": record.to_dict()}
            if path == "/healthz":
                return 200, {"ok": True, "version": __version__, "provider": self.engine.provider_name, "degraded": self.engine.degraded, "model_connectivity": "not_checked_by_healthz"}
            return 404, {"error": "not found"}
        except ValueError as exc:
            return 422, {"error": str(exc)}
        except ProviderError as exc:
            return 502, {"error": str(exc), "code": exc.code, "provider_status": exc.status_code, "retryable": exc.retryable}
        except Exception as exc:
            return 500, {"error": "internal error", "detail": str(exc)}
