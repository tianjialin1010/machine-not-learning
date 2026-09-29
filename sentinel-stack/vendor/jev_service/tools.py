from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
import json
import threading
from typing import Any, Callable, Literal

from .memory import MemoryService
from .models import new_id


ToolKind = Literal["read", "suggest", "write"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: ToolKind
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    schema: dict[str, Any] | None = None
    timeout_seconds: float = 10.0
    max_retries: int = 0


@dataclass
class ToolResult:
    tool_call_id: str
    tool: str
    status: str
    output: dict[str, Any]
    requires_approval: bool = False
    idempotency_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "status": self.status,
            "output": self.output,
            "requires_approval": self.requires_approval,
            "idempotency_key": self.idempotency_key,
        }


class ToolGateway:
    """Allow-list, approval and idempotency boundary for future external tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._completed: dict[str, tuple[str, ToolResult]] = {}
        self._lock = threading.RLock()
        self.audit_log: list[dict[str, Any]] = []

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        approved: bool = False,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        with self._lock:
            if not isinstance(arguments, dict):
                return self._record(ToolResult(new_id("tool"), name, "rejected", {"reason": "arguments_must_be_object"}, False, idempotency_key))
            if name not in self._tools:
                return self._record(ToolResult(new_id("tool"), name, "rejected", {"reason": "tool_not_allowlisted"}, False, idempotency_key))
            spec = self._tools[name]
            fingerprint = _fingerprint(name, arguments)
            if idempotency_key and idempotency_key in self._completed:
                previous_fingerprint, previous = self._completed[idempotency_key]
                if previous_fingerprint != fingerprint:
                    return self._record(ToolResult(new_id("tool"), name, "rejected", {"reason": "idempotency_key_conflict"}, False, idempotency_key))
                return self._record(ToolResult(previous.tool_call_id, name, "deduplicated", previous.output, previous.requires_approval, idempotency_key))
            invalid = _validate_arguments(arguments, spec.schema)
            if invalid:
                return self._record(ToolResult(new_id("tool"), name, "rejected", {"reason": invalid}, False, idempotency_key))
            if spec.kind == "write" and not approved:
                return self._record(ToolResult(new_id("tool"), name, "approval_required", {"reason": "side_effect_requires_approval"}, True, idempotency_key))
            try:
                output = self._invoke(spec, arguments)
                result = ToolResult(new_id("tool"), name, "ok", output, False, idempotency_key)
            except Exception as exc:
                result = ToolResult(new_id("tool"), name, "failed", {"error": str(exc)}, spec.kind == "write", idempotency_key)
            if idempotency_key and result.status == "ok":
                self._completed[idempotency_key] = (fingerprint, result)
            return self._record(result)

    def _record(self, result: ToolResult) -> ToolResult:
        self.audit_log.append(result.to_dict())
        return result

    def _invoke(self, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        attempts = 1 if spec.kind == "write" else max(1, spec.max_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(spec.handler, arguments)
            try:
                output = future.result(timeout=spec.timeout_seconds)
                executor.shutdown(wait=True)
                return output
            except FutureTimeoutError as exc:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                last_error = TimeoutError(f"tool timed out after {spec.timeout_seconds}s")
            except Exception as exc:
                executor.shutdown(wait=True)
                last_error = exc
            if attempt + 1 < attempts:
                continue
        raise last_error or RuntimeError("tool failed")


def build_default_gateway(memory: MemoryService) -> ToolGateway:
    gateway = ToolGateway()
    gateway.register(
        ToolSpec(
            name="evidence_search",
            kind="read",
            description="Search prior conversation evidence",
            handler=lambda args: {
                "evidence": [item.to_dict() for item in memory.retrieve(str(args["conversation_id"]), str(args["query"]))]
            },
        )
    )
    return gateway


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
    if schema is None:
        return None
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        return f"missing_required_arguments:{','.join(missing)}"
    properties = schema.get("properties", {})
    for key, definition in properties.items():
        if key not in arguments or not isinstance(definition, dict):
            continue
        expected = definition.get("type")
        actual = arguments[key]
        if expected == "string" and not isinstance(actual, str):
            return f"invalid_argument_type:{key}"
        if expected == "number" and (not isinstance(actual, (int, float)) or isinstance(actual, bool)):
            return f"invalid_argument_type:{key}"
        if expected == "object" and not isinstance(actual, dict):
            return f"invalid_argument_type:{key}"
    return None


def _fingerprint(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"tool": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
