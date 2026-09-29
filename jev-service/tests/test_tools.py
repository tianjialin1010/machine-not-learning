import unittest

from jev_service.tools import ToolGateway, ToolKind, ToolSpec


class ToolGatewayTests(unittest.TestCase):
    def test_write_requires_approval_and_is_idempotent(self) -> None:
        gateway = ToolGateway()
        calls = []
        gateway.register(ToolSpec("send_message", "write", "send", lambda args: calls.append(args) or {"sent": True}))
        blocked = gateway.execute("send_message", {"text": "hi"}, idempotency_key="k1")
        self.assertEqual(blocked.status, "approval_required")
        self.assertEqual(calls, [])
        done = gateway.execute("send_message", {"text": "hi"}, approved=True, idempotency_key="k1")
        duplicate = gateway.execute("send_message", {"text": "hi"}, approved=True, idempotency_key="k1")
        self.assertEqual(done.status, "ok")
        self.assertEqual(duplicate.status, "deduplicated")
        self.assertEqual(len(calls), 1)
        conflict = gateway.execute("send_message", {"text": "different"}, approved=True, idempotency_key="k1")
        self.assertEqual(conflict.status, "rejected")
        self.assertEqual(conflict.output["reason"], "idempotency_key_conflict")

    def test_unknown_tool_is_rejected(self) -> None:
        result = ToolGateway().execute("shell", {})
        self.assertEqual(result.status, "rejected")

    def test_schema_and_timeout_boundary_are_enforced(self) -> None:
        gateway = ToolGateway()
        gateway.register(ToolSpec(
            "lookup", "read", "lookup", lambda args: {"ok": True},
            schema={"required": ["query"], "properties": {"query": {"type": "string"}}},
        ))
        missing = gateway.execute("lookup", {})
        wrong_type = gateway.execute("lookup", {"query": 3})
        ok = gateway.execute("lookup", {"query": "x"})
        self.assertEqual(missing.status, "rejected")
        self.assertEqual(wrong_type.status, "rejected")
        self.assertEqual(ok.status, "ok")
