from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
import re
from typing import Iterable, Literal, Protocol

from .models import Evidence, MemoryWriteProposal, Turn, new_id


MemoryType = Literal["working", "episodic", "semantic", "procedural"]


@dataclass
class MemoryRecord:
    id: str
    conversation_id: str
    record_type: MemoryType
    content: str
    source: str
    confidence: float | None = None
    confirmed: bool = False
    created_at: str = ""
    expires_at: str | None = None
    deleted: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MemoryStore(Protocol):
    def append(self, record: MemoryRecord) -> MemoryRecord: ...

    def search(self, conversation_id: str, query: str, limit: int = 5) -> list[MemoryRecord]: ...

    def feedback(self, record_id: str, action: str, content: str | None = None) -> MemoryRecord | None: ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def append(self, record: MemoryRecord) -> MemoryRecord:
        self.records[record.id] = record
        return record

    def search(self, conversation_id: str, query: str, limit: int = 5) -> list[MemoryRecord]:
        terms = {term for term in re.findall(r"[\w\u4e00-\u9fff]+", query.lower()) if len(term) > 1}
        candidates: list[tuple[int, MemoryRecord]] = []
        for record in self.records.values():
            if record.deleted or record.conversation_id != conversation_id or _expired(record.expires_at):
                continue
            overlap = sum(1 for term in terms if term in record.content.lower())
            if overlap:
                candidates.append((overlap, record))
        candidates.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [record for _, record in candidates[:limit]]

    def feedback(self, record_id: str, action: str, content: str | None = None) -> MemoryRecord | None:
        record = self.records.get(record_id)
        if not record:
            return None
        if action == "delete":
            record.deleted = True
        elif action == "confirm":
            record.confirmed = True
        elif action == "correct" and content:
            record.content = content
            record.confirmed = True
        else:
            raise ValueError("feedback action must be confirm, correct, or delete")
        return record


class MemoryService:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or InMemoryMemoryStore()

    def retrieve(self, conversation_id: str, query: str, limit: int = 5) -> list[Evidence]:
        return [
            Evidence(id=record.id, content=record.content, source=record.source, score=0.5, occurred_at=record.created_at)
            for record in self.store.search(conversation_id, query, limit)
        ]

    def append_event(self, conversation_id: str, content: str, source: str = "conversation") -> MemoryRecord:
        return self.store.append(
            MemoryRecord(
                id=new_id("event"),
                conversation_id=conversation_id,
                record_type="episodic",
                content=content,
                source=source,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    def propose_write(
        self,
        conversation_id: str,
        content: str,
        record_type: Literal["semantic", "procedural"] = "semantic",
        source: str = "inference",
        confidence: float | None = None,
    ) -> MemoryWriteProposal:
        if record_type not in {"semantic", "procedural"}:
            raise ValueError("record_type must be semantic or procedural")
        if confidence is not None and not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        eligible_sources = {"explicit", "confirmed", "repeat"}
        eligible = source in eligible_sources
        reason = "accepted source policy" if eligible else "inferred facts require confirmation before persistence"
        if confidence is not None and confidence < 0.8:
            eligible = False
            reason = "confidence below persistence threshold"
        if eligible:
            self.store.append(
                MemoryRecord(
                    id=new_id("memory"),
                    conversation_id=conversation_id,
                    record_type=record_type,
                    content=content,
                    source=source,
                    confidence=confidence,
                    confirmed=source in {"explicit", "confirmed"},
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return MemoryWriteProposal(content, record_type, source, eligible, reason)

    def feedback(self, record_id: str, action: str, content: str | None = None) -> MemoryRecord | None:
        return self.store.feedback(record_id, action, content)


def _expired(expires_at: str | None) -> bool:
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at).timestamp() <= datetime.now(timezone.utc).timestamp()
    except ValueError:
        return True
