"""Bounded, per-chat long-term memory backed by SQLite."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)
_KEY_RE = re.compile(r"^[a-zA-Z0-9:_-]{2,80}$")


@dataclass(slots=True)
class MemoryEntry:
    key: str
    owner_user_id: int | None
    subject: str
    category: str
    content: str
    importance: int
    updated_at: str
    evidence_count: int = 1
    last_confirmed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Stores only compact, useful facts and enforces hard per-chat limits.

    Existing records are updated by stable key. When limits are exceeded,
    lower-importance and older records are removed first.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        max_chars: int = 4000,
        max_items: int = 30,
        max_changes_per_batch: int = 8,
    ) -> None:
        self.connection = connection
        self.max_chars = max(500, max_chars)
        self.max_items = max(1, max_items)
        self.max_changes_per_batch = max(1, max_changes_per_batch)
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    memory_key TEXT NOT NULL,
                    owner_user_id INTEGER,
                    subject TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    updated_at TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    last_confirmed_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(chat_id, memory_key)
                )
                """
            )
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(chat_memory)")}
            if "evidence_count" not in columns:
                self.connection.execute("ALTER TABLE chat_memory ADD COLUMN evidence_count INTEGER NOT NULL DEFAULT 1")
            if "last_confirmed_at" not in columns:
                self.connection.execute("ALTER TABLE chat_memory ADD COLUMN last_confirmed_at TEXT NOT NULL DEFAULT ''")
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_memory_priority "
                "ON chat_memory(chat_id, importance DESC, updated_at DESC)"
            )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        owner = row["owner_user_id"]
        return MemoryEntry(
            key=str(row["memory_key"]),
            owner_user_id=int(owner) if owner is not None else None,
            subject=str(row["subject"]),
            category=str(row["category"]),
            content=str(row["content"]),
            importance=int(row["importance"]),
            updated_at=str(row["updated_at"]),
            evidence_count=int(row["evidence_count"]) if "evidence_count" in row.keys() else 1,
            last_confirmed_at=str(row["last_confirmed_at"]) if "last_confirmed_at" in row.keys() else str(row["updated_at"]),
        )

    @staticmethod
    def _entry_size(entry: MemoryEntry) -> int:
        # Count the actual compact JSON representation, including timestamps and keys.
        return len(
            json.dumps(entry.to_dict(), ensure_ascii=False, separators=(",", ":"))
        )

    def get(self, chat_id: int) -> list[MemoryEntry]:
        rows = self.connection.execute(
            """
            SELECT memory_key, owner_user_id, subject, category, content,
                   importance, updated_at, evidence_count, last_confirmed_at
            FROM chat_memory
            WHERE chat_id = ?
            ORDER BY importance DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (chat_id, self.max_items),
        ).fetchall()

        result: list[MemoryEntry] = []
        # Reserve two characters for [] and one comma between adjacent entries.
        used = 2
        for row in rows:
            entry = self._row_to_entry(row)
            size = self._entry_size(entry) + (1 if result else 0)
            if used + size > self.max_chars:
                continue
            result.append(entry)
            used += size
        return result

    def to_prompt(self, chat_id: int) -> str:
        entries = self.get(chat_id)
        if not entries:
            return "[]"
        return json.dumps(
            [entry.to_dict() for entry in entries],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def apply_changes(
        self,
        chat_id: int,
        changes: list[Any],
        allowed_user_ids: set[int],
    ) -> int:
        """Apply validated model suggestions; return number of changed rows."""
        changed = 0
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        existing_rows = self.connection.execute(
            "SELECT memory_key, owner_user_id FROM chat_memory WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()
        existing = {
            str(row["memory_key"]): (
                int(row["owner_user_id"])
                if row["owner_user_id"] is not None
                else None
            )
            for row in existing_rows
        }

        # Apply more important updates first; avoid filling the store with trivia.
        ordered = sorted(
            changes,
            key=lambda item: int(getattr(item, "importance", 3) or 3),
            reverse=True,
        )[: self.max_changes_per_batch]

        with self.connection:
            for item in ordered:
                action = str(getattr(item, "action", "upsert"))
                key = str(getattr(item, "key", "")).strip()
                if not _KEY_RE.fullmatch(key):
                    continue

                owner_value = getattr(item, "owner_user_id", None)
                owner_id = int(owner_value) if owner_value is not None else None

                if action == "delete":
                    # Personal notes may only be deleted in a batch containing
                    # their owner. Shared group notes have owner_id=None.
                    if key not in existing or existing[key] != owner_id:
                        continue
                    if owner_id is not None and owner_id not in allowed_user_ids:
                        continue
                    cursor = self.connection.execute(
                        "DELETE FROM chat_memory WHERE chat_id = ? AND memory_key = ?",
                        (chat_id, key),
                    )
                    changed += cursor.rowcount
                    existing.pop(key, None)
                    continue

                if action != "upsert":
                    continue
                if owner_id is not None and owner_id not in allowed_user_ids:
                    continue
                if owner_id is None and str(getattr(item, "category", "")) != "group":
                    # Group-wide notes must be explicitly categorized as such.
                    continue
                if key in existing and existing[key] != owner_id:
                    # A stable key cannot be reassigned to a different person.
                    continue

                subject = str(getattr(item, "subject", "") or "")[:80].strip()
                category = str(getattr(item, "category", "other"))[:24].strip()
                content = str(getattr(item, "content", "") or "").strip()
                content = content[:320]
                importance = max(1, min(5, int(getattr(item, "importance", 3) or 3)))
                if not content or len(content) < 5:
                    continue

                evidence_level = str(getattr(item, "evidence_level", "tentative"))
                old = self.connection.execute(
                    """SELECT owner_user_id, subject, category, content, importance,
                              evidence_count, last_confirmed_at
                       FROM chat_memory WHERE chat_id = ? AND memory_key = ?""",
                    (chat_id, key),
                ).fetchone()
                evidence_count = 2 if evidence_level == "strong" else 1
                last_confirmed_at = now
                if old is not None:
                    same_fact = old["category"] == category and old["subject"] == subject
                    if same_fact and evidence_level in {"repeated", "strong"}:
                        evidence_count = int(old["evidence_count"] or 1) + 1
                        last_confirmed_at = now
                        # Repeated independent evidence stabilizes the memory.
                        importance = max(importance, min(5, int(old["importance"]) + 1))
                    else:
                        evidence_count = int(old["evidence_count"] or 1)
                        last_confirmed_at = str(old["last_confirmed_at"] or old["updated_at"])
                    if (old["subject"] == subject and old["category"] == category
                        and old["content"] == content and int(old["importance"]) == importance
                        and evidence_level == "tentative"):
                        continue

                self.connection.execute(
                    """
                    INSERT INTO chat_memory
                        (chat_id, memory_key, owner_user_id, subject, category,
                         content, importance, updated_at, evidence_count, last_confirmed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, memory_key) DO UPDATE SET
                        owner_user_id = excluded.owner_user_id,
                        subject = excluded.subject,
                        category = excluded.category,
                        content = excluded.content,
                        importance = excluded.importance,
                        updated_at = excluded.updated_at,
                        evidence_count = excluded.evidence_count,
                        last_confirmed_at = excluded.last_confirmed_at
                    """,
                    (
                        chat_id,
                        key,
                        owner_id,
                        subject,
                        category,
                        content,
                        importance,
                        now,
                        evidence_count,
                        last_confirmed_at,
                    ),
                )
                existing[key] = owner_id
                changed += 1

        self._cleanup_stale(chat_id, now_dt)
        self._trim(chat_id)
        return changed

    def _trim(self, chat_id: int) -> None:
        rows = self.connection.execute(
            """
            SELECT id, memory_key, owner_user_id, subject, category, content,
                   importance, updated_at, evidence_count, last_confirmed_at
            FROM chat_memory
            WHERE chat_id = ?
            ORDER BY importance DESC, updated_at DESC, id DESC
            """,
            (chat_id,),
        ).fetchall()

        keep_ids: list[int] = []
        used = 2
        for row in rows:
            entry = self._row_to_entry(row)
            size = self._entry_size(entry) + (1 if keep_ids else 0)
            if len(keep_ids) >= self.max_items or used + size > self.max_chars:
                continue
            keep_ids.append(int(row["id"]))
            used += size

        if rows:
            placeholders = ",".join("?" for _ in keep_ids)
            with self.connection:
                if keep_ids:
                    self.connection.execute(
                        f"DELETE FROM chat_memory WHERE chat_id = ? AND id NOT IN ({placeholders})",
                        [chat_id, *keep_ids],
                    )
                else:
                    self.connection.execute(
                        "DELETE FROM chat_memory WHERE chat_id = ?", (chat_id,)
                    )

    def _cleanup_stale(self, chat_id: int, now: datetime) -> None:
        """Decay weak, unconfirmed inferences; preserve repeatedly confirmed memories."""
        cutoff = (now - timedelta(days=120)).isoformat()
        very_old = (now - timedelta(days=240)).isoformat()
        with self.connection:
            # Tentative low-value notes expire first. Strong/repeated notes are stable.
            self.connection.execute(
                "DELETE FROM chat_memory WHERE chat_id = ? AND importance <= 2 "
                "AND evidence_count <= 1 AND COALESCE(last_confirmed_at, updated_at) < ?",
                (chat_id, cutoff),
            )
            # Old medium-value notes are removed only if they were never reinforced.
            self.connection.execute(
                "DELETE FROM chat_memory WHERE chat_id = ? AND importance = 3 "
                "AND evidence_count <= 1 AND COALESCE(last_confirmed_at, updated_at) < ?",
                (chat_id, very_old),
            )

    def forget_user(self, chat_id: int, user_id: int) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM chat_memory WHERE chat_id = ? AND owner_user_id = ?",
                (chat_id, user_id),
            )
        return cursor.rowcount

    def close(self) -> None:
        # The SQLite connection is owned by MessageDatabase; do not close it here.
        return
