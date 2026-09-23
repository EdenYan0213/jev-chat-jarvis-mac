"""Local-only persistent sessions and message history."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid


APP_SUPPORT_DIR = (
    Path.home() / "Library" / "Application Support" / "jev-jarvis")
DEFAULT_DB_PATH = APP_SUPPORT_DIR / "conversations.sqlite3"
SCHEMA_VERSION = 1
VALID_SIDES = {"them", "me", "unknown"}


@dataclass(frozen=True)
class SessionRecord:
    id: str
    chat_key: str
    name: str
    created_at: float
    updated_at: float
    last_active_at: float
    deleted_at: float | None
    summary_text: str
    summary_until_message_id: int | None
    summary_version: int
    has_observation_gap: bool


@dataclass(frozen=True)
class MessageInput:
    side: str
    sender: str | None
    text: str


@dataclass(frozen=True)
class StoredMessage:
    id: int
    session_id: str
    ordinal: int
    side: str
    sender: str | None
    text: str
    observed_at: float
    segment_id: str


def normalize_chat_key(value: str) -> str:
    return " ".join(str(value or "").split()) or "未命名聊天"


class ConversationStore:
    """Thread-safe SQLite boundary for all conversation state."""

    def __init__(self, path: str | Path | None = None, now=None):
        raw_path = path if path is not None else DEFAULT_DB_PATH
        self._memory = str(raw_path) == ":memory:"
        self.path = Path(raw_path) if not self._memory else Path(":memory:")
        self._now = now or time.time
        self._lock = threading.RLock()
        self._closed = False

        if not self._memory:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
        self._db = sqlite3.connect(
            str(raw_path), check_same_thread=False, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA busy_timeout = 5000")
            if not self._memory:
                self._db.execute("PRAGMA journal_mode = WAL")
                self._db.execute("PRAGMA synchronous = NORMAL")
            self._migrate()
        if not self._memory:
            os.chmod(self.path, 0o600)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def _migrate(self) -> None:
        version = int(self._db.execute(
            "PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"conversation database version {version} is newer than "
                f"supported version {SCHEMA_VERSION}")
        if version == 0:
            with self._db:
                self._db.executescript("""
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY,
                        chat_key TEXT NOT NULL,
                        name TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        last_active_at REAL NOT NULL,
                        deleted_at REAL,
                        summary_text TEXT NOT NULL DEFAULT '',
                        summary_until_message_id INTEGER,
                        summary_version INTEGER NOT NULL DEFAULT 1,
                        has_observation_gap INTEGER NOT NULL DEFAULT 0
                            CHECK (has_observation_gap IN (0, 1))
                    );

                    CREATE INDEX sessions_chat_active
                        ON sessions(chat_key, deleted_at, last_active_at DESC);

                    CREATE TABLE messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL
                            REFERENCES sessions(id) ON DELETE CASCADE,
                        ordinal INTEGER NOT NULL,
                        side TEXT NOT NULL
                            CHECK (side IN ('them', 'me', 'unknown')),
                        sender TEXT,
                        text TEXT NOT NULL,
                        observed_at REAL NOT NULL,
                        segment_id TEXT NOT NULL,
                        UNIQUE(session_id, ordinal)
                    );

                    CREATE INDEX messages_session_ordinal
                        ON messages(session_id, ordinal);

                    CREATE TABLE chat_bindings (
                        chat_key TEXT PRIMARY KEY,
                        active_session_id TEXT NOT NULL
                            REFERENCES sessions(id) ON DELETE CASCADE,
                        updated_at REAL NOT NULL
                    );

                    PRAGMA user_version = 1;
                """)
            version = 1
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"failed to migrate conversation database to {SCHEMA_VERSION}")

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionRecord | None:
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            chat_key=row["chat_key"],
            name=row["name"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_active_at=float(row["last_active_at"]),
            deleted_at=(None if row["deleted_at"] is None
                        else float(row["deleted_at"])),
            summary_text=row["summary_text"] or "",
            summary_until_message_id=row["summary_until_message_id"],
            summary_version=int(row["summary_version"]),
            has_observation_gap=bool(row["has_observation_gap"]),
        )

    @staticmethod
    def _message(row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            id=int(row["id"]),
            session_id=row["session_id"],
            ordinal=int(row["ordinal"]),
            side=row["side"],
            sender=row["sender"],
            text=row["text"],
            observed_at=float(row["observed_at"]),
            segment_id=row["segment_id"],
        )

    def _fetch_session(self, session_id: str,
                       include_deleted: bool = False) -> sqlite3.Row | None:
        query = "SELECT * FROM sessions WHERE id = ?"
        params: tuple = (session_id,)
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        return self._db.execute(query, params).fetchone()

    def _default_name(self, chat_key: str, timestamp: float) -> str:
        base = f"{chat_key} · {datetime.fromtimestamp(timestamp):%Y-%m-%d}"
        names = {
            row[0] for row in self._db.execute(
                "SELECT name FROM sessions WHERE chat_key = ?", (chat_key,))
        }
        if base not in names:
            return base
        suffix = 2
        while f"{base} {suffix}" in names:
            suffix += 1
        return f"{base} {suffix}"

    def _insert_session(self, chat_key: str, name: str | None,
                        timestamp: float) -> SessionRecord:
        session_id = str(uuid.uuid4())
        session_name = (str(name).strip() if name is not None else "")
        if not session_name:
            session_name = self._default_name(chat_key, timestamp)
        self._db.execute(
            """INSERT INTO sessions (
                   id, chat_key, name, created_at, updated_at, last_active_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, chat_key, session_name,
             timestamp, timestamp, timestamp))
        self._bind(chat_key, session_id, timestamp)
        return self._session(self._fetch_session(session_id))

    def _bind(self, chat_key: str, session_id: str, timestamp: float) -> None:
        self._db.execute(
            """INSERT INTO chat_bindings (
                   chat_key, active_session_id, updated_at
               ) VALUES (?, ?, ?)
               ON CONFLICT(chat_key) DO UPDATE SET
                   active_session_id = excluded.active_session_id,
                   updated_at = excluded.updated_at""",
            (chat_key, session_id, timestamp))

    def resolve_session(self, chat_title: str) -> SessionRecord:
        chat_key = normalize_chat_key(chat_title)
        with self._lock, self._db:
            row = self._db.execute(
                """SELECT s.*
                     FROM chat_bindings b
                     JOIN sessions s ON s.id = b.active_session_id
                    WHERE b.chat_key = ? AND s.deleted_at IS NULL""",
                (chat_key,)).fetchone()
            if row is not None:
                return self._session(row)
            return self._insert_session(chat_key, None, float(self._now()))

    def create_session(self, chat_title: str,
                       name: str | None = None) -> SessionRecord:
        chat_key = normalize_chat_key(chat_title)
        with self._lock, self._db:
            return self._insert_session(
                chat_key, name, float(self._now()))

    def active_session(self, chat_title: str) -> SessionRecord | None:
        chat_key = normalize_chat_key(chat_title)
        with self._lock:
            row = self._db.execute(
                """SELECT s.*
                     FROM chat_bindings b
                     JOIN sessions s ON s.id = b.active_session_id
                    WHERE b.chat_key = ? AND s.deleted_at IS NULL""",
                (chat_key,)).fetchone()
            return self._session(row)

    def get_session(self, session_id: str,
                    include_deleted: bool = False) -> SessionRecord | None:
        with self._lock:
            return self._session(self._fetch_session(
                session_id, include_deleted=include_deleted))

    def list_sessions(self, chat_title: str | None = None,
                      include_deleted: bool = False) -> list[SessionRecord]:
        clauses = []
        params: list = []
        if chat_title is not None:
            clauses.append("chat_key = ?")
            params.append(normalize_chat_key(chat_title))
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        query = "SELECT * FROM sessions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY chat_key COLLATE NOCASE, created_at, rowid"
        with self._lock:
            return [
                self._session(row)
                for row in self._db.execute(query, params).fetchall()
            ]

    def set_active_session(self, chat_title: str,
                           session_id: str) -> SessionRecord:
        chat_key = normalize_chat_key(chat_title)
        timestamp = float(self._now())
        with self._lock, self._db:
            row = self._fetch_session(session_id)
            if row is None:
                raise KeyError(f"active session not found: {session_id}")
            if row["chat_key"] != chat_key:
                raise ValueError("session belongs to a different chat")
            self._db.execute(
                """UPDATE sessions
                      SET updated_at = ?, last_active_at = ?
                    WHERE id = ?""",
                (timestamp, timestamp, session_id))
            self._bind(chat_key, session_id, timestamp)
            return self._session(self._fetch_session(session_id))

    def rename_session(self, session_id: str, name: str) -> SessionRecord:
        clean_name = " ".join(str(name or "").split())
        if not clean_name:
            raise ValueError("session name cannot be empty")
        timestamp = float(self._now())
        with self._lock, self._db:
            if self._fetch_session(session_id, include_deleted=True) is None:
                raise KeyError(f"session not found: {session_id}")
            self._db.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (clean_name, timestamp, session_id))
            return self._session(self._fetch_session(
                session_id, include_deleted=True))

    def trash_session(self, session_id: str) -> SessionRecord | None:
        timestamp = float(self._now())
        with self._lock, self._db:
            row = self._fetch_session(session_id)
            if row is None:
                raise KeyError(f"active session not found: {session_id}")
            chat_key = row["chat_key"]
            binding = self._db.execute(
                "SELECT active_session_id FROM chat_bindings WHERE chat_key = ?",
                (chat_key,)).fetchone()
            self._db.execute(
                """UPDATE sessions
                      SET deleted_at = ?, updated_at = ?
                    WHERE id = ?""",
                (timestamp, timestamp, session_id))

            active_id = binding["active_session_id"] if binding else None
            if active_id != session_id:
                return self._session(self._fetch_session(active_id)) \
                    if active_id else None

            replacement = self._db.execute(
                """SELECT * FROM sessions
                    WHERE chat_key = ? AND deleted_at IS NULL
                    ORDER BY last_active_at DESC, created_at DESC, id DESC
                    LIMIT 1""",
                (chat_key,)).fetchone()
            if replacement is None:
                return self._insert_session(chat_key, None, timestamp)
            self._db.execute(
                """UPDATE sessions
                      SET updated_at = ?, last_active_at = ?
                    WHERE id = ?""",
                (timestamp, timestamp, replacement["id"]))
            self._bind(chat_key, replacement["id"], timestamp)
            return self._session(self._fetch_session(replacement["id"]))

    def restore_session(self, session_id: str) -> SessionRecord:
        timestamp = float(self._now())
        with self._lock, self._db:
            row = self._fetch_session(session_id, include_deleted=True)
            if row is None:
                raise KeyError(f"session not found: {session_id}")
            self._db.execute(
                """UPDATE sessions
                      SET deleted_at = NULL, updated_at = ?
                    WHERE id = ?""",
                (timestamp, session_id))
            return self._session(self._fetch_session(session_id))

    def delete_session_permanently(self, session_id: str) -> None:
        timestamp = float(self._now())
        with self._lock, self._db:
            row = self._fetch_session(session_id, include_deleted=True)
            if row is None:
                return
            chat_key = row["chat_key"]
            binding = self._db.execute(
                "SELECT active_session_id FROM chat_bindings WHERE chat_key = ?",
                (chat_key,)).fetchone()
            was_active = bool(
                binding and binding["active_session_id"] == session_id)
            self._db.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,))
            if was_active:
                replacement = self._db.execute(
                    """SELECT * FROM sessions
                        WHERE chat_key = ? AND deleted_at IS NULL
                        ORDER BY last_active_at DESC, created_at DESC, id DESC
                        LIMIT 1""",
                    (chat_key,)).fetchone()
                if replacement is None:
                    self._insert_session(chat_key, None, timestamp)
                else:
                    self._bind(chat_key, replacement["id"], timestamp)

    def purge_expired(self, retention_days: int = 30,
                      now: float | None = None) -> int:
        current = float(self._now() if now is None else now)
        cutoff = current - max(0, retention_days) * 86400
        with self._lock, self._db:
            rows = self._db.execute(
                """SELECT id FROM sessions
                    WHERE deleted_at IS NOT NULL AND deleted_at <= ?""",
                (cutoff,)).fetchall()
            if rows:
                self._db.executemany(
                    "DELETE FROM sessions WHERE id = ?",
                    [(row["id"],) for row in rows])
            return len(rows)

    @staticmethod
    def _clean_message(message: MessageInput) -> MessageInput:
        side = str(message.side)
        if side not in VALID_SIDES:
            raise ValueError(f"invalid message side: {side}")
        text = str(message.text or "").strip()
        if not text:
            raise ValueError("message text cannot be empty")
        sender = str(message.sender).strip() if message.sender else None
        return MessageInput(side=side, sender=sender or None, text=text)

    def append_messages(self, session_id: str,
                        messages: list[MessageInput],
                        segment_id: str | None = None) -> list[StoredMessage]:
        clean = [self._clean_message(message) for message in messages]
        if not clean:
            return []
        segment = segment_id or str(uuid.uuid4())
        timestamp = float(self._now())
        with self._lock, self._db:
            if self._fetch_session(session_id) is None:
                raise KeyError(f"active session not found: {session_id}")
            row = self._db.execute(
                """SELECT coalesce(max(ordinal), 0) AS ordinal
                     FROM messages WHERE session_id = ?""",
                (session_id,)).fetchone()
            ordinal = int(row["ordinal"])
            ids = []
            for message in clean:
                ordinal += 1
                cursor = self._db.execute(
                    """INSERT INTO messages (
                           session_id, ordinal, side, sender, text,
                           observed_at, segment_id
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, ordinal, message.side, message.sender,
                     message.text, timestamp, segment))
                ids.append(int(cursor.lastrowid))
            self._db.execute(
                """UPDATE sessions
                      SET updated_at = ?, last_active_at = ?
                    WHERE id = ?""",
                (timestamp, timestamp, session_id))
            placeholders = ",".join("?" for _ in ids)
            rows = self._db.execute(
                f"SELECT * FROM messages WHERE id IN ({placeholders}) "
                "ORDER BY ordinal",
                ids).fetchall()
            return [self._message(row) for row in rows]

    def list_messages(self, session_id: str, *,
                      after_id: int | None = None,
                      limit: int | None = None) -> list[StoredMessage]:
        clauses = ["session_id = ?"]
        params: list = [session_id]
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
        query = (
            "SELECT * FROM messages WHERE " + " AND ".join(clauses)
            + " ORDER BY ordinal")
        if limit is not None:
            if limit <= 0:
                return []
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
            return [self._message(row) for row in rows]

    def recent_messages(self, session_id: str, limit: int = 100,
                        after_id: int | None = None) -> list[StoredMessage]:
        if after_id is not None:
            return self.list_messages(
                session_id, after_id=after_id, limit=limit)
        if limit <= 0:
            return []
        with self._lock:
            rows = self._db.execute(
                """SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY ordinal DESC
                    LIMIT ?""",
                (session_id, limit)).fetchall()
            return [self._message(row) for row in reversed(rows)]

    def message_count(self, session_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT count(*) AS count FROM messages WHERE session_id = ?",
                (session_id,)).fetchone()
            return int(row["count"])

    def mark_observation_gap(self, session_id: str) -> None:
        timestamp = float(self._now())
        with self._lock, self._db:
            if self._fetch_session(session_id, include_deleted=True) is None:
                raise KeyError(f"session not found: {session_id}")
            self._db.execute(
                """UPDATE sessions
                      SET has_observation_gap = 1, updated_at = ?
                    WHERE id = ?""",
                (timestamp, session_id))

    def update_summary(self, session_id: str, text: str,
                       until_message_id: int,
                       version: int = 1) -> SessionRecord:
        timestamp = float(self._now())
        with self._lock, self._db:
            if self._fetch_session(session_id) is None:
                raise KeyError(f"active session not found: {session_id}")
            boundary = self._db.execute(
                """SELECT id FROM messages
                    WHERE id = ? AND session_id = ?""",
                (until_message_id, session_id)).fetchone()
            if boundary is None:
                raise ValueError(
                    "summary boundary does not belong to the session")
            self._db.execute(
                """UPDATE sessions
                      SET summary_text = ?,
                          summary_until_message_id = ?,
                          summary_version = ?,
                          updated_at = ?
                    WHERE id = ?""",
                (str(text).strip(), until_message_id,
                 int(version), timestamp, session_id))
            return self._session(self._fetch_session(session_id))
