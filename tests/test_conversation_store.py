"""Local SQLite conversation storage and session lifecycle."""
from pathlib import Path
from datetime import datetime
import sqlite3
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_store import (
    chat_identity_base,
    ConversationStore,
    LEGACY_UNNAMED_ARCHIVE,
    MessageInput,
)


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "nested" / "conversations.sqlite3"
        self.clock = [datetime(2026, 9, 23, 12, 0).timestamp()]
        self.store = ConversationStore(
            path=self.path, now=lambda: self.clock[0])
        self.addCleanup(self.store.close)

    def test_schema_version_and_private_file_permissions(self):
        with sqlite3.connect(self.path) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        self.assertEqual(version, 2)
        self.assertTrue({"sessions", "messages", "chat_bindings"} <= tables)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o077, 0)

    def test_resolve_session_binds_and_reuses_last_active(self):
        first = self.store.resolve_session(" Alice ")
        again = self.store.resolve_session("Alice")

        self.assertEqual(first.id, again.id)
        self.assertEqual(first.chat_key, "Alice")
        self.assertEqual(first.name, "Alice · 2026-09-23")
        self.assertEqual(self.store.active_session("Alice").id, first.id)

    def test_multiple_sessions_switch_and_keep_chat_boundary(self):
        first = self.store.resolve_session("Alice")
        second = self.store.create_session("Alice", "Project B")
        self.assertEqual(self.store.active_session("Alice").id, second.id)

        self.store.set_active_session("Alice", first.id)
        self.assertEqual(self.store.active_session("Alice").id, first.id)
        self.assertEqual(
            [session.name for session in self.store.list_sessions("Alice")],
            ["Alice · 2026-09-23", "Project B"])

        other = self.store.resolve_session("Bob")
        with self.assertRaises(ValueError):
            self.store.set_active_session("Alice", other.id)

    def test_truncated_group_count_resolves_to_existing_chat(self):
        existing = self.store.resolve_session("小海（9")

        for variant in ("小海（", "小海", "小海②"):
            resolved = self.store.resolve_session(variant)
            self.assertEqual(resolved.id, existing.id)
            self.assertEqual(
                self.store.resolve_chat_key(variant), "小海（9")

        self.assertEqual(chat_identity_base("小海②"), "小海")

    def test_existing_ocr_aliases_are_consolidated_without_losing_sessions(self):
        first = self.store.resolve_session("小海（9")
        self.store.append_messages(
            first.id, [MessageInput("them", None, "old")])
        self.store.close()
        timestamp = self.clock[0] + 10
        alias_id = "alias-session"
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT INTO sessions (
                       id, chat_key, name, created_at, updated_at, last_active_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (alias_id, "小海②", "小海② · 2026-09-23",
                 timestamp, timestamp, timestamp))
            db.execute(
                """INSERT INTO messages (
                       session_id, ordinal, side, sender, text,
                       observed_at, segment_id
                   ) VALUES (?, 1, 'them', NULL, 'new', ?, 'alias')""",
                (alias_id, timestamp))
            db.execute(
                """INSERT INTO chat_bindings (
                       chat_key, active_session_id, updated_at
                   ) VALUES (?, ?, ?)""",
                ("小海②", alias_id, timestamp))
            db.commit()

        reopened = ConversationStore(
            self.path, now=lambda: self.clock[0])
        self.store = reopened
        self.addCleanup(reopened.close)

        sessions = reopened.list_sessions("小海")
        self.assertEqual({row.id for row in sessions}, {first.id, alias_id})
        self.assertEqual(
            {row.chat_key for row in sessions}, {"小海（9"})
        self.assertEqual(
            reopened.active_session("小海").id, alias_id)
        self.assertEqual(
            sum(reopened.message_count(row.id) for row in sessions), 2)

    def test_v1_unnamed_history_is_archived_without_deletion(self):
        path = Path(self.tempdir.name) / "legacy.sqlite3"
        legacy = ConversationStore(path, now=lambda: self.clock[0])
        session = legacy.resolve_session("未命名聊天")
        legacy.append_messages(
            session.id, [MessageInput("them", None, "mixed")])
        legacy.close()
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA user_version = 1")
            db.commit()

        migrated = ConversationStore(path, now=lambda: self.clock[0])
        self.addCleanup(migrated.close)

        rows = migrated.list_sessions(include_deleted=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].chat_key, LEGACY_UNNAMED_ARCHIVE)
        self.assertIn("旧混合记录", rows[0].name)
        self.assertEqual(migrated.message_count(rows[0].id), 1)

    def test_change_token_tracks_sessions_messages_and_names(self):
        original = self.store.change_token()
        session = self.store.resolve_session("Alice")
        after_session = self.store.change_token()
        self.assertNotEqual(original, after_session)

        self.store.append_messages(
            session.id, [MessageInput("them", "Alice", "hello")])
        after_message = self.store.change_token()
        self.assertNotEqual(after_session, after_message)

        self.store.rename_session(session.id, "Other")
        self.assertNotEqual(after_message, self.store.change_token())

    def test_append_messages_preserves_both_sides_sender_and_order(self):
        session = self.store.resolve_session("Group")
        added = self.store.append_messages(session.id, [
            MessageInput(side="them", sender="Alice", text="在吗"),
            MessageInput(side="me", sender=None, text="在"),
            MessageInput(side="unknown", sender=None, text="系统提示"),
        ])

        self.assertEqual([message.ordinal for message in added], [1, 2, 3])
        recent = self.store.recent_messages(session.id)
        self.assertEqual(
            [(message.side, message.sender, message.text)
             for message in recent],
            [("them", "Alice", "在吗"),
             ("me", None, "在"),
             ("unknown", None, "系统提示")])
        self.assertEqual(len({message.segment_id for message in recent}), 1)

    def test_trashing_active_session_rebinds_or_creates_replacement(self):
        first = self.store.resolve_session("Alice")
        second = self.store.create_session("Alice", "Project B")

        replacement = self.store.trash_session(second.id)
        self.assertEqual(replacement.id, first.id)
        self.assertEqual(self.store.active_session("Alice").id, first.id)
        self.assertIsNotNone(
            self.store.get_session(second.id, include_deleted=True).deleted_at)

        only = self.store.resolve_session("Solo")
        replacement = self.store.trash_session(only.id)
        self.assertNotEqual(replacement.id, only.id)
        self.assertEqual(replacement.chat_key, "Solo")
        self.assertEqual(self.store.active_session("Solo").id, replacement.id)

    def test_restore_rename_and_permanent_delete_cascade(self):
        session = self.store.resolve_session("Alice")
        message = self.store.append_messages(
            session.id, [MessageInput("them", "Alice", "hello")])[0]

        self.store.trash_session(session.id)
        restored = self.store.restore_session(session.id)
        self.assertIsNone(restored.deleted_at)
        renamed = self.store.rename_session(session.id, "Restored")
        self.assertEqual(renamed.name, "Restored")

        self.store.trash_session(session.id)
        self.store.delete_session_permanently(session.id)
        self.assertIsNone(self.store.get_session(
            session.id, include_deleted=True))
        with sqlite3.connect(self.path) as db:
            count = db.execute(
                "SELECT count(*) FROM messages WHERE id = ?",
                (message.id,)).fetchone()[0]
        self.assertEqual(count, 0)

    def test_purge_removes_sessions_after_thirty_days(self):
        session = self.store.resolve_session("Alice")
        self.store.append_messages(
            session.id, [MessageInput("them", "Alice", "old")])
        self.store.trash_session(session.id)

        self.assertEqual(self.store.purge_expired(
            retention_days=30,
            now=self.clock[0] + 30 * 86400 - 1), 0)
        self.assertEqual(self.store.purge_expired(
            retention_days=30,
            now=self.clock[0] + 30 * 86400 + 1), 1)
        self.assertIsNone(self.store.get_session(
            session.id, include_deleted=True))

    def test_summary_update_requires_message_from_same_session(self):
        session = self.store.resolve_session("Alice")
        messages = self.store.append_messages(session.id, [
            MessageInput("them", "Alice", "one"),
            MessageInput("me", None, "two"),
        ])
        updated = self.store.update_summary(
            session.id, "Alice asked; I answered.", messages[0].id, version=2)
        self.assertEqual(updated.summary_text, "Alice asked; I answered.")
        self.assertEqual(updated.summary_until_message_id, messages[0].id)
        self.assertEqual(updated.summary_version, 2)

        other = self.store.resolve_session("Bob")
        other_message = self.store.append_messages(
            other.id, [MessageInput("them", "Bob", "wrong")])[0]
        with self.assertRaises(ValueError):
            self.store.update_summary(
                session.id, "bad boundary", other_message.id)

    def test_mark_observation_gap_is_durable(self):
        session = self.store.resolve_session("Alice")
        self.store.mark_observation_gap(session.id)
        self.assertTrue(self.store.get_session(
            session.id).has_observation_gap)


if __name__ == "__main__":
    unittest.main()
