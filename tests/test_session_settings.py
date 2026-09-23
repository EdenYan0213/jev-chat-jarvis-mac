"""Cocoa-free session management model tests."""
from datetime import datetime
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_store import ConversationStore, MessageInput
from session_settings import SessionManagerModel


class SessionManagerModelTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "conversations.sqlite3"
        self.clock = [datetime(2026, 9, 23, 12, 0).timestamp()]
        self.store = ConversationStore(
            self.path, now=lambda: self.clock[0])
        self.addCleanup(self.store.close)
        self.model = SessionManagerModel(self.store)

    def test_active_rows_search_name_and_chat_case_insensitively(self):
        first = self.store.resolve_session("Alice Team")
        self.store.rename_session(first.id, "客户跟进")
        second = self.store.resolve_session("Bob")

        by_chat = self.model.active_rows(search="ALICE")
        by_name = self.model.active_rows(search="客户")
        all_sessions = self.model.active_rows()

        self.assertEqual([row.id for row in by_chat], [first.id])
        self.assertEqual([row.id for row in by_name], [first.id])
        self.assertEqual(
            {row.id for row in all_sessions}, {first.id, second.id})
        self.assertIsNone(by_name[0].deleted_at)
        self.assertIsNone(by_name[0].deletes_at)

    def test_detail_returns_summary_gap_and_complete_message_history(self):
        session = self.store.resolve_session("Alice")
        messages = self.store.append_messages(session.id, [
            MessageInput("them", "Alice", "第一条"),
            MessageInput("me", None, "第二条"),
            MessageInput("unknown", None, "第三条"),
        ])
        self.store.update_summary(
            session.id, "这是摘要", messages[1].id)
        self.store.mark_observation_gap(session.id)

        detail = self.model.detail(session.id)

        self.assertEqual(detail.row.chat_key, "Alice")
        self.assertEqual(detail.row.message_count, 3)
        self.assertEqual(detail.summary_text, "这是摘要")
        self.assertTrue(detail.has_observation_gap)
        self.assertEqual(
            [(message.side, message.sender, message.text)
             for message in detail.messages],
            [
                ("them", "Alice", "第一条"),
                ("me", None, "第二条"),
                ("unknown", None, "第三条"),
            ],
        )

    def test_rename_trash_restore_and_scheduled_deletion(self):
        session = self.store.resolve_session("Alice")
        self.model.rename(session.id, "重要会话")
        self.model.trash(session.id)

        trash = self.model.trash_rows()
        row = next(item for item in trash if item.id == session.id)
        self.assertEqual(row.name, "重要会话")
        self.assertEqual(row.deletes_at, row.deleted_at + 30 * 86400)

        self.model.restore(session.id)
        self.assertIn(
            session.id, [row.id for row in self.model.active_rows()])
        self.assertNotIn(
            session.id, [row.id for row in self.model.trash_rows()])

    def test_permanent_delete_cascades_messages(self):
        session = self.store.resolve_session("Alice")
        message = self.store.append_messages(
            session.id, [MessageInput("them", "Alice", "hello")])[0]
        self.model.trash(session.id)

        self.model.delete_permanently(session.id)

        self.assertIsNone(self.store.get_session(
            session.id, include_deleted=True))
        with sqlite3.connect(self.path) as db:
            count = db.execute(
                "SELECT count(*) FROM messages WHERE id = ?",
                (message.id,)).fetchone()[0]
        self.assertEqual(count, 0)

    def test_rows_are_sorted_by_recent_update(self):
        first = self.store.resolve_session("Alice")
        self.clock[0] += 10
        second = self.store.create_session("Alice", "Second")

        rows = self.model.active_rows()

        self.assertLess(
            [row.id for row in rows].index(second.id),
            [row.id for row in rows].index(first.id))


if __name__ == "__main__":
    unittest.main()
