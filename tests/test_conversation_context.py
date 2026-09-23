"""OCR snapshot alignment and bounded model context."""
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_context import ContextBuilder, ConversationTracker
from conversation_store import ConversationStore, MessageInput


@dataclass
class Visible:
    text: str
    side: str
    sender: str | None = None


def them(text, sender="Alice"):
    return Visible(text=text, side="them", sender=sender)


def me(text):
    return Visible(text=text, side="me")


class ConversationContextTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = ConversationStore(
            Path(self.tempdir.name) / "conversations.sqlite3")
        self.addCleanup(self.store.close)
        self.tracker = ConversationTracker(self.store)

    def test_alignment_appends_only_unmatched_suffix(self):
        first = self.tracker.observe(
            "Alice", [them("A"), me("B"), them("C")])
        self.assertEqual(
            [message.text for message in first.appended], ["A", "B", "C"])

        again = self.tracker.observe("Alice", [me("B"), them("C")])
        self.assertEqual(again.appended, ())
        self.assertEqual(
            again.visible_ids,
            tuple(message.id for message in first.appended[1:]))

        extended = self.tracker.observe(
            "Alice", [me("B"), them("C"), them("D")])
        self.assertEqual(
            [message.text for message in extended.appended], ["D"])
        self.assertEqual(
            [message.text for message in self.store.recent_messages(
                first.session.id)],
            ["A", "B", "C", "D"])

    def test_repeated_identical_messages_remain_separate_occurrences(self):
        observed = self.tracker.observe(
            "Alice", [them("收到"), me("好"), them("收到")])

        self.assertEqual(
            [message.text for message in observed.appended],
            ["收到", "好", "收到"])
        self.assertEqual(len(set(observed.visible_ids)), 3)

    def test_scrolling_to_known_older_sequence_does_not_duplicate(self):
        first = self.tracker.observe(
            "Alice", [them("A"), me("B"), them("C"), me("D")])
        older = self.tracker.observe("Alice", [them("A"), me("B")])

        self.assertEqual(older.appended, ())
        self.assertEqual(
            older.visible_ids,
            (first.appended[0].id, first.appended[1].id))
        self.assertEqual(self.store.message_count(first.session.id), 4)

    def test_no_overlap_creates_new_segment_and_marks_gap(self):
        first = self.tracker.observe("Alice", [them("A"), me("B")])
        gap = self.tracker.observe("Alice", [them("完全不重叠")])

        self.assertTrue(gap.gap_detected)
        self.assertEqual(
            [message.text for message in gap.appended], ["完全不重叠"])
        self.assertNotEqual(
            first.appended[0].segment_id, gap.appended[0].segment_id)
        self.assertTrue(
            self.store.get_session(first.session.id).has_observation_gap)

    def test_empty_snapshot_changes_nothing(self):
        session = self.store.resolve_session("Alice")
        observed = self.tracker.observe("Alice", [])
        self.assertEqual(observed.session.id, session.id)
        self.assertEqual(observed.visible_ids, ())
        self.assertEqual(observed.appended, ())
        self.assertFalse(observed.gap_detected)

    def test_switching_session_routes_next_snapshot_to_selected_history(self):
        first = self.tracker.observe("Alice", [them("first")]).session
        second = self.store.create_session("Alice", "Second")
        observed = self.tracker.observe("Alice", [them("second")])
        self.assertEqual(observed.session.id, second.id)

        self.store.set_active_session("Alice", first.id)
        back = self.tracker.observe(
            "Alice", [them("first"), them("continued")])
        self.assertEqual(back.session.id, first.id)
        self.assertEqual(
            [message.text for message in back.appended], ["continued"])

    def test_context_uses_summary_recent_raw_and_excludes_target(self):
        session = self.store.resolve_session("Alice")
        messages = self.store.append_messages(session.id, [
            MessageInput("them", "Alice", "old question"),
            MessageInput("me", None, "old answer"),
            MessageInput("them", "Alice", "new question"),
            MessageInput("me", None, "draft response"),
        ])
        self.store.update_summary(
            session.id, "Alice asked an earlier question and I answered.",
            messages[1].id)

        rendered = ContextBuilder(
            self.store, hard_limit=400, recent_count=3).build(
                session.id, exclude_message_id=messages[2].id)

        self.assertIn("会话摘要：", rendered.text)
        self.assertIn(
            "Alice asked an earlier question and I answered.", rendered.text)
        self.assertIn("我: draft response", rendered.text)
        self.assertNotIn("new question", rendered.text)
        self.assertNotIn("old question", rendered.text)
        self.assertLessEqual(len(rendered.text), 400)
        self.assertEqual(rendered.message_count, 1)

    def test_context_labels_group_sender_and_unknown_direction(self):
        session = self.store.resolve_session("Group")
        self.store.append_messages(session.id, [
            MessageInput("them", "王总", "今天确认"),
            MessageInput("me", None, "收到"),
            MessageInput("unknown", None, "系统提示"),
        ])

        text = ContextBuilder(self.store).build(session.id).text
        self.assertIn("王总: 今天确认", text)
        self.assertIn("我: 收到", text)
        self.assertIn("方向未确认: 系统提示", text)

    def test_hard_limit_prunes_old_raw_before_summary(self):
        session = self.store.resolve_session("Alice")
        messages = self.store.append_messages(session.id, [
            MessageInput("them", "Alice", f"message-{index}-" + "x" * 35)
            for index in range(8)
        ])
        self.store.update_summary(
            session.id, "important-summary-" + "s" * 30, messages[1].id)

        rendered = ContextBuilder(
            self.store, hard_limit=190, recent_count=3).build(session.id)

        self.assertLessEqual(len(rendered.text), 190)
        self.assertIn("important-summary", rendered.text)
        self.assertIn("message-7", rendered.text)
        self.assertNotIn("message-2", rendered.text)
        self.assertTrue(rendered.truncated)
        self.assertGreaterEqual(rendered.message_count, 3)


if __name__ == "__main__":
    unittest.main()
