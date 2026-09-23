"""OCR snapshot alignment and bounded model context."""
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_context import (
    ContextBuilder,
    ConversationTracker,
    SummaryWorker,
)
from conversation_store import ConversationStore, MessageInput
from generate import Generator


@dataclass
class Visible:
    text: str
    side: str
    sender: str | None = None


def them(text, sender="Alice"):
    return Visible(text=text, side="them", sender=sender)


def me(text):
    return Visible(text=text, side="me")


def unknown(text):
    return Visible(text=text, side="unknown")


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

    def test_sender_drift_and_unknown_side_do_not_duplicate_snapshot(self):
        first = self.tracker.observe(
            "Alice", [them("A", sender="15:00|"), me("B")])

        again = self.tracker.observe(
            "Alice", [unknown("A"), me("B")])

        self.assertEqual(again.appended, ())
        self.assertEqual(again.visible_ids, tuple(
            message.id for message in first.appended))
        self.assertEqual(self.store.message_count(first.session.id), 2)

    def test_mid_snapshot_overlap_appends_only_the_new_tail(self):
        first = self.tracker.observe(
            "Alice", [them("A"), me("B"), me("C"), them("D")])

        shifted = self.tracker.observe(
            "Alice", [unknown("A B"), me("C"), them("D"), me("E")])

        self.assertEqual(
            [message.text for message in shifted.appended], ["E"])
        self.assertEqual(shifted.visible_ids, (
            None,
            first.appended[2].id,
            first.appended[3].id,
            shifted.appended[0].id,
        ))
        self.assertEqual(
            [message.text for message in self.store.recent_messages(
                first.session.id)],
            ["A", "B", "C", "D", "E"])

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


class GeneratorSummaryTests(unittest.TestCase):
    def test_summary_request_uses_low_temperature_and_retention_prompt(self):
        generator = Generator()
        generator._call = Mock(return_value="  更新后的摘要  ")

        result = generator.summarize(
            "Alice 在等待确认。", "Alice: 明天下午能给吗\n我: 可以")

        self.assertEqual(result, "更新后的摘要")
        prompt = generator._call.call_args.args[0]
        self.assertIn("Alice 在等待确认。", prompt)
        self.assertIn("Alice: 明天下午能给吗", prompt)
        self.assertIn("人物关系", prompt)
        self.assertIn("未解决问题", prompt)
        self.assertEqual(
            generator._call.call_args.kwargs,
            {"max_tokens": 800, "temperature": 0.2})


class SummaryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = ConversationStore(
            Path(self.tempdir.name) / "conversations.sqlite3")
        self.addCleanup(self.store.close)
        self.session = self.store.resolve_session("Alice")

    def add_messages(self, count, width=40):
        return self.store.append_messages(self.session.id, [
            MessageInput(
                "them" if index % 2 == 0 else "me",
                "Alice" if index % 2 == 0 else None,
                f"message-{index}-" + "x" * width,
            )
            for index in range(count)
        ])

    def test_below_threshold_does_not_call_summarizer(self):
        self.add_messages(4, width=5)
        summarizer = Mock()
        worker = SummaryWorker(
            self.store, summarizer, soft_limit=10_000, recent_count=2)

        self.assertFalse(worker.run_once(self.session.id))
        summarizer.summarize.assert_not_called()

    def test_success_advances_boundary_but_keeps_recent_raw(self):
        messages = self.add_messages(7)
        summarizer = Mock()
        summarizer.summarize.return_value = "压缩后的摘要"
        worker = SummaryWorker(
            self.store, summarizer, soft_limit=100, recent_count=2)

        self.assertTrue(worker.run_once(self.session.id))

        updated = self.store.get_session(self.session.id)
        self.assertEqual(updated.summary_text, "压缩后的摘要")
        self.assertEqual(updated.summary_until_message_id, messages[-3].id)
        self.assertEqual(
            [message.id for message in self.store.list_messages(
                self.session.id,
                after_id=updated.summary_until_message_id)],
            [messages[-2].id, messages[-1].id])
        prompt_messages = summarizer.summarize.call_args.args[1]
        self.assertIn("message-0", prompt_messages)
        self.assertIn("message-4", prompt_messages)
        self.assertNotIn("message-5", prompt_messages)

    def test_failure_keeps_previous_summary_and_boundary(self):
        messages = self.add_messages(8)
        self.store.update_summary(
            self.session.id, "旧摘要", messages[1].id)
        summarizer = Mock()
        summarizer.summarize.side_effect = RuntimeError("offline")
        logs = []
        worker = SummaryWorker(
            self.store, summarizer, soft_limit=100, recent_count=2,
            logger=logs.append)

        self.assertFalse(worker.run_once(self.session.id))

        unchanged = self.store.get_session(self.session.id)
        self.assertEqual(unchanged.summary_text, "旧摘要")
        self.assertEqual(
            unchanged.summary_until_message_id, messages[1].id)
        self.assertEqual(len(logs), 1)
        self.assertNotIn("message-", logs[0])

    def test_interactive_generation_defers_summary(self):
        self.add_messages(7)
        summarizer = Mock()
        worker = SummaryWorker(
            self.store, summarizer, soft_limit=100, recent_count=2,
            interactive_busy=lambda: True)

        self.assertFalse(worker.run_once(self.session.id))
        summarizer.summarize.assert_not_called()
        self.assertEqual(
            self.store.get_session(
                self.session.id).summary_until_message_id,
            None)


if __name__ == "__main__":
    unittest.main()
