"""Align visible OCR messages with durable history and render model context."""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Iterable

from conversation_store import (
    ConversationStore,
    MessageInput,
    SessionRecord,
    StoredMessage,
)


DEFAULT_SOFT_LIMIT = 8_000
DEFAULT_HARD_LIMIT = 12_000
DEFAULT_RECENT_COUNT = 20
RECENT_ALIGNMENT_LIMIT = 100
SUMMARY_INPUT_LIMIT = 8_000


@dataclass(frozen=True)
class Observation:
    session: SessionRecord
    visible_ids: tuple[int | None, ...]
    appended: tuple[StoredMessage, ...]
    gap_detected: bool


@dataclass(frozen=True)
class ContextEnvelope:
    text: str
    message_count: int
    source_chars: int
    truncated: bool


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _clean_sender(value: str | None) -> str | None:
    sender = _clean_text(value or "")
    return sender or None


def message_key(message) -> tuple[str, str, str]:
    return (
        str(message.side),
        _clean_sender(getattr(message, "sender", None)) or "",
        _clean_text(message.text),
    )


def _message_input(message) -> MessageInput:
    side, sender, text = message_key(message)
    return MessageInput(side=side, sender=sender or None, text=text)


def _alignment_key(message) -> tuple[str, str]:
    return str(message.side), _clean_text(message.text)


def _alignment_matches(left: tuple[str, str],
                       right: tuple[str, str]) -> bool:
    left_side, left_text = left
    right_side, right_text = right
    return (
        left_text == right_text
        and (
            left_side == right_side
            or "unknown" in (left_side, right_side)
        )
    )


def _sequence_matches(left: list[tuple[str, str]],
                      right: list[tuple[str, str]]) -> bool:
    return (
        len(left) == len(right)
        and all(
            _alignment_matches(a, b)
            for a, b in zip(left, right)
        )
    )


def _latest_subsequence(haystack: list[tuple], needle: list[tuple]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if _sequence_matches(
                haystack[start:start + len(needle)], needle):
            return start
    return None


def _suffix_overlap(haystack: list[tuple], needle: list[tuple]
                    ) -> tuple[int, int] | None:
    """Find the longest history suffix occurring anywhere in the visible snapshot."""
    max_size = min(len(haystack), len(needle))
    for size in range(max_size, 0, -1):
        suffix = haystack[-size:]
        for start in range(len(needle) - size, -1, -1):
            if _sequence_matches(
                    suffix, needle[start:start + size]):
                return size, start
    return None


class ConversationTracker:
    """Persist only the new suffix of each visible OCR snapshot."""

    def __init__(self, store: ConversationStore,
                 alignment_limit: int = RECENT_ALIGNMENT_LIMIT):
        self.store = store
        self.alignment_limit = max(1, int(alignment_limit))

    def observe(self, chat_title: str,
                visible_messages: Iterable) -> Observation:
        session = self.store.resolve_session(chat_title)
        visible = list(visible_messages)
        if not visible:
            return Observation(session, (), (), False)

        visible_keys = [_alignment_key(message) for message in visible]
        recent = self.store.recent_messages(
            session.id, limit=self.alignment_limit)
        recent_keys = [_alignment_key(message) for message in recent]

        existing_start = _latest_subsequence(recent_keys, visible_keys)
        if existing_start is not None:
            ids = tuple(
                message.id
                for message in recent[
                    existing_start:existing_start + len(visible)])
            return Observation(session, ids, (), False)

        overlap = _suffix_overlap(recent_keys, visible_keys)
        if overlap is not None:
            overlap_size, visible_start = overlap
            appended = self.store.append_messages(
                session.id,
                [
                    _message_input(message)
                    for message in visible[
                        visible_start + overlap_size:]
                ])
            ids = tuple(
                [None] * visible_start
                + [message.id for message in recent[-overlap_size:]]
                + [message.id for message in appended])
            return Observation(
                self.store.get_session(session.id),
                ids,
                tuple(appended),
                False,
            )

        gap_detected = bool(recent)
        appended = self.store.append_messages(
            session.id, [_message_input(message) for message in visible])
        if gap_detected:
            self.store.mark_observation_gap(session.id)
        return Observation(
            self.store.get_session(session.id),
            tuple(message.id for message in appended),
            tuple(appended),
            gap_detected,
        )


def _speaker(message: StoredMessage) -> str:
    if message.sender:
        return message.sender
    return {"me": "我", "them": "对方"}.get(
        message.side, "方向未确认")


def _line(message: StoredMessage) -> str:
    return f"{_speaker(message)}: {message.text}"


def _render(summary: str, lines: list[str]) -> str:
    parts = []
    if summary:
        parts.append(f"会话摘要：\n{summary}")
    if lines:
        parts.append("最近原文：\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[:limit - 1] + "…"


class ContextBuilder:
    """Render summary plus recent exact messages under a hard character limit."""

    def __init__(self, store: ConversationStore, *,
                 soft_limit: int = DEFAULT_SOFT_LIMIT,
                 hard_limit: int = DEFAULT_HARD_LIMIT,
                 recent_count: int = DEFAULT_RECENT_COUNT):
        self.store = store
        self.soft_limit = max(1, int(soft_limit))
        self.hard_limit = max(1, int(hard_limit))
        self.recent_count = max(1, int(recent_count))

    def build(self, session_id: str,
              exclude_message_id: int | None = None) -> ContextEnvelope:
        session = self.store.get_session(session_id, include_deleted=True)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        messages = self.store.list_messages(
            session_id, after_id=session.summary_until_message_id)
        if exclude_message_id is not None:
            messages = [
                message for message in messages
                if message.id != exclude_message_id
            ]
        lines = [_line(message) for message in messages]
        summary = session.summary_text.strip()
        source = _render(summary, lines)
        source_chars = len(source)
        truncated = False

        while (len(_render(summary, lines)) > self.hard_limit
               and len(lines) > self.recent_count):
            lines.pop(0)
            truncated = True

        text = _render(summary, lines)
        if len(text) > self.hard_limit and summary:
            raw = _render("", lines)
            fixed = len("会话摘要：\n") + (2 if raw else 0)
            summary = _clip(
                summary, self.hard_limit - len(raw) - fixed)
            text = _render(summary, lines)
            truncated = True

        while len(text) > self.hard_limit and len(lines) > 1:
            lines.pop(0)
            text = _render(summary, lines)
            truncated = True

        if len(text) > self.hard_limit and lines:
            summary = ""
            line_budget = self.hard_limit - len("最近原文：\n")
            lines = [_clip(lines[-1], line_budget)]
            text = _render("", lines)
            truncated = True

        if len(text) > self.hard_limit:
            text = _clip(text, self.hard_limit)
            truncated = True

        return ContextEnvelope(
            text=text,
            message_count=len(lines),
            source_chars=source_chars,
            truncated=truncated,
        )


class SummaryWorker:
    """Low-priority rolling summaries with a synchronous core for tests."""

    def __init__(self, store: ConversationStore, summarizer, *,
                 soft_limit: int = DEFAULT_SOFT_LIMIT,
                 recent_count: int = DEFAULT_RECENT_COUNT,
                 input_limit: int = SUMMARY_INPUT_LIMIT,
                 interactive_busy=None,
                 logger=None,
                 retry_delay: float = 0.5):
        self.store = store
        self.summarizer = summarizer
        self.soft_limit = max(1, int(soft_limit))
        self.recent_count = max(1, int(recent_count))
        self.input_limit = max(1, int(input_limit))
        self.interactive_busy = interactive_busy or (lambda: False)
        self.logger = logger or (lambda _message: None)
        self.retry_delay = max(0.01, float(retry_delay))
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def needs_summary(self, session_id: str) -> bool:
        session = self.store.get_session(session_id)
        if session is None:
            return False
        messages = self.store.list_messages(
            session_id, after_id=session.summary_until_message_id)
        if len(messages) <= self.recent_count:
            return False
        source = _render(
            session.summary_text.strip(),
            [_line(message) for message in messages])
        return len(source) > self.soft_limit

    def run_once(self, session_id: str) -> bool:
        if self.interactive_busy():
            return False
        try:
            session = self.store.get_session(session_id)
            if session is None or not self.needs_summary(session_id):
                return False
            messages = self.store.list_messages(
                session_id, after_id=session.summary_until_message_id)
            candidates = messages[:-self.recent_count]
            selected = []
            selected_chars = 0
            for message in candidates:
                line = _line(message)
                added = len(line) + (1 if selected else 0)
                if selected and selected_chars + added > self.input_limit:
                    break
                selected.append(message)
                selected_chars += added
            if not selected:
                return False
            source = "\n".join(_line(message) for message in selected)
            summary = self.summarizer.summarize(
                session.summary_text, source).strip()
            if not summary:
                raise ValueError("empty summary")
            self.store.update_summary(
                session_id, summary, selected[-1].id, version=1)
            return True
        except Exception as exc:
            self.logger(f"摘要更新失败 {type(exc).__name__}")
            return False

    def schedule(self, session_id: str) -> None:
        if not session_id or self._stopping.is_set():
            return
        with self._pending_lock:
            if session_id in self._pending:
                return
            self._pending.add(session_id)
        self._queue.put(session_id)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._loop, name="jev-summary", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stopping.set()
        self._queue.put(None)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))

    def _loop(self) -> None:
        while not self._stopping.is_set():
            session_id = self._queue.get()
            if session_id is None:
                return
            with self._pending_lock:
                self._pending.discard(session_id)
            if self.interactive_busy():
                if self._stopping.wait(self.retry_delay):
                    return
                self.schedule(session_id)
                continue
            changed = self.run_once(session_id)
            if changed and self.needs_summary(session_id):
                self.schedule(session_id)
