"""Align visible OCR messages with durable history and render model context."""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Observation:
    session: SessionRecord
    visible_ids: tuple[int, ...]
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


def _latest_subsequence(haystack: list[tuple], needle: list[tuple]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
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

        visible_keys = [message_key(message) for message in visible]
        recent = self.store.recent_messages(
            session.id, limit=self.alignment_limit)
        recent_keys = [message_key(message) for message in recent]

        existing_start = _latest_subsequence(recent_keys, visible_keys)
        if existing_start is not None:
            ids = tuple(
                message.id
                for message in recent[
                    existing_start:existing_start + len(visible)])
            return Observation(session, ids, (), False)

        overlap = 0
        max_overlap = min(len(recent), len(visible))
        for size in range(max_overlap, 0, -1):
            if recent_keys[-size:] == visible_keys[:size]:
                overlap = size
                break

        if overlap:
            appended = self.store.append_messages(
                session.id,
                [_message_input(message) for message in visible[overlap:]])
            ids = tuple(
                [message.id for message in recent[-overlap:]]
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
