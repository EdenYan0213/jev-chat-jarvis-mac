"""Session management model and native AppKit pane."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sqlite3
import threading
from typing import Callable

import AppKit as A
import objc
from Foundation import NSIndexSet, NSObject, NSMakeRect, NSTimer

from conversation_store import ConversationStore, SessionRecord, StoredMessage
import ui_style


RETENTION_DAYS = 30
PALETTE = ui_style.PALETTE


@dataclass(frozen=True)
class SessionRow:
    id: str
    chat_key: str
    name: str
    message_count: int
    updated_at: float
    last_active_at: float
    deleted_at: float | None
    deletes_at: float | None


@dataclass(frozen=True)
class SessionDetail:
    row: SessionRow
    summary_text: str
    has_observation_gap: bool
    messages: tuple[StoredMessage, ...]


class SessionManagerModel:
    """Cocoa-free session lifecycle operations for the settings pane."""

    def __init__(self, store: ConversationStore,
                 retention_days: int = RETENTION_DAYS):
        self.store = store
        self.retention_days = max(0, int(retention_days))

    def _row(self, session: SessionRecord) -> SessionRow:
        deletes_at = None
        if session.deleted_at is not None:
            deletes_at = session.deleted_at + self.retention_days * 86400
        return SessionRow(
            id=session.id,
            chat_key=session.chat_key,
            name=session.name,
            message_count=self.store.message_count(session.id),
            updated_at=session.updated_at,
            last_active_at=session.last_active_at,
            deleted_at=session.deleted_at,
            deletes_at=deletes_at,
        )

    @staticmethod
    def _matches(row: SessionRow, search: str | None) -> bool:
        needle = " ".join(str(search or "").split()).casefold()
        if not needle:
            return True
        return needle in row.name.casefold() or needle in row.chat_key.casefold()

    def _rows(self, deleted: bool,
              search: str | None = None) -> list[SessionRow]:
        rows = [
            self._row(session)
            for session in self.store.list_sessions(include_deleted=True)
            if (session.deleted_at is not None) == deleted
        ]
        rows = [row for row in rows if self._matches(row, search)]
        return sorted(
            rows,
            key=lambda row: (
                row.deleted_at if deleted else row.updated_at,
                row.last_active_at,
                row.id,
            ),
            reverse=True,
        )

    def active_rows(self, search: str | None = None) -> list[SessionRow]:
        return self._rows(False, search)

    def trash_rows(self, search: str | None = None) -> list[SessionRow]:
        return self._rows(True, search)

    def detail(self, session_id: str) -> SessionDetail:
        session = self.store.get_session(session_id, include_deleted=True)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return SessionDetail(
            row=self._row(session),
            summary_text=session.summary_text,
            has_observation_gap=session.has_observation_gap,
            messages=tuple(self.store.list_messages(session_id)),
        )

    def change_token(self) -> tuple:
        return self.store.change_token()

    def create(self, chat_key: str, name: str | None = None) -> SessionRow:
        return self._row(self.store.create_session(chat_key, name))

    def rename(self, session_id: str, name: str) -> SessionRow:
        return self._row(self.store.rename_session(session_id, name))

    def trash(self, session_id: str) -> SessionRow:
        self.store.trash_session(session_id)
        session = self.store.get_session(session_id, include_deleted=True)
        if session is None:
            raise KeyError(f"session not found after trash: {session_id}")
        return self._row(session)

    def restore(self, session_id: str) -> SessionRow:
        return self._row(self.store.restore_session(session_id))

    def delete_permanently(self, session_id: str) -> None:
        self.store.delete_session_permanently(session_id)


class SessionManagerPane(NSObject):
    """Native session list used inside the settings window."""

    @objc.python_method
    def build(self, store: ConversationStore, frame=None,
              on_change: Callable[[str, bool], None] | None = None,
              operation_lock=None):
        self.model = SessionManagerModel(store)
        self.on_change = on_change
        self.operation_lock = operation_lock or threading.RLock()
        self.showing_trash = False
        self.rows: list[SessionRow] = []
        self.refresh_timer = None
        self._last_change_token = None
        self._refreshing = False

        frame = frame or NSMakeRect(0, 0, 760, 596)
        self.view = A.NSView.alloc().initWithFrame_(frame)
        self.view.setWantsLayer_(True)
        self.view.layer().setBackgroundColor_(PALETTE["bg"].CGColor())

        title = self._label(
            "会话管理", 24, 548, 300, 28, 22, PALETTE["text"], True)
        title.setAccessibilityRoleDescription_("会话管理")
        self._label(
            "显示所有聊天的 Session 和双方消息，消息会自动更新；回收站内容 30 天后清理。",
            24, 522, 620, 20, 11, PALETTE["muted"])

        self.search = A.NSSearchField.alloc().initWithFrame_(
            NSMakeRect(24, 482, 410, 30))
        self.search.setPlaceholderString_("搜索会话名或聊天名")
        self.search.setTarget_(self)
        self.search.setAction_("searchChanged:")
        if hasattr(self.search, "setSendsSearchStringImmediately_"):
            self.search.setSendsSearchStringImmediately_(True)
        self.search.setAccessibilityLabel_("搜索会话")
        self._style_field(self.search)
        self.view.addSubview_(self.search)

        self.mode = A.NSSegmentedControl.alloc().initWithFrame_(
            NSMakeRect(546, 483, 190, 28))
        self.mode.setSegmentCount_(2)
        self.mode.setLabel_forSegment_("使用中", 0)
        self.mode.setLabel_forSegment_("回收站", 1)
        self.mode.setSelectedSegment_(0)
        self.mode.setTarget_(self)
        self.mode.setAction_("modeChanged:")
        self.view.addSubview_(self.mode)

        self.table = A.NSTableView.alloc().initWithFrame_(
            NSMakeRect(0, 0, 310, 360))
        self.table.setDelegate_(self)
        self.table.setDataSource_(self)
        self.table.setRowHeight_(29)
        self.table.setAllowsMultipleSelection_(False)
        self.table.setUsesAlternatingRowBackgroundColors_(True)
        self.table.setTarget_(self)
        self.table.setDoubleAction_("renameSelected:")
        for identifier, label, width in (
            ("chat", "聊天", 118),
            ("name", "Session", 132),
            ("count", "消息", 52),
        ):
            column = A.NSTableColumn.alloc().initWithIdentifier_(identifier)
            column.headerCell().setStringValue_(label)
            column.setWidth_(width)
            column.setMinWidth_(44)
            self.table.addTableColumn_(column)

        scroll = A.NSScrollView.alloc().initWithFrame_(
            NSMakeRect(24, 102, 318, 368))
        scroll.setDocumentView_(self.table)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(A.NSBezelBorder)
        self.view.addSubview_(scroll)

        self.detail_title = self._label(
            "选择一个 Session", 360, 446, 376, 24, 15,
            PALETTE["text"], True)
        self.detail_title.setLineBreakMode_(A.NSLineBreakByTruncatingTail)
        self.detail_meta = self._label(
            "左侧列出了本机保存的全部聊天。", 360, 402, 376, 40,
            11, PALETTE["muted"])
        self.detail_meta.cell().setWraps_(True)

        self.history = A.NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, 360, 286))
        self.history.setEditable_(False)
        self.history.setSelectable_(True)
        self.history.setRichText_(False)
        self.history.setFont_(A.NSFont.systemFontOfSize_(12))
        self.history.setTextColor_(PALETTE["text"])
        self.history.setBackgroundColor_(PALETTE["field"])
        self.history.setString_("选择左侧 Session 后，这里会显示摘要和完整消息记录。")
        self.history.setAccessibilityLabel_("Session 消息记录")

        history_scroll = A.NSScrollView.alloc().initWithFrame_(
            NSMakeRect(360, 102, 376, 294))
        history_scroll.setDocumentView_(self.history)
        history_scroll.setHasVerticalScroller_(True)
        history_scroll.setAutohidesScrollers_(True)
        history_scroll.setBorderType_(A.NSBezelBorder)
        self.view.addSubview_(history_scroll)

        self.new_button = self._button(
            "新建 Session", "createSelected:", 24, 54, 112)
        self.rename_button = self._button(
            "重命名", "renameSelected:", 146, 54, 96)
        self.trash_button = self._button(
            "移到回收站", "trashSelected:", 252, 54, 118)
        self.restore_button = self._button(
            "恢复", "restoreSelected:", 24, 54, 112)
        self.delete_button = self._button(
            "永久删除", "deleteSelected:", 146, 54, 128)
        self.status = self._label(
            "", 388, 50, 348, 40, 11, PALETTE["muted"])
        self.status.cell().setWraps_(True)
        self.refresh()
        return self

    @objc.python_method
    def _label(self, text, x, y, w, h, size=13, color=None, bold=False):
        label = ui_style.make_label(
            text, x, y, w, h, size, color, bold=bold)
        self.view.addSubview_(label)
        return label

    @objc.python_method
    def _button(self, title, action, x, y, width):
        button = A.NSButton.alloc().initWithFrame_(
            NSMakeRect(x, y, width, 32))
        button.setTitle_(title)
        ui_style.style_button(button, font_size=11, radius=8)
        button.setTarget_(self)
        button.setAction_(action)
        self.view.addSubview_(button)
        return button

    @objc.python_method
    def _style_field(self, field):
        field.setFont_(A.NSFont.systemFontOfSize_(12))
        field.setTextColor_(PALETTE["text"])
        field.setBackgroundColor_(PALETTE["field"])
        field.setWantsLayer_(True)
        field.layer().setBorderColor_(PALETTE["edge"].CGColor())
        field.layer().setBorderWidth_(0.75)
        field.layer().setCornerRadius_(ui_style.RADIUS_FIELD)

    @objc.python_method
    def _selected(self) -> SessionRow | None:
        index = self.table.selectedRow()
        if index < 0 or index >= len(self.rows):
            return None
        return self.rows[index]

    @objc.python_method
    def _set_status(self, text: str, error: bool = False):
        self.status.setStringValue_(text)
        self.status.setTextColor_(
            PALETTE["red"] if error else PALETTE["muted"])

    @objc.python_method
    def _notify(self, chat_key: str, active_changed: bool):
        if self.on_change is not None:
            self.on_change(chat_key, active_changed)

    @objc.python_method
    def _run(self, operation, success: str):
        try:
            with self.operation_lock:
                row, active_changed = operation()
                self.refresh(preferred_id=row.id if row is not None else None)
                if row is not None:
                    self._notify(row.chat_key, active_changed)
        except sqlite3.Error:
            self._set_status(
                "会话数据库暂不可用，请重新打开应用。", True)
            return
        except (KeyError, ValueError, OSError) as exc:
            self._set_status(str(exc) or "操作失败。", True)
            return
        self._set_status(success)

    @objc.python_method
    def refresh(self, preferred_id: str | None = None,
                follow_latest: bool = False):
        selected = self._selected()
        selected_id = (
            preferred_id
            if preferred_id is not None
            else selected.id if selected is not None else None
        )
        previous_count = (
            selected.message_count
            if selected is not None and selected.id == selected_id
            else None
        )
        search = self.search.stringValue() if hasattr(self, "search") else ""
        try:
            self.rows = (
                self.model.trash_rows(search)
                if self.showing_trash else self.model.active_rows(search)
            )
        except sqlite3.Error:
            self.rows = []
            if hasattr(self, "status"):
                self._set_status(
                    "会话数据库暂不可用，请重新打开应用。", True)
        self._refreshing = True
        try:
            self.table.reloadData()
            if self.rows:
                index = next(
                    (i for i, row in enumerate(self.rows)
                     if row.id == selected_id),
                    0,
                )
                row = self.rows[index]
                self.table.selectRowIndexes_byExtendingSelection_(
                    NSIndexSet.indexSetWithIndex_(index), False)
                self.table.scrollRowToVisible_(index)
                show_latest = bool(
                    follow_latest
                    and previous_count is not None
                    and row.id == selected_id
                    and row.message_count > previous_count
                )
                self._show_detail(row, follow_latest=show_latest)
            else:
                self.table.deselectAll_(None)
                self._clear_detail(
                    "回收站中没有 Session。"
                    if self.showing_trash else
                    "没有匹配的 Session。")
        finally:
            self._refreshing = False
        self.rename_button.setHidden_(self.showing_trash)
        self.trash_button.setHidden_(self.showing_trash)
        self.new_button.setHidden_(self.showing_trash)
        self.restore_button.setHidden_(not self.showing_trash)
        self.delete_button.setHidden_(not self.showing_trash)
        self._update_buttons()
        try:
            self._last_change_token = self.model.change_token()
        except sqlite3.Error:
            self._last_change_token = None

    @objc.python_method
    def _update_buttons(self):
        selected = self._selected() is not None
        self.new_button.setEnabled_(selected and not self.showing_trash)
        self.rename_button.setEnabled_(selected and not self.showing_trash)
        self.trash_button.setEnabled_(selected and not self.showing_trash)
        self.restore_button.setEnabled_(selected and self.showing_trash)
        self.delete_button.setEnabled_(selected and self.showing_trash)

    @objc.python_method
    def start_auto_refresh(self):
        timer = self.refresh_timer
        if timer is not None and timer.isValid():
            return
        self.refresh_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.5, self, "pollForUpdates:", None, True))

    @objc.python_method
    def stop_auto_refresh(self):
        timer = self.refresh_timer
        if timer is not None:
            timer.invalidate()
        self.refresh_timer = None

    def pollForUpdates_(self, timer):
        if self.view.isHidden():
            return
        try:
            token = self.model.change_token()
        except sqlite3.Error:
            self._set_status(
                "会话数据库暂不可用，请重新打开应用。", True)
            return
        if token != self._last_change_token:
            self.refresh(follow_latest=True)

    @objc.python_method
    def _clear_detail(self, message: str):
        self.detail_title.setStringValue_("没有可显示的 Session")
        self.detail_meta.setStringValue_("")
        self.history.setString_(message)

    @staticmethod
    def _speaker(message: StoredMessage) -> str:
        if message.side == "me":
            return "我"
        if message.side == "them":
            return message.sender or "对方"
        return message.sender or "方向未确认"

    @classmethod
    def _history_text(cls, detail: SessionDetail) -> str:
        summary = detail.summary_text.strip() or "尚未生成摘要"
        parts = ["会话摘要", summary, "", f"消息记录（{len(detail.messages)} 条）"]
        if not detail.messages:
            parts.extend(["", "这个 Session 还没有保存到消息。"])
        for message in detail.messages:
            timestamp = datetime.fromtimestamp(
                message.observed_at).strftime("%Y-%m-%d %H:%M")
            parts.extend([
                "",
                f"{timestamp} · {cls._speaker(message)}",
                message.text,
            ])
        return "\n".join(parts)

    @objc.python_method
    def _show_detail(self, row: SessionRow, follow_latest: bool = False):
        try:
            detail = self.model.detail(row.id)
        except sqlite3.Error:
            self._clear_detail("消息记录暂时无法读取，请重新打开应用。")
            self._set_status("会话数据库暂不可用，请重新打开应用。", True)
            return
        except KeyError:
            self._clear_detail("这个 Session 已不存在，请刷新后重试。")
            return

        updated = datetime.fromtimestamp(
            detail.row.updated_at).strftime("%Y-%m-%d %H:%M")
        state = "历史可能有识别缺口" if detail.has_observation_gap else "历史连续"
        if detail.row.deletes_at is not None:
            state += " · " + datetime.fromtimestamp(
                detail.row.deletes_at).strftime("%Y-%m-%d 自动清理")
        self.detail_title.setStringValue_(detail.row.name)
        self.detail_meta.setStringValue_(
            f"聊天：{detail.row.chat_key}\n"
            f"{detail.row.message_count} 条消息 · {updated} 更新 · {state}")
        self.history.setString_(self._history_text(detail))
        if follow_latest:
            self.history.scrollToEndOfDocument_(None)
        else:
            self.history.scrollToBeginningOfDocument_(None)

    def numberOfRowsInTableView_(self, table):
        return len(self.rows)

    def tableView_objectValueForTableColumn_row_(self, table, column, row):
        item = self.rows[row]
        identifier = str(column.identifier())
        if identifier == "name":
            return item.name
        if identifier == "chat":
            return item.chat_key
        if identifier == "count":
            return str(item.message_count)
        if identifier == "updated":
            return datetime.fromtimestamp(
                item.updated_at).strftime("%Y-%m-%d %H:%M")
        if identifier == "deletes" and item.deletes_at is not None:
            return datetime.fromtimestamp(
                item.deletes_at).strftime("%Y-%m-%d")
        return ""

    def tableViewSelectionDidChange_(self, notification):
        self._update_buttons()
        if self._refreshing:
            return
        row = self._selected()
        if row is not None:
            self._show_detail(row)

    def searchChanged_(self, sender):
        self.refresh()

    def modeChanged_(self, sender):
        self.showing_trash = sender.selectedSegment() == 1
        self.refresh()
        self._set_status("")

    def createSelected_(self, sender):
        row = self._selected()
        if row is None or self.showing_trash:
            return
        field = A.NSTextField.alloc().initWithFrame_(
            NSMakeRect(0, 0, 320, 26))
        field.setPlaceholderString_("名称可留空，应用会自动命名")
        alert = A.NSAlert.alloc().init()
        alert.setMessageText_(f"在「{row.chat_key}」中新建 Session")
        alert.setInformativeText_(
            "新 Session 会立即成为这个聊天正在使用的会话。")
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("新建")
        alert.addButtonWithTitle_("取消")
        if alert.runModal() != A.NSAlertFirstButtonReturn:
            return
        name = field.stringValue().strip() or None
        self._run(
            lambda: (self.model.create(row.chat_key, name), True),
            f"已在「{row.chat_key}」中新建 Session。",
        )

    def renameSelected_(self, sender):
        row = self._selected()
        if row is None or self.showing_trash:
            return
        field = A.NSTextField.alloc().initWithFrame_(
            NSMakeRect(0, 0, 320, 26))
        field.setStringValue_(row.name)
        alert = A.NSAlert.alloc().init()
        alert.setMessageText_("重命名 Session")
        alert.setInformativeText_("名称只用于本机区分会话。")
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("保存")
        alert.addButtonWithTitle_("取消")
        if alert.runModal() != A.NSAlertFirstButtonReturn:
            return
        name = field.stringValue()
        self._run(
            lambda: (self.model.rename(row.id, name), False),
            "会话名称已更新。",
        )

    def trashSelected_(self, sender):
        row = self._selected()
        if row is None or self.showing_trash:
            return

        def operation():
            active = self.model.store.active_session(row.chat_key)
            active_changed = active is not None and active.id == row.id
            return self.model.trash(row.id), active_changed

        self._run(
            operation,
            "会话已移到回收站，可在 30 天内恢复。",
        )

    def restoreSelected_(self, sender):
        row = self._selected()
        if row is None or not self.showing_trash:
            return
        self._run(
            lambda: (self.model.restore(row.id), False),
            "会话已恢复。",
        )

    def deleteSelected_(self, sender):
        row = self._selected()
        if row is None or not self.showing_trash:
            return
        alert = A.NSAlert.alloc().init()
        alert.setAlertStyle_(A.NSAlertStyleCritical)
        alert.setMessageText_("永久删除这个 Session？")
        alert.setInformativeText_(
            "本机会话记录和摘要将一并删除，此操作无法撤销。")
        alert.addButtonWithTitle_("永久删除")
        alert.addButtonWithTitle_("取消")
        if alert.runModal() != A.NSAlertFirstButtonReturn:
            return

        def operation():
            self.model.delete_permanently(row.id)
            return row, False

        self._run(operation, "会话已永久删除。")
