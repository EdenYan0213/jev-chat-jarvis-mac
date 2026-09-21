"""Floating HUD: a non-activating panel beside WeChat showing intent, risk and ranked replies.

Design notes
  * NSWindowStyleMaskNonactivatingPanel + floating level: the panel never steals focus
    from WeChat, and window-ID capture means it never appears in our own screenshots.
  * Poll loop: read the chat, hash the newest message, judge only when it changes.
  * Judgment lands first (fast, ~0.5 s) and candidates fill in when the generator
    finishes (~2 s), mirroring the phone demo's "生成中…" state.
  * The panel positions itself against WeChat's window each tick, so it follows moves,
    resizes and monitor changes without any window-server hooks.
  * Palette is WeChat's light theme (see PALETTE below); the Appearance is pinned to Aqua
    so the title bar and button bezels stay light even when the system is in dark mode.
  * 「填入」 writes through the Accessibility API into WeChat's input box (src/fill.py): no
    synthetic keystrokes, no clipboard, and nothing needs to be frontmost. It needs the
    Accessibility permission; when that is missing the HUD asks for it and reports the
    failure.
"""

from __future__ import annotations

import objc
import subprocess
import threading
import time
from pathlib import Path

import AppKit
import sys

from AppKit import (
    NSAppearance,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSFont,
    NSPanel,
    NSPasteboard,
    NSPasteboardTypeString,
    NSScreen,
    NSTextField,
    NSView,
    NSWindowMiniaturizeButton,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled,
    NSWindowZoomButton,
    NSWindowCloseButton,
)
from Foundation import NSMakeRect, NSObject, NSTimer

sys.path.insert(0, str(Path(__file__).parent))
import userconfig  # noqa: E402

userconfig.load()   # ~/.config/jev-jarvis/env -> os.environ (Finder apps inherit none)

from perception import read_conversation, screen_capture_ok, request_screen_capture  # noqa: E402
from judge import make_judge  # noqa: E402
from generate import Generator, load_credentials  # noqa: E402
import fill  # noqa: E402

PANEL_W, PANEL_H = 360, 614   # tall enough for 3-line candidates + the chat name row
COLLAPSED_H = 96              # height when the panel is rolled up
POLL_INTERVAL = 1.0     # detection granularity
SETTLE_S = 1.2          # wait this long with no new message before analysing (anti-flood)
MIN_GAP_S = 2.0         # never restart analysis faster than this
N_CANDIDATES = 3

# ---------------------------------------------------------------- palette
# WeChat's light theme: a #F7F7F7 surface, near-black body text, #888888 for anything
# secondary, and the brand green/amber/red carrying the risk state.


def _rgb(hex_code: int, alpha: float = 1.0) -> NSColor:
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(
        ((hex_code >> 16) & 0xFF) / 255.0,
        ((hex_code >> 8) & 0xFF) / 255.0,
        (hex_code & 0xFF) / 255.0,
        alpha,
    )


PALETTE = {
    "bg": _rgb(0xF7F7F7),     # panel surface
    "text": _rgb(0x191919),   # judged message, intent, candidates, action advice
    "muted": _rgb(0x888888),  # status, sender/context, confidence, percentages, headers
    "green": _rgb(0x07C160),  # WeChat brand green — risk 安全, success feedback
    "amber": _rgb(0xFA9D3B),  # risk 留神
    "red": _rgb(0xFA5151),    # risk 危险, failures
}

# Candidate row geometry: 复制 + 填入 share the panel's right edge, so they compete with the
# candidate text for width. Measured on screen: a rounded bezel holds a two-character Chinese
# title at 56 pt and clips it to one character at 44 pt ("复" / "填"), so the buttons cannot be
# shrunk further — the text column gives up the 24 pt instead. Height is fine at 24 pt; the
# cell's own 32 pt request is padding for a focus ring this button never draws.
CAND_BTN_W, CAND_BTN_H, CAND_BTN_GAP = 56, 24, 4
CAND_BTN_X = PANEL_W - 14 - (2 * CAND_BTN_W + CAND_BTN_GAP)   # 230
# Rank/percentage label ("#3 · 100%"): NSTextField's cell insets mean the widest string
# actually consumes 63 px at 11 pt, so the old 48 px frame clipped the "%" off every row.
# 72 px still clears that with 9 px to spare, and the 12 px it gives back go to the
# candidate text — which needs them: at 116 px a 30-character candidate (the generation
# prompt's own cap) lost its last two characters to the 3-line limit.
CAND_PROB_X, CAND_PROB_W = 14, 72                              # 14 .. 86
CAND_TEXT_X = CAND_PROB_X + CAND_PROB_W + 8                    # 106
CAND_TEXT_W = CAND_BTN_X - CAND_TEXT_X - 8                     # 128
CAND_TEXT_H = 48                                                # up to 3 wrapped lines


class HudController(NSObject):
    def init(self):
        self = objc.super(HudController, self).init()
        if self is None:
            return None
        self.last_seen = None          # newest message text observed
        self.last_change_ts = 0.0      # when it last changed (burst detection)
        self.last_analyze_ts = 0.0     # rate limit for analysis starts
        self.analyzed_text = None      # what the panel currently shows
        self.judge = make_judge()
        self.generator = Generator()
        self.candidates: list[str] = []
        self._busy = False
        self._collapsed = False
        self._expanded_h = None       # full height, captured the first time we collapse
        self._paused = False
        self._chat_title = ""
        self._asked_permission = False
        self._win_wid = None          # sticky WeChat window id
        self._last_origin = None      # last applied panel origin
        self._pending_origin = None   # candidate origin awaiting confirmation
        self._build_panel()
        self._expanded_h = self.panel.frame().size.height
        return self

    # ------------------------------------------------------------------ ui
    @objc.python_method
    def _build_panel(self):
        # Closable/Miniaturizable are what actually CREATE the standard window buttons;
        # NonactivatingPanel alone gives a title bar with no controls at all.
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                 | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskNonactivatingPanel)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PANEL_W, PANEL_H), style, NSBackingStoreBuffered, False)
        self.panel.setLevel_(AppKit.NSFloatingWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setAlphaValue_(1.0)   # light surfaces go grey/washed out below 1.0
        # The title bar and button bezels are drawn from the appearance, not from the
        # background colour, so pin Aqua: a dark-mode system would otherwise give a dark
        # title bar above a white panel.
        self.panel.setAppearance_(NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameAqua))
        self.panel.setBackgroundColor_(PALETTE["bg"])
        self.panel.setTitle_("jev-jarvis")
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setBecomesKeyOnlyIfNeeded_(True)

        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PANEL_H))
        self.rows: dict[str, NSTextField] = {}

        # Layout order matters: the message being judged is the anchor of the panel,
        # so it sits right under the title in the brightest, largest type.
        # The full-width rows (PANEL_W - 28 = 332 px) cannot clip their widest string:
        # "意图识别率 100%" measures 103 px at 12 pt.
        y = PANEL_H - 30
        for key, size, color, bold, height in (
            ("chat", 12, PALETTE["green"], True, 18),      # 群名 / 联系人
            ("status", 10, PALETTE["muted"], False, 14),
            ("message", 15, PALETTE["text"], False, 50),      # the message under analysis
            ("sender", 10, PALETTE["muted"], False, 14),
            ("intent", 21, PALETTE["text"], True, 28),
            ("confidence", 12, PALETTE["muted"], False, 18),
            ("risk", 14, PALETTE["green"], True, 20),
            ("actions", 13, PALETTE["text"], False, 18),
        ):
            tf = self._make_label(14, y - height, PANEL_W - 28, height,
                                  size=size, color=color, bold=bold)
            if key == "message":
                tf.cell().setWraps_(True)
            view.addSubview_(tf)
            self.rows[key] = tf
            y -= height + 8

        # ---- candidates section
        y -= 6
        header = self._make_label(14, y - 16, PANEL_W - 28, 16,
                                  size=11, color=PALETTE["muted"])
        header.setStringValue_("候选回复（按合适度排序）")
        view.addSubview_(header)
        self.rows["cand_header"] = header
        y -= 22

        self.cand_rows = []
        for i in range(N_CANDIDATES):
            prob = self._make_label(CAND_PROB_X, y - 14, CAND_PROB_W, 14,
                                    size=11, color=PALETTE["muted"])
            view.addSubview_(prob)
            text = self._make_label(CAND_TEXT_X, y - CAND_TEXT_H, CAND_TEXT_W, CAND_TEXT_H,
                                    size=12, color=PALETTE["text"])
            text.cell().setWraps_(True)
            view.addSubview_(text)
            copy_btn = self._make_button(CAND_BTN_X, y - 34, CAND_BTN_W, CAND_BTN_H,
                                         "复制", "copyCandidate:", i)
            fill_btn = self._make_button(CAND_BTN_X + CAND_BTN_W + CAND_BTN_GAP, y - 34,
                                         CAND_BTN_W, CAND_BTN_H, "填入", "fillCandidate:", i)
            view.addSubview_(copy_btn)
            view.addSubview_(fill_btn)
            self.cand_rows.append({"prob": prob, "text": text, "btn": copy_btn,
                                   "fill_btn": fill_btn})
            y -= 56

        self.panel.setContentView_(view)
        self.rows["status"].setStringValue_("等待微信消息…")
        self._wire_window_controls()
        self._install_status_item()

    @objc.python_method
    def _wire_window_controls(self):
        """Native traffic lights, mapped to this app's actions.

        red    -> quit. A hidden panel would otherwise be unreachable: LSUIElement apps
                  have no Dock icon, so a plain order-out looks like a crash.
        yellow -> roll the panel up instead of miniaturizing, for the same reason.
        green  -> hidden: the HUD has a fixed size and nothing to zoom.
        """
        close = self.panel.standardWindowButton_(NSWindowCloseButton)
        mini = self.panel.standardWindowButton_(NSWindowMiniaturizeButton)
        zoom = self.panel.standardWindowButton_(NSWindowZoomButton)
        if close:
            close.setTarget_(self)
            close.setAction_("quitApp:")
            close.setToolTip_("退出 jev-jarvis")
        if mini:
            mini.setTarget_(self)
            mini.setAction_("collapsePanel:")
            mini.setToolTip_("收起 / 展开面板")
        if zoom:
            zoom.setHidden_(True)

    @objc.python_method
    def _install_status_item(self):
        """Menu-bar item — the standard place for a background helper's controls."""
        bar = AppKit.NSStatusBar.systemStatusBar()
        self.status_item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        self.status_item.button().setTitle_("J")
        self.status_item.button().setToolTip_("jev-jarvis · 微信意图助手")

        menu = AppKit.NSMenu.alloc().init()
        for title, action, key in (
            ("显示 / 收起面板", "collapsePanel:", ""),
            ("暂停读屏", "togglePause:", ""),
            ("立即重新分析", "reanalyze:", ""),
        ):
            menu.addItemWithTitle_action_keyEquivalent_(title, action, key)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        menu.addItemWithTitle_action_keyEquivalent_("退出 jev-jarvis", "quitApp:", "q")
        for item in menu.itemArray():
            item.setTarget_(self)
        self.pause_item = menu.itemArray()[1]
        self.status_item.setMenu_(menu)

    @objc.python_method
    def _make_label(self, x, y, w, h, size=13, color=None, bold=False):
        tf = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        tf.setStringValue_("")
        tf.setBezeled_(False)
        tf.setDrawsBackground_(False)
        tf.setEditable_(False)
        tf.setSelectable_(True)
        tf.setTextColor_(PALETTE["text"] if color is None else color)
        tf.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        return tf

    @objc.python_method
    def _make_button(self, x, y, w, h, title, action, tag):
        """A native rounded bezel with a WeChat-green label — reads correctly on #F7F7F7."""
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setFont_(NSFont.systemFontOfSize_(11))
        btn.setContentTintColor_(PALETTE["green"])
        btn.setTarget_(self)
        btn.setAction_(action)
        btn.setTag_(tag)
        btn.setHidden_(True)
        return btn

    @objc.python_method
    def _show(self):
        if not self.panel.isVisible():
            self.panel.orderFrontRegardless()

    @objc.python_method
    def _context_line(self, sender, prev: str) -> str:
        parts = []
        if sender:
            parts.append(f"来自 {sender}")
        if prev:
            parts.append(f"上文：{prev[:26]}")
        return " · ".join(parts)

    @objc.python_method
    def _render(self, key: str, text: str, color: NSColor | None = None):
        tf = self.rows[key]
        tf.setStringValue_(text)
        if color is not None:
            tf.setTextColor_(color)

    @objc.python_method
    def _render_candidates(self, ranked: list[dict]):
        self.candidates = [r["text"] for r in ranked]
        for i, row in enumerate(self.cand_rows):
            if i < len(ranked):
                r = ranked[i]
                row["prob"].setStringValue_(f"#{i + 1} · {r['prob'] * 100:.0f}%")
                row["text"].setStringValue_(r["text"])
                row["btn"].setHidden_(False)
                row["fill_btn"].setHidden_(False)
            else:
                row["prob"].setStringValue_("")
                row["text"].setStringValue_("")
                row["btn"].setHidden_(True)
                row["fill_btn"].setHidden_(True)

    @objc.python_method
    def _clear_candidates(self):
        for row in self.cand_rows:
            row["prob"].setStringValue_("")
            row["text"].setStringValue_("")
            row["btn"].setHidden_(True)
            row["fill_btn"].setHidden_(True)

    @objc.python_method
    def _display_height(self) -> float:
        """Height of the display whose origin is (0,0) — the Quartz<->Cocoa flip constant.

        Taking this from the *target* screen is wrong on multi-display setups: a screen
        placed above the main one has origin.y > 0 and the flip must still use the
        primary display's height.
        """
        for scr in NSScreen.screens():
            f = scr.frame()
            if f.origin.x == 0 and f.origin.y == 0:
                return f.size.height
        return NSScreen.mainScreen().frame().size.height

    @objc.python_method
    def _position_near(self, win: dict | None):
        """Dock the panel beside WeChat, on the screen WeChat is actually on.

        Uses global Cocoa coordinates throughout. NSScreen.mainScreen() must NOT be used:
        it follows whichever display holds the key window, so relying on it made the panel
        hop ~1369 px between displays a few times a minute.
        """
        flip = self._display_height()
        panel_h = self.panel.frame().size.height or PANEL_H
        panel_w = self.panel.frame().size.width or PANEL_W
        screens = list(NSScreen.screens())
        primary = next((s for s in screens
                        if s.frame().origin.x == 0 and s.frame().origin.y == 0), screens[0])

        if win:
            # CGWindow bounds are top-left origin global pixels -> Cocoa bottom-left
            wx, wy = win["x"], win["y"]
            ww, wh = win["w"], win["h"]
            cx_win = wx + ww / 2.0
            cyan = flip - (wy + wh / 2.0)
            host = next((s for s in screens
                         if s.frame().origin.x <= cx_win <= s.frame().origin.x + s.frame().size.width
                         and s.frame().origin.y <= cyan <= s.frame().origin.y + s.frame().size.height),
                        primary)
            sf = host.frame()
            # dock right of WeChat if it fits on that screen, else left, else its right edge
            x = wx + ww + 8
            if x + panel_w > sf.origin.x + sf.size.width:
                x = wx - panel_w - 8
            if x < sf.origin.x:
                x = sf.origin.x + sf.size.width - panel_w - 12
            y = flip - wy - panel_h
            y = max(sf.origin.y + 40, min(y, sf.origin.y + sf.size.height - panel_h - 40))
        else:
            sf = primary.frame()
            x = sf.size.width - panel_w - 12
            y = sf.size.height - panel_h - 60

        # dead-band: ignore sub-2pt corrections and one-off blips, so WeChat's own window
        # animations (and our own numeric noise) stop nudging the panel around
        target = (round(x), round(y))
        last = self._last_origin
        if last is None:                    # first placement: apply without debounce
            self._last_origin = target
            self._pending_origin = target
            self.panel.setFrameOrigin_(target)
            return
        if abs(target[0] - last[0]) <= 2 and abs(target[1] - last[1]) <= 2:
            return
        if target != self._pending_origin:
            self._pending_origin = target
            return  # require the same target on two consecutive ticks before moving
        self._last_origin = target
        self.panel.setFrameOrigin_(target)

    # ------------------------------------------------------------ actions
    def copyCandidate_(self, sender):
        idx = sender.tag()
        if 0 <= idx < len(self.candidates):
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(self.candidates[idx], NSPasteboardTypeString)
            self._render("status", f"已复制候选 #{idx + 1}", PALETTE["green"])

    def fillCandidate_(self, sender):
        """Paste the candidate into WeChat's input box (src/fill.py)."""
        idx = sender.tag()
        if not 0 <= idx < len(self.candidates):
            return
        text = self.candidates[idx]
        # Paste only reaches the frontmost app, so this clicks a moment later than the
        # panel would like; paint the state before the blocking wait.
        self._render("status", "填入中…", PALETTE["muted"])
        self.panel.displayIfNeeded()
        if not fill.has_accessibility():
            # First click is the moment to ask: the system dialog is the only way in.
            fill.request_accessibility()
        ok, reason = fill.fill_text(text)
        if ok:
            self._render("status", f"已填入候选 #{idx + 1}", PALETTE["green"])
        else:
            self._render("status", f"填入失败：{reason}", PALETTE["red"])

    # ------------------------------------------------------------ controls
    def collapsePanel_(self, sender):
        self._set_collapsed(not self._collapsed)

    def togglePause_(self, sender):
        self._paused = not self._paused
        self.pause_item.setTitle_("继续读屏" if self._paused else "暂停读屏")
        if self._paused:
            self._render("status", "已暂停 · 不再读屏", PALETTE["amber"])
            self._render("message", "", PALETTE["text"])
            self._render("sender", "", PALETTE["muted"])
            self._render("intent", "—", PALETTE["muted"])
            self._render("confidence", "", PALETTE["muted"])
            self._render("risk", "", PALETTE["muted"])
            self._render("actions", "", PALETTE["text"])
            self.rows["cand_header"].setStringValue_("")
            self._clear_candidates()
        else:
            self.last_seen = None      # force a fresh read of whatever is on screen
            self.analyzed_text = None
            self._render("status", "已恢复 · 读屏中", PALETTE["muted"])

    def reanalyze_(self, sender):
        self.last_seen = None
        self.analyzed_text = None
        self._render("status", "重新分析中…", PALETTE["muted"])

    def quitApp_(self, sender):
        AppKit.NSApplication.sharedApplication().terminate_(None)

    @objc.python_method
    def _set_collapsed(self, collapsed: bool):
        """Roll the panel up to a title+status strip, or back to full height."""
        self._collapsed = collapsed
        controlled = ["message", "sender", "intent", "confidence", "risk", "actions",
                      "cand_header"]   # "chat" and "status" survive collapsing
        for key in controlled:
            self.rows[key].setHidden_(collapsed)
        for row in self.cand_rows:
            for part in ("prob", "text", "btn", "fill_btn"):
                if part == "btn":
                    row["btn"].setHidden_(collapsed)
                elif part == "fill_btn":
                    row["fill_btn"].setHidden_(collapsed)
                else:
                    row[part].setHidden_(collapsed)

        rect = self.panel.frame()
        # _expanded_h was captured once in init(); never re-derive it here, or expanding
        # would read the collapsed height and stay collapsed
        new_h = COLLAPSED_H if collapsed else (self._expanded_h or PANEL_H)
        self.panel.setFrame_display_(
            NSMakeRect(rect.origin.x, rect.origin.y + (rect.size.height - new_h),
                       rect.size.width, new_h), True)
        self._last_origin = None      # let the next tick re-dock cleanly

    # --------------------------------------------------------------- loop
    def tick_(self, timer):
        if self._busy or self._paused:
            return  # paused, or a previous tick is still running
        self._busy = True
        threading.Thread(target=self._work, daemon=True).start()

    @objc.python_method
    def _work(self):
        try:
            self._work_inner()
        finally:
            self._busy = False

    @objc.python_method
    def _work_inner(self):
        if not screen_capture_ok():
            if not self._asked_permission:
                self._asked_permission = True
                request_screen_capture()      # opens the system prompt
            self._push("applyError:", "需要屏幕录制权限 · 系统设置 › 隐私与安全性")
            return
        try:
            res = read_conversation(previous_wid=self._win_wid)
        except Exception as e:
            self._push("applyError:", f"读取失败: {type(e).__name__}: {str(e)[:40]}")
            return
        if not res["ok"]:
            self._push("applyError:", f"{res['error']} · 微信没开或窗口被最小化？")
            return

        # position immediately: analysis takes seconds, and a delayed correction
        # showed up as a visible jump after the verdict landed
        self._win_wid = res["window"]["wid"]
        self._push("applyChat:", res.get("chat_title") or "")
        self._push("applyPosition:", res["window"])

        msgs = res["messages"]
        if not msgs:
            self._push("applyError:", "聊天区没读到文字")
            return

        thems = [m for m in msgs if m.side == "them"]
        newest = thems[-1] if thems else msgs[-1]
        prev_text = thems[-2].text if len(thems) > 1 else ""
        now = time.time()

        # --- anti-flood: track arrivals, never analyze mid-burst
        if newest.text != self.last_seen:
            self.last_seen = newest.text
            self.last_change_ts = now
            # keep the previous verdict readable; just badge that something new landed
            self._push("applyIncoming:", (newest.text, newest.sender, prev_text))

        settled = (now - self.last_change_ts) >= SETTLE_S
        cooled = (now - self.last_analyze_ts) >= MIN_GAP_S
        if newest.text != self.analyzed_text and settled and cooled:
            self.last_analyze_ts = now
            self.analyzed_text = newest.text
            self._push("applyPending:", (newest.text, newest.sender, prev_text))
            self._analyze(newest, msgs, prev_text)

    @objc.python_method
    def _analyze(self, newest, msgs, prev_text: str = ""):
        """Judge and generate in parallel, then rank. Judgment lands on screen first."""
        import concurrent.futures as cf

        context = "\n".join(m.text for m in msgs[:-1][-4:]) or None
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            # generation does not need the intent, so it runs while judging
            gen_future = ex.submit(self.generator.generate, newest.text, "")
            verdict = None
            try:
                verdict = self.judge.judge(newest.text, context=context)
                self._push("applyJudgment:", (verdict, newest.sender, prev_text))
            except Exception as e:
                self._push("applyError:", f"判断失败: {type(e).__name__}: {str(e)[:40]}")

            try:
                gen = gen_future.result()
            except Exception as e:
                self._push("applyError:", f"候选生成失败: {type(e).__name__}: {str(e)[:40]}")
                return
            texts = [c["text"] for c in gen.get("candidates", [])]
            if not texts:
                self._push("applyError:", f"候选生成失败: {gen.get('error', '空结果')[:60]}")
                return
            if verdict is not None:
                try:
                    ranked = self.judge.rank_candidates(newest.text, verdict["intent"], texts)
                    self._push("applyCandidates:", ranked)
                except Exception as e:
                    self._push("applyError:", f"排序失败: {type(e).__name__}: {str(e)[:40]}")
            else:
                self._push("applyCandidates:", [{"text": t, "prob": 1.0 / len(texts)}
                                                for t in texts])

    @objc.python_method
    def _push(self, selector: str, payload=None):
        self.performSelectorOnMainThread_withObject_waitUntilDone_(selector, payload, False)

    # --- main-thread callbacks (AppKit is not thread safe)
    def applyChat_(self, title):
        self._chat_title = title
        self._render("chat", title, PALETTE["green"])

    def applyIncoming_(self, payload):
        # a new message landed but we are not analysing yet (burst in progress):
        # keep the previous verdict visible, just badge it
        text, sender, prev = payload
        self._show()
        self._render("status", "有新消息 · 等消息停稳…", PALETTE["muted"])
        self._render("message", text, PALETTE["muted"])   # grey: not analysed yet
        self._render("sender", self._context_line(sender, prev), PALETTE["muted"])

    def applyPending_(self, payload):
        text, sender, prev = payload
        self._show()
        self._render("status", "分析中…", PALETTE["muted"])
        self._render("message", text, PALETTE["text"])    # inked: this is the one
        self._render("sender", self._context_line(sender, prev), PALETTE["muted"])
        self._clear_candidates()
        self.rows["cand_header"].setStringValue_("候选回复 · 等待判断…")

    def applyJudgment_(self, payload):
        v, sender, prev = payload
        self._show()
        self._render("message", v["message"], PALETTE["text"])
        self._render("sender", self._context_line(sender, prev), PALETTE["muted"])
        backend = v.get("backend", "")
        if backend.startswith("local (Jev"):
            # the backend label is "local (Jev-shaped decider-2b)": take what is inside the
            # parens without the paren, or the status line reads "... decider-2b)"
            detail = backend.split("(", 1)[1].rstrip(")")
            self._render("status", f"本地兜底 · {detail[:26]}", PALETTE["amber"])
        elif backend:
            self._render("status", f"分析完成 · {backend}", PALETTE["muted"])
        else:
            self._render("status", "分析完成", PALETTE["muted"])
        self._render("intent", v["intent"], PALETTE["text"])
        # the intent recognition rate, read off the judged intent — same muted slot
        self._render("confidence", f"意图识别率 {v['confidence']:.0%}", PALETTE["muted"])
        # Rounded, so the panel does not claim a precision it has: the judge reports a
        # mean like 4.7 out of a 10-level distribution, and "4.7/9" reads as a measurement
        # while "5/9" reads as the estimate it is. Deliberately the mean and not the most
        # likely level — measured on 8 real messages, this model's top level never exceeds
        # 0.4 and the argmax jumps 1/3/6 across near-identical criticism messages, while the
        # mean holds (派活 2.0–2.4, 批评 3.0–4.0, 闲聊 1.7).
        risk = int(round(float(v.get("risk", 0))))
        label = "安全" if risk <= 3 else ("留神" if risk <= 6 else "危险")
        color = PALETTE["green"] if risk <= 3 else (
            PALETTE["amber"] if risk <= 6 else PALETTE["red"])
        self._render("risk", f"● {label}  {risk}/9", color)
        self._render("actions", " · ".join(v.get("actions", [])), PALETTE["text"])
        self.rows["cand_header"].setStringValue_("候选回复 · 生成中…")

    def applyCandidates_(self, ranked):
        self.rows["cand_header"].setStringValue_("候选回复（按合适度排序）")
        self._render_candidates(ranked)

    def applyError_(self, text):
        self._show()                       # never vanish without telling the user why
        self._render("status", text, PALETTE["red"])

    def applyHidden_(self, reason):
        # WeChat gone or unreadable -> take the panel away (the app "opens with WeChat")
        self._render("status", reason, PALETTE["muted"])
        if self.panel.isVisible():
            self.panel.orderOut_(None)

    def applyPosition_(self, win):
        self._position_near(win)


def warn_if_no_generation_key() -> None:
    """Say it out loud at launch when the candidate half has no key behind it.

    The judgment half runs locally and needs nothing, so a panel with an empty candidate
    area reads as "the app is broken" rather than "I never configured this". One dialog at
    launch is the cheapest way to tell the two apart — it cannot be missed the way a line
    of grey text in a floating panel can.

    OPENAI_* and ANTHROPIC_* are two ways to configure the same generation layer, so this
    fires only when NEITHER is set: either one on its own is a complete configuration.
    TypeSafe is not checked — it has a local fallback, so it is never missing, only
    different.

    Drawn with osascript rather than NSAlert, which was measured to not work here: an
    accessory app cannot activate itself (NSApp.isActive stays False after
    activateIgnoringOtherApps_), and an NSAlert stayed isVisible=False even inside its own
    modal session — so the user would get nothing to click while the app sat in a modal
    loop, i.e. an app that looks hung. osascript's dialog belongs to a process that can
    activate, and Popen does not wait, so a dialog nobody dismisses cannot stall us.
    """
    if load_credentials()[1]:
        return
    path = str(userconfig.ENV_FILE).replace(str(Path.home()), "~")
    # AppleScript string escapes (\n) work inside the literal; keep it free of double quotes
    script = (
        'display alert "生成层还没配 Key，候选回复会是空的" message "'
        "意图和风险判断不受影响 —— 那部分跑在本地模型上，不需要 Key。\\n\\n"
        f"在下面的文件里填这两组中的任意一组（二选一即可），然后重启本应用：\\n{path}\\n\\n"
        "    OPENAI_API_KEY      （任意 OpenAI 兼容端点，如 DeepSeek）\\n"
        '    ANTHROPIC_API_KEY   （任意 Anthropic 兼容端点，如智谱）" as informational'
    )
    try:
        subprocess.Popen(["osascript", "-e", script],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass          # no osascript: the panel still shows the hint in the candidate area


def main() -> None:
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    warn_if_no_generation_key()
    controller = HudController.alloc().init()
    controller._show()
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        POLL_INTERVAL, controller, "tick:", None, True)
    AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, AppKit.NSDefaultRunLoopMode)
    app.run()


if __name__ == "__main__":
    main()
