# Persistent Conversation Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local persistent sessions, rolling context summaries, and contextual emotion analysis to the macOS WeChat assistant without changing its read-only safety model.

**Architecture:** A new SQLite store owns sessions, messages, bindings, summaries, and the recycle bin. A conversation-context module aligns OCR snapshots with stored history, builds bounded prompts, and schedules summaries; the existing HUD, judge, generator, and settings surfaces consume these focused APIs.

**Tech Stack:** Python 3.12, sqlite3, threading, PyObjC/AppKit, TypeSafe-compatible Jev API, OpenAI-compatible local generation, unittest.

---

## File Map

- Create `src/conversation_store.py`: SQLite schema, session/message CRUD, bindings, summaries, recycle bin.
- Create `src/conversation_context.py`: OCR sequence alignment, observation mapping, bounded context rendering, summary worker.
- Create `src/emotions.py`: shared emotion labels, trends, normalization, and fallback fields.
- Create `src/session_settings.py`: testable session-management model and AppKit management pane.
- Modify `src/hud.py`: persist snapshots, use managed context, show emotion, and expose quick session controls.
- Modify `src/judge_jev.py`: ask and parse structured emotion questions and retain short candidate IDs.
- Modify `src/judge.py`: expose the same emotion result contract for the bundled local fallback.
- Modify `src/generate.py`: preserve local-generation controls and add low-temperature summarization.
- Modify `src/settings.py`: add top-level model/session mode and host the session pane.
- Modify `src/builtin.py`: keep shared credentials disabled in source.
- Modify `README.md`: document local history, session controls, compression, storage location, and deletion.
- Create `tests/test_local_runtime.py`: repository coverage for installed local adaptations.
- Create `tests/test_conversation_store.py`: database and lifecycle tests.
- Create `tests/test_conversation_context.py`: alignment, context limits, and summary tests.
- Create `tests/test_emotion_judgment.py`: Jev/local result-shape tests.
- Create `tests/test_session_settings.py`: management-model tests without starting Cocoa.
- Modify `tests/test_outgoing_messages.py`: managed-context and session-switch epoch regressions.

### Task 1: Preserve the Installed Local Runtime Behavior

**Files:**
- Create: `tests/test_local_runtime.py`
- Modify: `src/builtin.py`
- Modify: `src/generate.py`
- Modify: `src/hud.py`
- Modify: `src/judge_jev.py`

- [ ] **Step 1: Write failing tests for all installed-only changes**

```python
import ast
from pathlib import Path
from unittest.mock import Mock, patch

import builtin
import generate
from judge_jev import JevJudge


def test_shared_credentials_are_disabled():
    assert builtin.API_KEY == ""
    assert builtin.BASE_URL == ""
    assert builtin.MODEL == ""


def test_generation_can_be_disabled_without_loading_credentials():
    with patch.object(generate.userconfig, "get", return_value="0"), \
         patch.object(generate, "load_credentials") as load:
        result = generate.Generator().generate("synthetic")
    assert result == {"groups": [], "disabled": True, "elapsed_s": 0.0}
    load.assert_not_called()


def test_hud_local_generator_allows_cold_start():
    tree = ast.parse(Path("src/hud.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "Generator"]
    assert any(any(k.arg == "timeout" and k.value.value >= 90
                   for k in call.keywords) for call in calls)


def test_local_rank_uses_short_ids_and_maps_back_to_text():
    judge = JevJudge(base="http://127.0.0.1:1", key="local", model="test")
    judge._post = Mock(return_value={"answers": {"best": {
        "probabilities": {"reply_1": 0.2, "reply_2": 0.8}}}})
    replies = ["first " * 40, "second " * 40]
    ranked = judge.rank_candidates("message", "intent", replies)
    payload = judge._post.call_args.args[0]
    assert payload["questions"]["best"]["criteria"] == {
        "reply_1": None, "reply_2": None}
    assert ranked[0]["text"] == replies[1]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
uv run python -m unittest tests.test_local_runtime -v
```

Expected: failures for non-empty bundled credentials, missing
`generation_enabled`, 30-second HUD timeout, and long candidate labels.

- [ ] **Step 3: Port the installed changes into source**

Implement:

```python
def generation_enabled() -> bool:
    return userconfig.get("JEV_GENERATION_ENABLED").strip().lower() not in (
        "0", "false", "no", "off")
```

Return a disabled result before credential loading in `Generator.generate`,
reject direct `_call` attempts when disabled or keyless, instantiate
`Generator(timeout=90)` in the HUD, suppress generation-only errors and labels
when disabled, empty all constants in `builtin.py`, and map Jev ranking through
`reply_1`, `reply_2`, through `reply_<candidate_count>`.

- [ ] **Step 4: Run focused and existing generation tests**

Run:

```bash
uv run python -m unittest tests.test_local_runtime tests.test_settings tests.test_outgoing_messages -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the compatibility baseline**

```bash
git add src/builtin.py src/generate.py src/hud.py src/judge_jev.py tests/test_local_runtime.py
git commit -m "fix: preserve local model runtime behavior"
```

### Task 2: Add the SQLite Session Store

**Files:**
- Create: `src/conversation_store.py`
- Create: `tests/test_conversation_store.py`

- [ ] **Step 1: Write lifecycle and persistence tests**

Cover these concrete calls:

```python
store = ConversationStore(path=db, now=lambda: 1_700_000_000.0)
first = store.resolve_session("Alice")
assert first.chat_key == "Alice"
assert store.active_session("Alice").id == first.id

second = store.create_session("Alice", "Project B")
store.set_active_session("Alice", second.id)
store.append_messages(second.id, [
    MessageInput(side="them", sender="Alice", text="在吗"),
    MessageInput(side="me", sender=None, text="在"),
])
assert [m.text for m in store.recent_messages(second.id)] == ["在吗", "在"]

store.trash_session(second.id)
assert store.active_session("Alice").id == first.id
store.restore_session(second.id)
store.rename_session(second.id, "Restored")
assert store.get_session(second.id).name == "Restored"

store.trash_session(second.id)
assert store.purge_expired(retention_days=30, now=1_702_592_001.0) == 1
assert store.get_session(second.id, include_deleted=True) is None
```

Also assert `summary_text`, `summary_until_message_id`,
`has_observation_gap`, foreign-key cascades, `PRAGMA user_version == 1`, and
mode `0o600` for an explicitly created database file.

- [ ] **Step 2: Run the store tests and verify import failure**

Run:

```bash
uv run python -m unittest tests.test_conversation_store -v
```

Expected: `ModuleNotFoundError: conversation_store`.

- [ ] **Step 3: Implement schema and public records**

Create immutable records:

```python
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
```

Create schema version 1 with `sessions`, `messages`, and `chat_bindings`. Use
UUID strings for sessions/segments, real UTC timestamps, WAL, foreign keys, and
an `RLock`.

- [ ] **Step 4: Implement session and message methods**

Provide:

```python
resolve_session(chat_title: str) -> SessionRecord
create_session(chat_title: str, name: str | None = None) -> SessionRecord
active_session(chat_title: str) -> SessionRecord | None
set_active_session(chat_title: str, session_id: str) -> SessionRecord
list_sessions(chat_title: str | None = None, include_deleted: bool = False) -> list[SessionRecord]
get_session(session_id: str, include_deleted: bool = False) -> SessionRecord | None
rename_session(session_id: str, name: str) -> SessionRecord
trash_session(session_id: str) -> SessionRecord | None
restore_session(session_id: str) -> SessionRecord
delete_session_permanently(session_id: str) -> None
purge_expired(retention_days: int = 30, now: float | None = None) -> int
append_messages(session_id: str, messages: list[MessageInput],
                segment_id: str | None = None) -> list[StoredMessage]
recent_messages(session_id: str, limit: int = 100,
                after_id: int | None = None) -> list[StoredMessage]
mark_observation_gap(session_id: str) -> None
update_summary(session_id: str, text: str, until_message_id: int,
               version: int = 1) -> SessionRecord
```

Deleting the active session must rebind the chat to its most recently active
remaining session; if none remains, `resolve_session` creates a replacement.

- [ ] **Step 5: Run the store tests**

Run:

```bash
uv run python -m unittest tests.test_conversation_store -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the store**

```bash
git add src/conversation_store.py tests/test_conversation_store.py
git commit -m "feat: add local conversation session store"
```

### Task 3: Align OCR Snapshots and Build Managed Context

**Files:**
- Create: `src/conversation_context.py`
- Create: `tests/test_conversation_context.py`

- [ ] **Step 1: Write sequence-alignment tests**

Use lightweight visible messages with `side`, `sender`, and `text`. Verify:

```python
tracker.observe("Alice", [them("A"), me("B"), them("C")])
again = tracker.observe("Alice", [me("B"), them("C")])
assert again.appended == ()

extended = tracker.observe("Alice", [me("B"), them("C"), them("D")])
assert [m.text for m in extended.appended] == ["D"]

tracker.observe("Alice", [them("收到"), me("好"), them("收到")])
assert [m.text for m in store.recent_messages(session.id)][-3:] == [
    "收到", "好", "收到"]

older = tracker.observe("Alice", [them("A"), me("B")])
assert older.appended == ()

gap = tracker.observe("Alice", [them("完全不重叠")])
assert gap.gap_detected is True
assert store.get_session(session.id).has_observation_gap is True
```

Also assert that each visible message maps to a stored ID, including messages
that were already present.

- [ ] **Step 2: Write bounded-context tests**

Create a session with a summary and messages before/after its boundary. Verify:

```python
rendered = ContextBuilder(store, hard_limit=240, recent_count=3).build(
    session.id, exclude_message_id=target.id)
assert "会话摘要" in rendered.text
assert "最近原文" in rendered.text
assert target.text not in rendered.text
assert len(rendered.text) <= 240
assert rendered.message_count >= 3
```

Verify exact speaker labels, group sender names, the newest 20-message reserve
under normal limits, and deterministic truncation under a tiny hard limit.

- [ ] **Step 3: Run tests and verify the module is missing**

Run:

```bash
uv run python -m unittest tests.test_conversation_context -v
```

Expected: import failure.

- [ ] **Step 4: Implement observation records and alignment**

Create:

```python
@dataclass(frozen=True)
class Observation:
    session: SessionRecord
    visible_ids: tuple[int | None, ...]
    appended: tuple[StoredMessage, ...]
    gap_detected: bool


def message_key(message) -> tuple[str, str, str]:
    text = " ".join(str(message.text).split())
    return message.side, message.sender or "", text
```

The alignment order is:

1. If the full visible key sequence occurs contiguously in the latest 100 stored
   keys, map IDs and append nothing.
2. Otherwise find the longest stored-tail/visible-prefix overlap and append the
   remaining suffix.
3. For an empty session append everything without a gap.
4. For an existing session with no overlap append the snapshot as a new segment
   and mark a possible observation gap.

- [ ] **Step 5: Implement context rendering**

Create `ContextEnvelope(text, message_count, source_chars, truncated)` and
`ContextBuilder.build(session_id, exclude_message_id=None)`.

Render:

```text
会话摘要：
Alice 正在等明天下午三点的确认；我承诺整理完成后发送。

最近原文：
Alice: 明天下午三点能确认吗
我: 可以，我整理好后发你
```

Exclude the message currently under judgment by stored ID. Apply the hard limit
from the oldest raw line forward while preserving the configured recent count;
truncate the summary only after raw-line pruning is exhausted.

- [ ] **Step 6: Run context tests**

Run:

```bash
uv run python -m unittest tests.test_conversation_context -v
```

Expected: all alignment and context tests pass.

- [ ] **Step 7: Commit tracker and builder**

```bash
git add src/conversation_context.py tests/test_conversation_context.py
git commit -m "feat: persist OCR snapshots and build session context"
```

### Task 4: Add Background Rolling Summaries

**Files:**
- Modify: `src/generate.py`
- Modify: `src/conversation_context.py`
- Modify: `tests/test_conversation_context.py`
- Modify: `tests/test_settings.py`

- [ ] **Step 1: Add failing summarization tests**

Verify `Generator.summarize` sends a non-streaming request with low temperature,
a larger answer allowance, prior summary, and speaker-labelled messages. Verify
`SummaryWorker.run_once`:

- does nothing below 8,000 rendered characters;
- leaves the newest 20 messages raw;
- atomically advances `summary_until_message_id` on success;
- leaves summary and boundary unchanged on failure;
- defers when `interactive_busy()` is true.

Use a fake summarizer:

```python
class FakeSummarizer:
    def summarize(self, prior, messages):
        return "压缩后的摘要"
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
uv run python -m unittest tests.test_conversation_context tests.test_settings -v
```

Expected: missing `summarize` and `SummaryWorker`.

- [ ] **Step 3: Parameterize generator calls and add summarization**

Change `_call` to accept:

```python
def _call(self, prompt: str, on_delta=None, *,
          max_tokens: int = 300, temperature: float = 0.9) -> str:
```

Use those values for OpenAI and Anthropic payloads while preserving existing
generation defaults. Add:

```python
def summarize(self, prior_summary: str, messages: str) -> str:
    prompt = SUMMARY_PROMPT.format(
        prior=prior_summary or "无",
        messages=messages,
    )
    return self._call(
        prompt, max_tokens=800, temperature=0.2).strip()
```

The prompt must retain people, relationships, decisions, promises, unresolved
questions, facts, boundaries, and emotional changes; it must prohibit invented
facts and output only the updated summary.

- [ ] **Step 4: Implement `SummaryWorker`**

Provide synchronous `needs_summary(session_id)` and `run_once(session_id)` for
tests plus a daemon queue:

```python
schedule(session_id: str) -> None
start() -> None
stop() -> None
```

Deduplicate queued session IDs. If interactive work is busy, requeue after a
short wait. Summarize the oldest unsummarized chunk while reserving the newest
20 messages. Catch errors, call a content-free logger callback, and never move
the summary boundary on failure.

- [ ] **Step 5: Run summary and generation regressions**

Run:

```bash
uv run python -m unittest tests.test_conversation_context tests.test_settings tests.test_local_runtime -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit rolling summaries**

```bash
git add src/generate.py src/conversation_context.py tests/test_conversation_context.py tests/test_settings.py
git commit -m "feat: compress long session context in background"
```

### Task 5: Add Contextual Emotion Judgment

**Files:**
- Create: `src/emotions.py`
- Create: `tests/test_emotion_judgment.py`
- Modify: `src/judge_jev.py`
- Modify: `src/judge.py`
- Modify: `src/hud.py`

- [ ] **Step 1: Write Jev payload and parsing tests**

Mock `_post` and assert the payload contains `emotion`,
`emotion_intensity`, and `emotion_trend` alongside `intent` and `risk`.

Use:

```python
answers = {
    "emotion": {"choice": "焦虑", "confidence": 0.82},
    "emotion_intensity": {"score": 3.2},
    "emotion_trend": {"choice": "升温", "confidence": 0.73},
    "intent": {"choice": "催进度", "confidence": 0.8},
    "risk": {"score": 2.0},
}
```

Assert malformed or unknown values normalize independently to:

```python
{
    "emotion": "平静",
    "emotion_confidence": 0.0,
    "emotion_intensity": 0.0,
    "emotion_trend": "稳定",
}
```

- [ ] **Step 2: Write local judge shape and HUD formatting tests**

Patch the local forward pass with deterministic logits and assert all backends
return the same emotion keys. Extract a small pure helper from HUD:

```python
format_emotion(verdict) == "焦虑 3/4 · 情绪升温"
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
uv run python -m unittest tests.test_emotion_judgment -v
```

Expected: missing emotion module and result keys.

- [ ] **Step 4: Implement shared emotion definitions**

Create:

```python
EMOTIONS = {
    "开心": "明确表达愉快、满意、轻松或兴奋",
    "平静": "语气中性稳定，没有明显情绪波动",
    "期待": "对后续结果、回应或行动抱有正向期待",
    "困惑": "不理解、需要澄清或无法判断下一步",
    "焦虑": "担心结果、时间、风险或失去控制",
    "委屈": "感到被误解、被忽视、不公平或没有被照顾",
    "生气": "表达不满、责备、敌意或明显急躁",
    "失望": "期待落空，对人或结果感到不满意",
    "悲伤": "表达难过、失落、痛苦或告别感",
    "疲惫": "表达精力耗尽、厌倦、无力或想暂停",
}
EMOTION_TRENDS = {
    "缓和": "相比前文，负面情绪减弱或正面状态恢复",
    "稳定": "相比前文，情绪方向和强度基本没有变化",
    "升温": "相比前文，情绪强度增加或冲突正在升级",
}
```

Provide `default_emotion_fields()` and normalization helpers that clamp
confidence to 0-1 and intensity to 0-4.

- [ ] **Step 5: Extend Jev and local judges**

The Jev request asks emotion relative to the full supplied conversation,
intensity on 0-4, and trend compared with earlier turns. The local judge adds
three answer slots after intent/risk and maps option probabilities into the same
fields. `FallbackJudge` preserves valid primary fields while filling any missing
emotion keys.

- [ ] **Step 6: Add the compact HUD emotion row**

Add an `emotion` row to HUD construction and layout. `applyWaiting_` clears it;
`applyJudgment_` renders `format_emotion(verdict)`. Keep labels within the
360-pixel panel and grow the fixed header area rather than overlaying candidate
rows.

- [ ] **Step 7: Run judgment regressions**

Run:

```bash
uv run python -m unittest tests.test_emotion_judgment tests.test_outgoing_messages -v
uv run python src/judge_zh_test.py
```

Expected: unit tests pass; record the 22-case intent result and do not accept a
regression caused by the additional slots.

- [ ] **Step 8: Commit emotion analysis**

```bash
git add src/emotions.py src/judge_jev.py src/judge.py src/hud.py tests/test_emotion_judgment.py tests/test_outgoing_messages.py
git commit -m "feat: add contextual emotion judgment"
```

### Task 6: Integrate Persistent Context Into the HUD

**Files:**
- Modify: `src/hud.py`
- Modify: `tests/test_outgoing_messages.py`

- [ ] **Step 1: Add failing HUD integration tests**

Extend the AST harness with fake store/tracker/context objects. Verify:

- every changed OCR snapshot persists both sides;
- unchanged frames do not write again;
- judge and generator receive the same managed session context;
- the target incoming message is excluded from context by stored ID;
- switching sessions increments `_reply_epoch`, clears early results, and
  triggers reanalysis;
- a store exception falls back to `_context_text` and emits a storage warning
  without stopping analysis.

- [ ] **Step 2: Run the HUD tests and confirm failure**

Run:

```bash
uv run python -m unittest tests.test_outgoing_messages -v
```

Expected: fake tracker/context methods are never called.

- [ ] **Step 3: Initialize session services**

In `HudController.init`, create:

```python
self.session_store = ConversationStore()
self.conversation_tracker = ConversationTracker(self.session_store)
self.context_builder = ContextBuilder(self.session_store)
self.summary_worker = SummaryWorker(
    self.session_store,
    Generator(timeout=90),
    interactive_busy=lambda: (
        self._analyzing or self._prejudging or self._pregen_running),
    logger=_log,
)
```

Purge expired sessions and start the summary worker. On initialization failure,
store `None` services and keep the existing screen-only context path.

Record the last purge time and repeat `purge_expired()` from the existing tick
loop no more than once every 24 hours. Stop the summary worker during normal app
termination so no database work remains in flight while the process exits.

- [ ] **Step 4: Persist only fresh OCR snapshots**

Before replacing an unchanged read with `_last_full`, retain a boolean showing
whether OCR produced a fresh snapshot. On fresh reads call
`tracker.observe(chat_title, msgs)` and keep the returned visible stored IDs.
Resolve the newest incoming message's stored ID by its visible index.

Replace all judge/generator `_context_text` calls for that message with one
managed context string. Continue using `_context_text` only as the explicit
fallback when persistence is unavailable.

- [ ] **Step 5: Schedule summary work**

After appending a fresh snapshot call `summary_worker.schedule(session.id)`.
The worker itself checks the threshold, so the OCR path performs no model work.

- [ ] **Step 6: Run HUD regressions**

Run:

```bash
uv run python -m unittest tests.test_outgoing_messages tests.test_conversation_context -v
```

Expected: all tests pass and existing latest-wins behavior remains intact.

- [ ] **Step 7: Commit HUD persistence**

```bash
git add src/hud.py tests/test_outgoing_messages.py
git commit -m "feat: use persistent session context in the HUD"
```

### Task 7: Add Quick Session Controls and Session Management

**Files:**
- Create: `src/session_settings.py`
- Create: `tests/test_session_settings.py`
- Modify: `src/hud.py`
- Modify: `src/settings.py`

- [ ] **Step 1: Write management-model tests**

Test a Cocoa-free `SessionManagerModel`:

```python
model = SessionManagerModel(store)
assert model.active_rows(search="alice")[0].name == "Alice · 2026-09-23"
model.rename(session.id, "客户跟进")
model.trash(session.id)
assert model.trash_rows()[0].deletes_at > model.trash_rows()[0].deleted_at
model.restore(session.id)
model.delete_permanently(session.id)
```

Verify search is case-insensitive, active/trash lists are separated, and
permanent deletion removes messages through foreign-key cascade.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run python -m unittest tests.test_session_settings -v
```

Expected: missing `session_settings`.

- [ ] **Step 3: Implement the testable model and AppKit pane**

`SessionManagerModel` wraps store methods and returns row records with chat,
name, updated time, deletion time, and scheduled purge time.

`SessionManagerPane` builds:

- a search field;
- `使用中` and `回收站` tabs;
- a table/list of sessions;
- rename, delete, restore, and permanent-delete actions;
- a native confirmation alert before permanent deletion.

Never render message or summary text in the management list.

- [ ] **Step 4: Add the HUD session popup**

Place a compact `NSPopUpButton` in the chat header. Populate it with active
sessions for the current chat plus separators and:

```text
新建 Session…
管理 Session…
```

Represent existing sessions by their UUID. Selecting one calls
`set_active_session`, increments the reply epoch, clears early work, refreshes
the label, and forces a fresh analysis. New session uses the current local date
in its default name and seeds from the currently visible snapshot.

- [ ] **Step 5: Add top-level settings mode**

Add an `NSSegmentedControl` with `模型设置` and `会话管理`. Keep the current
model controls in one container and host `SessionManagerPane` in another.
`管理 Session…` from the HUD opens settings directly in session mode.

- [ ] **Step 6: Run management and HUD tests**

Run:

```bash
uv run python -m unittest tests.test_session_settings tests.test_outgoing_messages tests.test_settings -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit session UI**

```bash
git add src/session_settings.py src/settings.py src/hud.py tests/test_session_settings.py tests/test_outgoing_messages.py
git commit -m "feat: add session switching and management"
```

### Task 8: Documentation, Full Verification, Build, and Launch

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` only if a release version bump is required for the
  local build.

- [ ] **Step 1: Document user-visible behavior**

Add concise sections covering:

- where sessions appear in the HUD;
- how to create, switch, rename, delete, and restore;
- that only visible messages are captured;
- that both sides are saved locally;
- that compression preserves raw history;
- the database path;
- the 30-day recycle bin;
- how to disable generation while retaining judgment;
- the guarantee that nothing is automatically sent.

- [ ] **Step 2: Run formatting and the complete offline suite**

Run:

```bash
git diff --check
uv run python -m unittest discover -s tests -v
uv run python src/judge_zh_test.py
uv run python src/generate.py --check
```

Expected: no whitespace errors, all unit tests pass, judgment regression remains
acceptable, and local credentials resolve to the intended Laya/Ollama endpoints.

- [ ] **Step 3: Run local service smoke checks**

Run the repository's local judgment and rank checks against
`127.0.0.1:18763`, then make one summarization/generation request against
`127.0.0.1:11434`. Expected: structured emotion fields, short-ID ranking, and a
non-empty Chinese summary without thinking-only output.

- [ ] **Step 4: Build the macOS application**

Run:

```bash
./packaging/build_app.sh
```

Expected: a new `jev-jarvis.app` bundle containing the repository sources and
no shared relay credentials.

- [ ] **Step 5: Install without losing the working app**

Quit the running app, move the current `/Applications/jev-jarvis.app` to a
timestamped backup under `~/Library/Application Support/jev-jarvis/backups/`,
copy the new bundle to `/Applications/jev-jarvis.app`, and retain the existing
`~/.config/jev-jarvis/env` unchanged.

- [ ] **Step 6: Launch and verify the real workflow**

Launch `/Applications/jev-jarvis.app` and verify from logs/UI:

1. Laya is reached at `127.0.0.1:18763`.
2. Ollama uses `jev-qwen3.5:4b` at `127.0.0.1:11434`.
3. Both sides of a visible chat are stored.
4. Reopening the chat restores the last session.
5. Creating and switching sessions invalidates stale results.
6. Emotion, intensity, and trend appear.
7. A forced small threshold produces a summary while raw messages remain.
8. Delete/restore works and no action sends a message.

- [ ] **Step 7: Commit docs and final fixes**

```bash
git add README.md pyproject.toml src tests
git commit -m "docs: document persistent conversation sessions"
```

- [ ] **Step 8: Report the installed build and verification**

Report the app path, database path, active local endpoints, test count, and any
manual verification that could not be completed without a live WeChat chat.
