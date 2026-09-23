# Persistent Conversation Sessions and Contextual Emotion Analysis

## Goal

Give the macOS assistant durable, local-only conversation memory. Both sides of a
WeChat conversation are saved in a named session, and both emotion judgment and
reply generation use the session's accumulated context instead of only the few
messages currently visible on screen.

The feature must preserve the project's read-only safety model: it may read the
visible WeChat window and fill text only after an explicit user action, but it
must never send a message or automatically scroll WeChat.

## Confirmed User Experience

- The app automatically binds the current WeChat chat title to its last active
  session.
- The HUD provides quick session switching and session creation.
- The settings window provides session renaming, deletion, recovery, and
  permanent removal.
- A chat can have multiple sessions. Reopening that chat resumes the last active
  session unless the user selects or creates another one.
- Creating a session imports the messages currently visible on screen. From that
  point forward, newly observed messages are saved incrementally.
- Deleting a session moves it to a local recycle bin. It can be restored for 30
  days, after which it is permanently removed.
- Judgment adds explicit primary emotion, intensity, and trend. Existing intent,
  risk, and action suggestions also use the full managed context.
- Complete raw history remains available locally. Compression changes only the
  prompt sent to a model; it never deletes source messages.

## Non-Goals

- Reading or decrypting the WeChat database.
- Automatically scrolling WeChat to import older history.
- Cloud sync, multi-device sync, or uploading the session database.
- Vector embeddings or semantic retrieval in the first version.
- Editing recognized history or automatically sending a generated reply.

## Architecture

### Session Store

Add a focused persistence module backed by the Python standard library's
`sqlite3`. The database lives at:

`~/Library/Application Support/jev-jarvis/conversations.sqlite3`

The containing directory is created with user-only permissions. SQLite uses WAL
mode and foreign keys. Store operations are serialized with a process-local
lock because OCR, judgment, summary work, and AppKit callbacks run on different
threads.

The module owns schema creation and versioned migrations through
`PRAGMA user_version`. HUD and settings code use its public methods rather than
issuing SQL directly.

### Conversation Tracker

The tracker receives each OCR snapshot together with its WeChat chat title. It:

1. Resolves the active session for that chat title.
2. Normalizes each recognized message while preserving `side`, `sender`, text,
   visible order, and observation time.
3. Aligns the visible sequence with the recent stored tail.
4. Appends only the unmatched suffix.

Sequence alignment is required because a single-message hash would incorrectly
collapse legitimate repeated replies such as two separate instances of
"收到". If the visible sequence already occurs in recent history, no messages
are appended. If an existing session has no overlap after an app restart or a
large gap, the snapshot is stored as a new observed segment and the session is
marked as having a possible unseen gap. This preserves all observed content
without claiming that the app captured messages that were never visible.

Only messages observed while the app is running are guaranteed to be stored.
The app does not manipulate WeChat to backfill history.

### Context Manager

One context builder serves both judgment and reply generation. Its rendered
context contains:

1. The latest rolling summary, when present.
2. Every unsummarized message after the summary boundary.
3. A recent raw-message reserve so the model always sees exact current wording,
   speaker labels, and conversational turns.
4. The new incoming message, supplied separately by the existing judge and
   generator interfaces to avoid duplication.

Speaker labels use the recognized group-chat sender when available, otherwise
`我`, `对方`, or `方向未确认`.

Context size is bounded by characters rather than provider-specific tokenizers.
The initial defaults are:

- Soft compression trigger: 8,000 rendered characters.
- Hard prompt limit: 12,000 rendered characters.
- Recent raw reserve: at least the newest 20 messages, subject to the hard limit.
- Summary target: at most 2,500 characters.

These constants remain internal in the first version. They can later become
advanced settings if real usage shows that users need provider-specific tuning.

### Rolling Summary

Compression runs on a single background worker and never blocks OCR or AppKit.
It starts at the soft threshold so a summary normally exists before the hard
limit is reached.

Each summary request receives:

- The prior summary.
- The oldest unsummarized messages that can be compacted.
- Instructions to retain people and relationships, decisions, promises,
  unresolved questions, important facts, boundaries, and emotional changes.

The newest raw-message reserve is not summarized away. On success, the store
atomically saves the new summary and the highest included message ID. On
failure, it keeps the previous summary and raw history, records only a
content-free diagnostic in the log, and retries after later messages arrive.
Until a first summary succeeds, the context builder uses the newest material
that fits the hard limit.

The configured OpenAI-compatible generation model performs summarization with a
low-temperature, non-streaming request. Summary work has lower priority than
reply generation so an Ollama cold start or busy local model does not delay the
visible reply workflow.

### Judgment Result

The judgment interface continues to return a dictionary but adds:

- `emotion`: a primary emotion label.
- `emotion_confidence`: confidence for that label.
- `emotion_intensity`: a normalized 0-4 score.
- `emotion_trend`: `缓和`, `稳定`, or `升温`.

The initial emotion choices cover `开心`, `平静`, `期待`, `困惑`, `焦虑`,
`委屈`, `生气`, `失望`, `悲伤`, and `疲惫`. The TypeSafe/Jev request asks
these questions alongside intent and risk in one structured call.

The bundled local fallback judge exposes the same result keys. If a backend
cannot produce a reliable emotion field, it returns `平静`, zero confidence,
zero intensity, and `稳定` rather than breaking reply generation.

The HUD displays a compact line such as `焦虑 3/4 · 情绪升温`, while retaining
the existing intent, confidence, risk, and action presentation.

## Data Model

### `sessions`

- `id`: stable UUID text primary key.
- `chat_key`: normalized WeChat chat title used for automatic binding.
- `name`: user-visible session name.
- `created_at`, `updated_at`, `last_active_at`: UTC timestamps.
- `deleted_at`: null for active sessions; timestamp while in the recycle bin.
- `summary_text`: latest rolling summary.
- `summary_until_message_id`: last raw message represented by the summary.
- `summary_version`: prompt/schema version for future regeneration.
- `has_observation_gap`: whether continuity could not be proven for a snapshot.

### `messages`

- `id`: integer primary key.
- `session_id`: owning session.
- `ordinal`: stable order within the session.
- `side`: `them`, `me`, or `unknown`.
- `sender`: nullable recognized sender.
- `text`: normalized message text.
- `observed_at`: UTC timestamp when the app saw it.
- `segment_id`: groups messages imported from one discontinuous snapshot.

### `chat_bindings`

- `chat_key`: primary key.
- `active_session_id`: last selected non-deleted session for that chat.

The store also records a small schema metadata table only if future migrations
need data beyond `PRAGMA user_version`.

## Session Lifecycle

- On first sight of a chat title, the app creates a default session named from
  the chat title and current local date.
- New-session creates another session for the same chat and immediately binds
  the HUD to it. The currently visible snapshot seeds the new session.
- Switching updates `chat_bindings`, invalidates in-flight judgment/generation,
  and re-analyzes the newest incoming message with the selected session context.
- Renaming affects only the session label, never the WeChat chat binding.
- Deleting the active session moves it to the recycle bin and switches to the
  most recently active remaining session. If none remains, a new default session
  is created.
- Restore returns a session to its original chat and makes it selectable.
- Startup and daily maintenance permanently purge sessions whose `deleted_at`
  is at least 30 days old.

## UI Design

The HUD chat header gains a compact session popup. It shows the current session,
lists other active sessions for the current WeChat chat, and provides:

- `新建 Session`
- `管理 Session`

Changing the popup never changes WeChat and never sends text. While a switch is
being applied, stale model results are rejected by the existing epoch guards.

The settings window gains a top-level segmented mode:

- `模型设置`
- `会话管理`

Session management shows active sessions and the recycle bin in separate tabs.
It supports search, rename, delete, restore, and explicit permanent deletion.
Destructive permanent deletion requires a native confirmation dialog. The
recycle-bin view shows the scheduled deletion date.

## Failure Handling and Privacy

- A database failure disables persistence for the current run but does not stop
  OCR, judgment, or reply generation. The HUD displays a concise local-storage
  warning.
- A summary failure does not modify its previous summary boundary.
- Malformed model emotion fields fall back independently; valid intent/risk
  results remain usable.
- Logs keep the existing privacy rule: no message text, summary text, candidate
  text, session name, or chat title.
- No network request is made solely for persistence. Summary and analysis
  requests use only the locally configured endpoints already chosen by the
  user.
- Database and WAL files are local and protected by macOS user-account file
  permissions. Database encryption is outside the first-version scope.

## Compatibility With the Installed Local Build

Before rebuilding the app, preserve and move the already-installed local
adaptations into the source repository:

- Shared relay credentials remain disabled.
- Judgment-only mode remains valid when reply generation is disabled.
- Local generation keeps the 90-second cold-start timeout.
- Local Laya ranking uses short candidate IDs to avoid oversized choice labels.
- Existing `OPENAI_EXTRA_BODY` behavior continues to disable thinking where
  configured.

These behaviors receive repository tests so rebuilding does not revert the
working local installation.

## Testing

Add offline tests for:

- Schema creation, migration, and user-only database location.
- Session create, bind, switch, rename, soft delete, restore, and 30-day purge.
- Sequence alignment with repeated identical messages, scrolling to an older
  visible sequence, and a discontinuous restart snapshot.
- Preserving both `me` and `them` messages and group sender names.
- Context rendering, summary boundaries, hard limits, and recent raw reserve.
- Atomic summary updates and summary failure retries.
- Jev emotion request shape and malformed-response fallbacks.
- HUD invalidation when switching sessions during judgment or generation.
- Settings actions without starting Cocoa in unit tests.
- All existing outgoing-message, settings, perception, and fill regressions.
- The installed-build compatibility changes listed above.

Finally, run a real-app smoke test against the local Laya service on
`127.0.0.1:18763` and Ollama on `127.0.0.1:11434`, verify that reopening the
same WeChat chat restores its last session, and verify that no action sends a
message automatically.
