# Lifelike Replies With Single-Pass Generation

## Goal

Make candidate replies sound like something the App user would naturally send
to a friend, acquaintance, or colleague while reducing latency on the local
`qwen3.5:4b` runtime.

The default experience produces two visibly different candidates. Judgment and
generation remain local, bounded, and latest-message-wins. The change must not
add a model review pass, enlarge conversation context, or weaken the existing
rule that every candidate is written from the App user's point of view.

## Confirmed Experience

- The default result contains two candidates.
- Both candidates are produced by one generation request.
- The first completed candidate is displayed while the second is still being
  generated.
- With the default tone selection, candidate one is natural and steady while
  candidate two is warmer. Other selections continue to follow their chosen
  tones, and every result must remain directly sendable.
- Wording adapts automatically to work versus personal conversation, relative
  familiarity, emotion, and seriousness.
- The App does not require a new relationship setting or an additional model
  call.
- Existing session history and rolling summaries continue to provide context.

## Non-Goals

- Adding a second model to review or rewrite candidates.
- Increasing the judgment or generation context limits.
- Automatically sending a reply.
- Adding manual per-session relationship profiles in this iteration.
- Replacing the session database or changing stored message history.
- Guaranteeing two results by retrying malformed model output.

## Architecture

### One Judgment Pass

The existing structured judgment request continues to identify scene, intent,
risk, and emotion. It also returns a compact `relationship` value:

- `正式`: formal, unfamiliar, customer-like, or authority-sensitive.
- `熟悉`: ordinary colleague, acquaintance, or friend.
- `亲近`: clearly close friend, family member, or similarly close contact.
- `不确定`: insufficient evidence.

Relationship is inferred from the bounded conversation already supplied to the
judge. It is not inferred from a contact name alone. Unknown or malformed
values normalize to `不确定`.

The field is added to the same compact JSON response, so it does not create
another request. Other judgment backends expose the same result key and fall
back to `不确定` when they cannot provide it.

### One Dual-Candidate Generation Pass

Replace one-request-per-tone generation with one request containing all active
tone slots. The request receives:

- The newest incoming message.
- The existing bounded generation context.
- Intent, scene, relationship, emotion, intensity, and risk when available.
- Each active slot number, tone name, and tone instruction.

The model emits one line per active slot in a strict slot-tagged format. A
streaming parser maps each completed line back to the existing
`on_candidate(slot, tone, text)` callback, so the first line can appear before
the request finishes. The final parser also handles a last line without a
trailing newline and tolerates harmless numbering or echoed tone labels.

The generator preserves the current grouped return shape, keeping HUD and
settings behavior compatible. If only one slot is active, the same path makes
one request and returns one group. Duplicate tone selections remain valid and
still produce separate slot-tagged replies.

Generation uses a single moderate sampling temperature of `0.55` for the
combined request. Existing provider-specific request overrides remain
supported. The output budget is capped at 96 tokens for two candidates and
scaled only for a third explicitly enabled slot.

### Bounded Context

No context budget increases:

- Judgment remains capped at approximately 1,200 characters.
- Generation remains capped at approximately 1,800 characters with the newest
  eight raw messages retained by the existing context builder.

The combined generation prompt includes the conversation only once. This
removes the duplicate prompt evaluation currently paid by the two tone
requests on Ollama's single request slot.

### Summary Scheduling

Rolling summary work retains the existing 30-second idle requirement and stops
polling a busy session every retry interval. A session may be marked as needing
a summary immediately, but the pending item remains parked until it is eligible
and the model request starts only after at least 30 seconds without
incoming-message judgment or candidate generation.

New interactive work cancels or defers a summary as it does today. The worker
does not repeatedly chase every pending historical session; each session keeps
at most one pending summary request.

## Reply Behavior

The generation prompt follows these priorities in order:

1. Write as the App user replying to the person who sent the newest message.
2. Answer the newest message before referring to older context.
3. Preserve known facts, commitments, availability, and boundaries.
4. Match the relationship and conversational seriousness.
5. Apply the selected tone without turning the reply into a performance.

Natural-language rules are:

- Prefer a short WeChat sentence, normally 8-28 Chinese characters. Factual
  replies may extend to 40 characters when shortening would lose necessary
  information.
- Echo the other person's level of formality, use of names, and message length
  without copying their sentence.
- Avoid customer-service openings such as `我理解你的感受`,
  `听起来确实不容易`, and `感谢你的理解` unless the conversation genuinely
  requires that formal register.
- Do not force empathy, advice, a follow-up question, an invitation, or a future
  action merely to make the reply seem helpful.
- Do not invent shared experiences, reasons, times, places, nicknames, plans,
  apologies, or promises.
- In serious, high-risk, or strongly negative messages, humor is disabled.
- In light conversation, humor may use wording and circumstances already
  present in the conversation. It may use gentle self-mockery or a small
  reversal, but never ridicule the other person.
- `正式` uses clear, restrained wording; `熟悉` allows ordinary spoken Chinese;
  `亲近` may be shorter and more playful. `不确定` uses neutral friend-like
  wording without intimacy or nicknames.

The two default candidates must not be paraphrases. With the default tone
selection, candidate one favors a direct, steady response and candidate two is
warmer. A user-selected humorous tone may add a restrained joke while
preserving the same facts and boundaries.

## Local Quality Guard

A deterministic post-processor performs no model calls. It:

- Removes numbering, wrapping quotes, slot tags, and echoed tone labels.
- Normalizes whitespace and repeated punctuation.
- Rejects empty, label-only, implausibly long, and exact duplicate candidates.
- Detects common customer-service boilerplate and ungrounded commitment phrases
  for diagnostics and fixture tests.

The guard does not attempt semantic rewriting. If only one valid candidate is
returned, the HUD shows that candidate instead of silently launching another
request. Candidate text remains absent from application logs.

## Data Flow

1. OCR observes a new incoming message and updates its bound session.
2. The context builder produces the existing bounded context.
3. The judge returns scene, intent, relationship, risk, and emotion in one
   request.
4. The HUD displays judgment and passes its compact signals to the generator.
5. The generator sends one request containing both selected tones.
6. Each completed slot-tagged line streams to its existing HUD row.
7. Final parsing and local validation produce the grouped result.
8. The shared-model judge keeps generation order and performs no ranking pass.
9. After 30 idle seconds, summary work may use the model if compression is
   needed.

Epoch checks continue to discard judgment, streaming fragments, and final
results when the visible chat, session, message, or selected tones change.

## Failure Handling

- A missing or malformed relationship falls back to `不确定`.
- Missing optional judgment signals do not block generation.
- A malformed slot line is recovered by output order when unambiguous.
- One valid line is displayed even if the second line is missing.
- A completely unusable response surfaces the existing concise generation
  error and does not trigger a hidden retry.
- Transport-level stale keep-alive recovery remains unchanged.
- A deferred or interrupted summary leaves the prior summary boundary intact.
- Generation-disabled and remote-provider configurations retain their current
  behavior.

## Testing

Add offline tests for:

- Parsing and normalizing relationship output from each judgment backend.
- Falling back to `不确定` for malformed relationship values.
- Making exactly one generation call for two active tones.
- Preserving slot order and selected tone metadata in the grouped result.
- Streaming the first complete candidate before the second finishes.
- Parsing a final line without a newline and recovering simple malformed tags.
- Returning one valid candidate without issuing a retry.
- Rejecting duplicates while preserving distinct candidates.
- Keeping replies in the App user's perspective.
- Deferring summaries until the 30-second idle threshold.
- Cancelling stale work after message, session, or tone changes.

Create a small offline lifestyle fixture suite covering ordinary friend chat,
complaints, concern, invitations, teasing, praise, apologies, private favors,
colleague small talk, and formal work messages. Fixtures check for perspective
reversal, invented commitments, forced follow-up questions, repeated
candidates, and common customer-service boilerplate.

## Performance Verification

Record a warm-model baseline from the current implementation before changing
the generator. Compare the same messages, context, tones, Ollama settings, and
`qwen3.5:4b` model after implementation.

Automated tests enforce these structural budgets:

- At most one judgment request and one generation request per settled message.
- No model-based candidate ranking or review for the shared local model.
- Judgment and generation context limits remain unchanged.
- Dual-candidate output is capped at 96 tokens.
- Summary work cannot start during the 30-second interactive idle window.

The local benchmark uses at least ten representative lifestyle messages after
model warm-up. Acceptance criteria are:

- Median dual-candidate generation time is at least 25 percent lower than the
  recorded two-request baseline.
- End-to-end P95 is no slower than the recorded baseline.
- Both candidates are present in at least 90 percent of benchmark cases.
- The first candidate is streamed before request completion when the endpoint
  supports streaming.

If timing variance prevents the 25 percent target, the implementation is not
accepted merely because unit tests pass. Prompt size, parsing, and Ollama
request behavior must be profiled without adding model calls or reducing the
confirmed two-candidate experience.
