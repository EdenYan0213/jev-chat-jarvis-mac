# Lifelike Fast Replies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate two natural, relationship-aware reply candidates in one local-model request while preserving user perspective, bounded context, streaming, and session history.

**Architecture:** Add a compact relationship signal to the existing judgment pass, then replace per-tone request fan-out with one slot-tagged generation prompt. A deterministic candidate cleaner handles formatting and duplicates without another model call, while the HUD passes the complete verdict to generation and keeps its existing stale-result guards.

**Tech Stack:** Python 3.12, unittest, PyObjC/AppKit, OpenAI-compatible Ollama API, TypeSafe/Jev API, sqlite3-backed conversation context.

---

## File Map

- Create `src/relationships.py`: shared relationship labels and normalization.
- Create `src/reply_quality.py`: deterministic candidate cleanup, deduplication, and quality flags.
- Create `probe/lifestyle_cases.json`: synthetic daily-conversation benchmark cases.
- Create `probe/lifelike_reply_benchmark.py`: warm-model latency and completion benchmark.
- Create `tests/test_relationship_judgment.py`: relationship contract across judge backends.
- Create `tests/test_multi_tone_generation.py`: one-call generation, parsing, streaming, and quality-guard tests.
- Create `tests/test_lifelike_benchmark.py`: benchmark statistics and comparison tests.
- Modify `src/judge.py`: add relationship as the seventh local decision slot.
- Modify `src/judge_jev.py`: add relationship to the existing Jev request.
- Modify `src/judge_openai.py`: append relationship to the compact local-model JSON.
- Modify `src/styles.py`: remove customer-service defaults and expose one combined-request temperature.
- Modify `src/generate.py`: build one prompt for all active slots and parse slot-tagged output.
- Modify `src/hud.py`: pass the full verdict into generation and retain it for tone changes.
- Modify `src/conversation_context.py`: park pending summary work while the foreground is busy.
- Modify `tests/test_emotion_judgment.py`: account for the added local decision slot.
- Modify `tests/test_openai_judge.py`: verify compact relationship parsing and fallback.
- Modify `tests/test_local_runtime.py`: verify the new natural-language prompt and temperature contract.
- Modify `tests/test_outgoing_messages.py`: verify full verdict propagation without extra calls.
- Modify `tests/test_conversation_context.py`: verify parked summary work is deduplicated.
- Modify `README.md`: document automatic relationship adaptation and single-pass candidates.

### Task 1: Add a Reproducible Lifestyle Performance Baseline

**Files:**
- Create: `probe/lifestyle_cases.json`
- Create: `probe/lifelike_reply_benchmark.py`
- Create: `tests/test_lifelike_benchmark.py`

- [ ] **Step 1: Write failing tests for benchmark statistics**

Create `tests/test_lifelike_benchmark.py`:

```python
from pathlib import Path
import importlib.util
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lifelike_reply_benchmark",
    ROOT / "probe" / "lifelike_reply_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class LifestyleBenchmarkTests(unittest.TestCase):
    def test_summary_uses_nearest_rank_p95_and_candidate_rate(self):
        rows = [
            {"judge_s": 1.0, "generation_s": float(i), "total_s": float(i + 1),
             "candidate_count": 2}
            for i in range(1, 11)
        ]
        rows[-1]["candidate_count"] = 1

        result = benchmark.summarize(rows)

        self.assertEqual(result["cases"], 10)
        self.assertEqual(result["generation_median_s"], 5.5)
        self.assertEqual(result["total_p95_s"], 11.0)
        self.assertEqual(result["two_candidate_rate"], 0.9)

    def test_compare_enforces_latency_and_completion_targets(self):
        baseline = {
            "generation_median_s": 10.0,
            "total_p95_s": 20.0,
        }
        after = {
            "generation_median_s": 7.4,
            "total_p95_s": 19.5,
            "two_candidate_rate": 0.9,
        }

        comparison = benchmark.compare(after, baseline)

        self.assertTrue(comparison["generation_target_met"])
        self.assertTrue(comparison["p95_target_met"])
        self.assertTrue(comparison["completion_target_met"])
        self.assertTrue(comparison["accepted"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the benchmark unit test and confirm it fails**

Run:

```bash
uv run python -B -m unittest tests.test_lifelike_benchmark -v
```

Expected: import failure because `probe/lifelike_reply_benchmark.py` does not
exist.

- [ ] **Step 3: Add ten synthetic daily-conversation cases**

Create `probe/lifestyle_cases.json` with these exact case names and fields:

```json
[
  {
    "name": "friend_tired",
    "message": "今天真的累麻了，回家只想躺平",
    "context": "我: 最近是不是又忙起来了\n对方: 连着加了三天班",
    "tones": ["高情商话术", "暖心阳光男孩"]
  },
  {
    "name": "friend_teasing",
    "message": "你这次居然没迟到，太阳打西边出来了",
    "context": "我: 我已经到门口了\n对方: 这么快？",
    "tones": ["高情商话术", "高情商幽默朋友"]
  },
  {
    "name": "casual_praise",
    "message": "你拍的这组照片真不错",
    "context": "我: 昨天出去随手拍了几张",
    "tones": ["高情商话术", "幽默男孩"]
  },
  {
    "name": "private_favor",
    "message": "下班能顺手帮我拿一下快递吗",
    "context": "对方: 柜子就在你公司楼下",
    "tones": ["高情商话术", "暖心阳光男孩"]
  },
  {
    "name": "friend_invitation",
    "message": "周六要不要一起去看电影",
    "context": "我: 最近那部新片好像还可以",
    "tones": ["高情商话术", "高情商幽默朋友"]
  },
  {
    "name": "concern",
    "message": "你今天感觉怎么样，好点了吗",
    "context": "我: 昨晚有点不舒服\n对方: 那你早点休息",
    "tones": ["高情商话术", "暖心阳光男孩"]
  },
  {
    "name": "apology",
    "message": "刚才是我说话太冲了，对不起",
    "context": "我: 你刚才那句话让我有点难受",
    "tones": ["高情商话术", "高情商幽默朋友"]
  },
  {
    "name": "colleague_smalltalk",
    "message": "今天地铁也太挤了，我差点没上来",
    "context": "我: 早高峰确实夸张",
    "tones": ["高情商话术", "幽默男孩"]
  },
  {
    "name": "formal_progress",
    "message": "这个版本今天几点可以给到",
    "context": "客户: 下午需要安排验收\n我: 我先确认一下当前进度",
    "tones": ["高情商话术", "稳如老狗"]
  },
  {
    "name": "friend_disappointed",
    "message": "说好了告诉我，结果你又忘了",
    "context": "我: 这次是我没及时说\n对方: 我等了半天",
    "tones": ["高情商话术", "暖心阳光男孩"]
  }
]
```

- [ ] **Step 4: Implement the benchmark probe**

Create `probe/lifelike_reply_benchmark.py` with these public helpers and CLI
behavior:

```python
"""Warm local-model benchmark for judgment plus two-candidate generation."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import userconfig
from generate import Generator
from judge_openai import OpenAIJudge


CASES_PATH = ROOT / "probe" / "lifestyle_cases.json"


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "judge_median_s": statistics.median(row["judge_s"] for row in rows),
        "generation_median_s": statistics.median(
            row["generation_s"] for row in rows),
        "total_p95_s": _p95([row["total_s"] for row in rows]),
        "two_candidate_rate": sum(
            row["candidate_count"] >= 2 for row in rows) / len(rows),
    }


def compare(after: dict, baseline: dict) -> dict:
    generation_ratio = (
        after["generation_median_s"] / baseline["generation_median_s"])
    result = {
        "generation_ratio": generation_ratio,
        "generation_target_met": generation_ratio <= 0.75,
        "p95_target_met": after["total_p95_s"] <= baseline["total_p95_s"],
        "completion_target_met": after["two_candidate_rate"] >= 0.9,
    }
    result["accepted"] = all(
        value for key, value in result.items() if key.endswith("_met"))
    return result


def _generate(generator, case: dict, verdict: dict) -> dict:
    kwargs = {}
    if "signals" in inspect.signature(generator.generate).parameters:
        kwargs["signals"] = verdict
    return generator.generate(
        case["message"],
        verdict["intent"],
        case["tones"] + ["不用"],
        case["context"],
        **kwargs,
    )


def run(cases: list[dict], show_text: bool = False) -> dict:
    judge = OpenAIJudge()
    generator = Generator(timeout=90)
    judge.keep_warm()

    warm = cases[0]
    warm_verdict = judge.judge(warm["message"], context=warm["context"])
    _generate(generator, warm, warm_verdict)

    rows = []
    for case in cases:
        started = time.perf_counter()
        judge_started = time.perf_counter()
        verdict = judge.judge(case["message"], context=case["context"])
        judge_s = time.perf_counter() - judge_started
        generation_started = time.perf_counter()
        generated = _generate(generator, case, verdict)
        generation_s = time.perf_counter() - generation_started
        texts = [
            text
            for group in generated.get("groups") or []
            for text in group.get("texts") or []
        ]
        row = {
            "name": case["name"],
            "judge_s": judge_s,
            "generation_s": generation_s,
            "total_s": time.perf_counter() - started,
            "candidate_count": len(texts),
        }
        if show_text:
            row["candidates"] = texts
        rows.append(row)
    return {"summary": summarize(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    userconfig.load()
    cases = json.loads(CASES_PATH.read_text())
    result = run(cases, show_text=args.show_text)
    if args.compare:
        baseline = json.loads(args.compare.read_text())["summary"]
        result["comparison"] = compare(result["summary"], baseline)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if "comparison" in result:
        print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
        return 0 if result["comparison"]["accepted"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the offline test**

Run:

```bash
uv run python -B -m unittest tests.test_lifelike_benchmark -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Stop the installed HUD and record the old implementation**

Run:

```bash
pkill -f '/Applications/jev-jarvis.app/Contents/Resources/app/src/hud.py' || true
pkill -f '/Applications/jev-jarvis.app/Contents/MacOS/jev-jarvis' || true
uv run python -B probe/lifelike_reply_benchmark.py \
  --output /tmp/jev-lifelike-baseline.json
```

Expected: ten rows are written, `two_candidate_rate` is at least `0.9`, and the
baseline file remains outside Git.

- [ ] **Step 7: Commit the benchmark harness**

```bash
git add probe/lifestyle_cases.json probe/lifelike_reply_benchmark.py \
  tests/test_lifelike_benchmark.py
git commit -m "test: add lifelike reply performance benchmark"
```

### Task 2: Add Relationship Familiarity to the Existing Judgment Pass

**Files:**
- Create: `src/relationships.py`
- Create: `tests/test_relationship_judgment.py`
- Modify: `src/judge.py:14-22, 257-346`
- Modify: `src/judge_jev.py:31-151`
- Modify: `src/judge_openai.py:15-261`
- Modify: `tests/test_emotion_judgment.py:118-164`
- Modify: `tests/test_openai_judge.py:16-166`

- [ ] **Step 1: Write failing normalization and backend tests**

Create `tests/test_relationship_judgment.py`:

```python
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotions import EMOTIONS, EMOTION_INTENSITY_LEVELS, EMOTION_TRENDS
from judge import CHAT_SCENES, INTENTS, Judge, RISK_LEVELS
from judge_jev import JevJudge
from relationships import (
    RELATIONSHIPS,
    RELATIONSHIP_CHOICE_CRITERIA,
    normalize_relationship,
)


class RelationshipHelpersTests(unittest.TestCase):
    def test_normalizes_supported_and_unknown_values(self):
        self.assertEqual(normalize_relationship("亲近"), "亲近")
        self.assertEqual(normalize_relationship("关系看起来很正式"), "正式")
        self.assertEqual(normalize_relationship("陌生"), "不确定")


class JevRelationshipTests(unittest.TestCase):
    def test_jev_adds_relationship_to_same_request(self):
        judge = JevJudge(
            base="http://127.0.0.1:1", key="local", model="test")
        judge._post = Mock(return_value={"answers": {
            "intent": {"choice": "闲聊", "confidence": 0.8},
            "risk": {"score": 1},
            "scene": {"choice": "朋友"},
            "relationship": {"choice": "熟悉"},
        }})

        result = judge.judge("今天地铁太挤了")

        payload = judge._post.call_args.args[0]
        self.assertEqual(
            payload["questions"]["relationship"]["criteria"],
            RELATIONSHIP_CHOICE_CRITERIA,
        )
        self.assertEqual(result["relationship"], "熟悉")
        judge._post.assert_called_once()


class LocalRelationshipTests(unittest.TestCase):
    def test_local_judge_uses_seventh_slot(self):
        judge = object.__new__(Judge)
        judge._load = Mock()
        judge._forward = Mock(return_value=(
            list(range(7)), list(range(7))))
        distributions = [
            np.eye(len(INTENTS))[list(INTENTS).index("闲聊")],
            np.eye(len(RISK_LEVELS))[1],
            np.eye(len(EMOTIONS))[list(EMOTIONS).index("平静")],
            np.eye(len(EMOTION_INTENSITY_LEVELS))[0],
            np.eye(len(EMOTION_TRENDS))[
                list(EMOTION_TRENDS).index("稳定")],
            np.eye(len(CHAT_SCENES))[list(CHAT_SCENES).index("朋友")],
            np.eye(len(RELATIONSHIPS))[list(RELATIONSHIPS).index("亲近")],
        ]
        judge._slot_probs = Mock(side_effect=distributions)

        result = Judge.judge(judge, "你可算来了")

        self.assertEqual(judge._forward.call_args.args[1], 7)
        self.assertEqual(result["relationship"], "亲近")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and confirm the module is missing**

Run:

```bash
uv run python -B -m unittest tests.test_relationship_judgment -v
```

Expected: import failure for `relationships`.

- [ ] **Step 3: Add the shared relationship taxonomy**

Create `src/relationships.py`:

```python
"""Shared relationship familiarity labels for every judgment backend."""
from __future__ import annotations


RELATIONSHIPS = {
    "正式": "陌生、客户、上下级或需要保持正式边界",
    "熟悉": "普通同事、熟人或日常朋友，可以自然口语交流",
    "亲近": "明确的亲近朋友、家人或长期亲密关系",
    "不确定": "上下文不足，无法可靠判断熟悉程度",
}

RELATIONSHIP_CHOICE_CRITERIA = {
    "正式": "正式或有身份边界",
    "熟悉": "普通熟人朋友同事",
    "亲近": "明确亲近关系",
    "不确定": "证据不足",
}


def normalize_relationship(value) -> str:
    text = str(value or "").strip()
    if text in RELATIONSHIPS:
        return text
    for name in ("正式", "熟悉", "亲近", "不确定"):
        if name in text:
            return name
    return "不确定"
```

- [ ] **Step 4: Add relationship to all three judge backends**

In `src/judge_jev.py`, add a `relationship` choice question beside `scene`,
parse it with `normalize_relationship`, and include it in the returned verdict.

In `src/judge_openai.py`, append relationship to the compact array without
moving existing indexes:

```python
'{"v":["场景名","意图名",意图置信度0到100,风险0到9,'
'"情绪名",情绪置信度0到100,强度0到4,"趋势名","关系名"]}。'
```

Parse it as:

```python
relationship_value = compact[8] if compact and len(compact) > 8 else result.get(
    "rel", result.get("relationship"))
relationship = normalize_relationship(relationship_value)
```

In `src/judge.py`, append a seventh question using `RELATIONSHIPS`, request
seven slots, derive `relationship_idx`, and return:

```python
"relationship": list(RELATIONSHIPS)[relationship_idx],
```

- [ ] **Step 5: Update existing judge tests**

Update compact mock output in `tests/test_openai_judge.py` to:

```python
{"v":["工作","催进度",82,4.5,"焦虑",75,3,"升温","正式"]}
```

Assert `result["relationship"] == "正式"` and assert malformed responses fall
back to `不确定`.

In `tests/test_emotion_judgment.py`, make the local `_forward` mock return seven
slots, append a relationship probability distribution to `_slot_probs`, and
assert the pre-existing emotion fields are unchanged.

- [ ] **Step 6: Run judgment tests and the intent regression**

Run:

```bash
uv run python -B -m unittest \
  tests.test_relationship_judgment \
  tests.test_emotion_judgment \
  tests.test_openai_judge -v
uv run python src/judge_zh_test.py
```

Expected: all unit tests pass; the 22-case intent regression does not lose a
previously correct intent because relationship is appended rather than
reordering existing decision slots.

- [ ] **Step 7: Commit relationship judgment**

```bash
git add src/relationships.py src/judge.py src/judge_jev.py \
  src/judge_openai.py tests/test_relationship_judgment.py \
  tests/test_emotion_judgment.py tests/test_openai_judge.py
git commit -m "feat: classify conversation familiarity"
```

### Task 3: Generate All Active Tones in One Model Request

**Files:**
- Create: `src/reply_quality.py`
- Create: `tests/test_multi_tone_generation.py`
- Modify: `src/generate.py:33-40, 148-174, 487-636`
- Modify: `src/styles.py:18-28, 97-184`
- Modify: `tests/test_local_runtime.py:31-125`

- [ ] **Step 1: Write failing one-call and streaming tests**

Create `tests/test_multi_tone_generation.py`:

```python
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import generate
from generate import Generator


class MultiToneGenerationTests(unittest.TestCase):
    def generator(self):
        generator = Generator()
        generator._creds = ("http://local/v1", "ollama", "test-model")
        return generator

    def test_two_tones_use_one_model_call(self):
        generator = self.generator()
        generator._call = Mock(return_value=(
            "<slot:0>\t我还好，你也别太累\n"
            "<slot:1>\t好多了，今天继续当个省电选手"
        ))

        with patch.object(generate, "generation_enabled", return_value=True):
            result = generator.generate(
                "你今天感觉怎么样",
                "关心问候",
                ["高情商话术", "暖心阳光男孩", "不用"],
                "对方: 昨天让你早点休息",
                signals={
                    "scene": "朋友",
                    "relationship": "熟悉",
                    "emotion": "平静",
                    "emotion_intensity": 1,
                    "risk": 1,
                },
            )

        self.assertEqual(generator._call.call_count, 1)
        self.assertEqual(
            [group["texts"] for group in result["groups"]],
            [["我还好，你也别太累"], ["好多了，今天继续当个省电选手"]],
        )
        kwargs = generator._call.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 96)
        self.assertEqual(kwargs["temperature"], 0.55)
        prompt = generator._call.call_args.args[0]
        self.assertIn("关系：熟悉", prompt)
        self.assertIn("<slot:0>", prompt)
        self.assertIn("<slot:1>", prompt)

    def test_first_complete_line_streams_before_second(self):
        generator = self.generator()
        events = []

        def fake_call(_prompt, on_delta, **_kwargs):
            on_delta("<slot:0>\t第一条先上屏\n")
            events.append("between")
            on_delta("<slot:1>\t第二条随后到")
            return "<slot:0>\t第一条先上屏\n<slot:1>\t第二条随后到"

        generator._call = Mock(side_effect=fake_call)
        with patch.object(generate, "generation_enabled", return_value=True):
            generator.generate(
                "消息", "闲聊",
                ["高情商话术", "暖心阳光男孩", "不用"],
                on_candidate=lambda slot, tone, text: events.append(
                    (slot, tone, text)),
            )

        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[1], "between")
        self.assertEqual(events[2][0], 1)

    def test_missing_or_duplicate_second_line_does_not_retry(self):
        generator = self.generator()
        generator._call = Mock(return_value=(
            "<slot:0>\t收到，我先确认一下\n"
            "<slot:1>\t收到，我先确认一下"
        ))

        with patch.object(generate, "generation_enabled", return_value=True):
            result = generator.generate(
                "今天能给吗", "催进度",
                ["高情商话术", "暖心阳光男孩", "不用"],
            )

        self.assertEqual(generator._call.call_count, 1)
        self.assertEqual(
            sum(len(group["texts"]) for group in result["groups"]), 1)
        self.assertTrue(result["groups"][1]["error"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the current fan-out fails**

Run:

```bash
uv run python -B -m unittest tests.test_multi_tone_generation -v
```

Expected: the first test observes two `_call` invocations or the `signals`
argument is unsupported.

- [ ] **Step 3: Add deterministic candidate cleanup**

Create `src/reply_quality.py`:

```python
"""Local-only formatting and diagnostics for generated reply candidates."""
from __future__ import annotations

import re


_NUMBERING = re.compile(r"^\s*\d+[.、)．]\s*")
_WRAPPING_QUOTES = re.compile(r"""^["“”「『'‘]+|["”」』'’]+$""")
_REPEATED_PUNCTUATION = re.compile(r"([。！？!?])\1+")
_SPACE = re.compile(r"\s+")

BOILERPLATE_PREFIXES = (
    "我理解你的感受",
    "听起来确实不容易",
    "感谢你的理解",
    "感谢你的分享",
)
UNGROUNDED_COMMITMENTS = (
    "我保证",
    "我一定",
    "马上给你",
    "肯定能",
)


def clean_candidate(text: str) -> str:
    value = _NUMBERING.sub("", str(text or "").strip())
    value = _WRAPPING_QUOTES.sub("", value).strip()
    value = _SPACE.sub(" ", value)
    value = _REPEATED_PUNCTUATION.sub(r"\1", value)
    return value.strip()


def dedupe_key(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?]", "", text).lower()


def usable(text: str) -> bool:
    return bool(text) and len(text) <= 80


def quality_flags(text: str, source: str = "") -> set[str]:
    flags = set()
    if text.startswith(BOILERPLATE_PREFIXES):
        flags.add("customer_service")
    if any(term in text and term not in source
           for term in UNGROUNDED_COMMITMENTS):
        flags.add("ungrounded_commitment")
    if text.count("?") + text.count("？") > 1:
        flags.add("question_heavy")
    return flags
```

- [ ] **Step 4: Replace the tone prompts and temperature contract**

In `src/styles.py`, replace the two default tone descriptions with:

```python
"高情商话术": (
    "像会说话但不端着的熟人：先对对方最新一句给真实反应，再说清我的态度或必要事实；"
    "能一句说完就不分两句，不固定先共情，也不为了显得周到强加建议、反问或下一步。"
    "不用客服套话，不擅自认错、承诺或安排时间。"
),
"暖心阳光男孩": (
    "像真诚温暖、相处自然的男生：关心落在对方刚说的具体事情上，语气轻松可靠；"
    "对方只是闲聊时就正常接话，不把每句话都回复成安慰或鸡汤。"
    "不油腻、不暧昧、不爹味，不擅自约见面或承诺后续行动。"
),
```

Replace `TONE_TEMPERATURES` with:

```python
GENERATION_TEMPERATURE = 0.55
```

- [ ] **Step 5: Implement one slot-tagged request in `src/generate.py`**

Replace `PROMPT_ONE` with a combined prompt that contains these exact
constraints:

```python
PROMPT_MULTI = """你正在替 App 用户本人拟微信回复。

身份：
- 「我」是 App 用户，也是候选回复发送者
- 「对方」是发来最新消息的人
- 绝不能替对方回复我，也不能把对方的经历写成我的

{context_line}对方最新消息：「{message}」
{signal_lines}
候选槽位：
{slot_lines}

要求：
- 先回应最新消息，前文只用于理解关系和事实
- 保留我的事实、利益和边界，不得擅自认错、答应、有空或承诺时间
- 通常 8-28 个汉字；必要事实最多 40 个汉字
- 贴近对方的正式程度和口语长度，不照抄对方
- 不使用「我理解你的感受」「听起来确实不容易」「感谢你的分享」等客服开场
- 不为了显得周到强加建议、追问、邀约或后续行动
- 两条不能只是同义改写
{serious_rule}

严格按槽位顺序输出，每个槽位一行：
<slot:槽位编号>	回复正文
不要编号、引号、解释或话术名称。"""
```

Add a slot parser using:

```python
_SLOT_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?<\s*slot\s*:\s*(\d+)\s*>\s*"
    r"(?:[:：|\t-]\s*)?(.*?)\s*$",
    re.IGNORECASE,
)
```

Build `signal_lines` only from present values, using lines such as
`场景：朋友`, `关系：熟悉`, `情绪：平静（强度 1/4）`, and `风险：1/9`.
Define:

```python
SERIOUS_EMOTIONS = {"焦虑", "委屈", "生气", "失望", "悲伤", "疲惫"}
```

When risk is at least `5`, or an emotion in `SERIOUS_EMOTIONS` has intensity at
least `2`, set:

```python
serious_rule = "- 当前是严肃模式：所有槽位禁止玩梗、调侃或轻佻表达"
```

Replace the thread pool in `Generator.generate` with one `_call`:

```python
raw = self._call(
    prompt,
    on_delta if on_candidate is not None else None,
    max_tokens=max(64, 48 * len(active)),
    temperature=styles.GENERATION_TEMPERATURE,
)
```

Parse tagged lines first, assign untagged lines to remaining active slots by
output order, clean with `reply_quality.clean_candidate`, strip echoed tone
labels with `styles.strip_label`, and remove duplicates by
`reply_quality.dedupe_key`. Keep the existing group shape:

```python
{"slot": slot, "tone": tone, "texts": [text], "error": ""}
```

For a missing or duplicate slot, return an empty `texts` list and:

```python
"error": "模型未返回可用候选"
```

Add `signals: dict | None = None` after `on_candidate` in the public
`generate` signature so existing positional callers remain compatible.

- [ ] **Step 6: Update generation regressions**

In `tests/test_local_runtime.py`, replace `_one_tone` assertions with one
`Generator.generate` call using mocked credentials and `_call`. Assert the
prompt contains:

```python
"「我」是 App 用户"
"先回应最新消息"
"不使用「我理解你的感受」"
"不得擅自认错"
```

Assert `styles.GENERATION_TEMPERATURE == 0.55` and retain all existing
perspective and intent-guidance checks.

- [ ] **Step 7: Run generation tests**

Run:

```bash
uv run python -B -m unittest \
  tests.test_multi_tone_generation \
  tests.test_local_runtime -v
```

Expected: all tests pass and the two-tone test observes exactly one `_call`.

- [ ] **Step 8: Commit single-pass generation**

```bash
git add src/reply_quality.py src/generate.py src/styles.py \
  tests/test_multi_tone_generation.py tests/test_local_runtime.py
git commit -m "feat: generate reply tones in one pass"
```

### Task 4: Pass Complete Judgment Signals Through the HUD

**Files:**
- Modify: `src/hud.py:270-321, 999-1128, 1749-1800, 1881-2014, 2040-2142`
- Modify: `tests/test_outgoing_messages.py:20-82, 300-355`

- [ ] **Step 1: Write failing HUD propagation tests**

Extend `test_shared_model_generation_receives_judged_intent` in
`tests/test_outgoing_messages.py`:

```python
verdict = {
    "intent": "玩笑调侃",
    "confidence": 0.9,
    "risk": 1,
    "scene": "朋友",
    "relationship": "亲近",
    "emotion": "开心",
    "emotion_intensity": 1,
}

Harness._run_generation(self.h, newest, verdict, "session context")

kwargs = self.h.generator.generate.call_args.kwargs
self.assertEqual(kwargs["signals"], verdict)
```

Add a tone-change test that calls `_regen_work` with the stored verdict and
asserts the same `signals` object reaches `generator.generate`.

- [ ] **Step 2: Run the focused HUD tests and confirm failure**

Run:

```bash
uv run python -B -m unittest \
  tests.test_outgoing_messages.OutgoingTests.test_shared_model_generation_receives_judged_intent \
  tests.test_outgoing_messages.OutgoingTests.test_tone_regeneration_reuses_full_verdict -v
```

Expected: `signals` is absent or `_regen_work` still accepts only an intent.

- [ ] **Step 3: Store and clear the newest verdict**

In `HudController.init`, add:

```python
self._last_verdict = {}
```

In `applyJudgment_`, set:

```python
self._last_verdict = dict(v)
```

Clear it wherever `_last_intent` and `_last_risk` are cleared, including
`applyWaiting_` and `_reset_for_session_change`.

- [ ] **Step 4: Pass signals into every post-judgment generation path**

Change `_regen_work` to accept `verdict` rather than a bare intent:

```python
def _regen_work(self, text: str, verdict: dict, slot_tones: list[str]):
    intent = verdict.get("intent", "")
    gen = self.generator.generate(
        text,
        intent,
        slot_tones,
        None,
        self._stream_hook(t0, "换话术"),
        signals=verdict,
    )
```

Change `_regenerate` to pass `dict(self._last_verdict)`.

Add `signals=None` to `_gen_with_pregen` and pass it only to a fresh
`generator.generate` call. In both `_run_generation` and the shared-model
branch of `_analyze`, call `_gen_with_pregen` with the current verdict. Keep
non-shared early generation unchanged so remote Jev and generation can still
overlap.

Update the generation log to report one request:

```python
_log(
    f"生成 {gen.get('elapsed_s', 0) * 1000:.0f}ms{note}"
    f" · 1 次请求/{len(groups)} 个话术"
    f" → {sum(len(g['texts']) for g in groups)} 条候选"
    + (f" · 失败: {'; '.join(failed)}" if failed else "")
)
```

- [ ] **Step 5: Run all outgoing-message tests**

Run:

```bash
uv run python -B -m unittest tests.test_outgoing_messages -v
```

Expected: stale epochs still suppress old stream lines, session changes still
invalidate work, and shared-model generation receives the complete verdict.

- [ ] **Step 6: Commit HUD integration**

```bash
git add src/hud.py tests/test_outgoing_messages.py
git commit -m "feat: use judgment signals in reply generation"
```

### Task 5: Park Summary Work Until the Foreground Is Idle

**Files:**
- Modify: `src/conversation_context.py:276-399`
- Modify: `tests/test_conversation_context.py:305-342`

- [ ] **Step 1: Write a failing parked-worker test**

Add to `SummaryWorkerTests`:

```python
def test_busy_session_stays_parked_and_deduplicated(self):
    busy = threading.Event()
    busy.set()
    worker = SummaryWorker(
        self.store, Mock(), retry_delay=0.01,
        interactive_busy=busy.is_set,
    )
    ran = threading.Event()
    worker.run_once = Mock(
        side_effect=lambda _session_id: ran.set() or True)
    worker.schedule = Mock(wraps=worker.schedule)
    worker.start()
    self.addCleanup(worker.stop)

    worker.schedule(self.session.id)
    time.sleep(0.04)
    self.assertEqual(worker.run_once.call_count, 0)
    self.assertEqual(worker.schedule.call_count, 1)

    busy.clear()
    self.assertTrue(ran.wait(0.5))
    time.sleep(0.03)
    self.assertEqual(worker.run_once.call_count, 1)
```

- [ ] **Step 2: Run the test and confirm the current requeue behavior fails**

Run:

```bash
uv run python -B -m unittest \
  tests.test_conversation_context.SummaryWorkerTests.test_busy_session_stays_parked_and_deduplicated -v
```

Expected: repeated scheduling/requeueing permits more than one pending cycle.

- [ ] **Step 3: Keep one parked item per session**

Add `self._dirty: set[str] = set()` beside `_pending`. Change `schedule`:

```python
with self._pending_lock:
    if session_id in self._pending:
        self._dirty.add(session_id)
        return
    self._pending.add(session_id)
self._queue.put(session_id)
```

Replace `_loop` with:

```python
def _loop(self) -> None:
    while not self._stopping.is_set():
        session_id = self._queue.get()
        if session_id is None:
            return
        while self.interactive_busy():
            if self._stopping.wait(self.retry_delay):
                return
        with self._pending_lock:
            self._dirty.discard(session_id)
        self.run_once(session_id)
        with self._pending_lock:
            if session_id in self._dirty:
                self._dirty.discard(session_id)
                self._queue.put(session_id)
            else:
                self._pending.discard(session_id)
```

This absorbs repeated snapshots while parked into the pending run. A snapshot
that arrives during the actual summary call marks the session dirty and earns
one later pass.

- [ ] **Step 4: Run all context tests**

Run:

```bash
uv run python -B -m unittest tests.test_conversation_context -v
```

Expected: all context and summary tests pass, including one pass per ordinary
schedule and one parked item while foreground work is active.

- [ ] **Step 5: Commit summary scheduling**

```bash
git add src/conversation_context.py tests/test_conversation_context.py
git commit -m "perf: park summary work until idle"
```

### Task 6: Add Lifestyle Quality Coverage and Documentation

**Files:**
- Modify: `tests/test_multi_tone_generation.py`
- Modify: `README.md`

- [ ] **Step 1: Add fixture-driven prompt and guard coverage**

Load `probe/lifestyle_cases.json` in `tests/test_multi_tone_generation.py` and
add:

```python
def test_every_lifestyle_case_builds_two_distinct_slots(self):
    cases = json.loads(
        (ROOT / "probe" / "lifestyle_cases.json").read_text())
    for case in cases:
        with self.subTest(case=case["name"]):
            generator = self.generator()
            generator._call = Mock(return_value=(
                "<slot:0>\t自然回复\n"
                "<slot:1>\t更贴近关系的回复"
            ))
            with patch.object(
                    generate, "generation_enabled", return_value=True):
                result = generator.generate(
                    case["message"],
                    "闲聊",
                    case["tones"] + ["不用"],
                    case["context"],
                    signals={
                        "scene": "朋友",
                        "relationship": "熟悉",
                        "emotion": "平静",
                        "emotion_intensity": 1,
                        "risk": 1,
                    },
                )
            prompt = generator._call.call_args.args[0]
            self.assertIn(case["message"], prompt)
            self.assertIn(case["context"], prompt)
            self.assertEqual(
                sum(len(group["texts"]) for group in result["groups"]), 2)
```

Add direct `reply_quality.quality_flags` assertions for customer-service
openings and ungrounded promises, and assert normal short replies have no
flags.

- [ ] **Step 2: Document the user-visible behavior**

Update `README.md` to state:

- judgment automatically distinguishes formal, familiar, close, and uncertain
  relationships from the bounded current session;
- two selected tones are generated in one request;
- the first candidate can appear before the second on streaming endpoints;
- serious or strongly negative conversations suppress humor;
- no review model or automatic retry is added;
- candidate text remains absent from logs.

- [ ] **Step 3: Run focused quality tests**

Run:

```bash
uv run python -B -m unittest \
  tests.test_multi_tone_generation \
  tests.test_relationship_judgment \
  tests.test_local_runtime -v
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 4: Commit quality fixtures and docs**

```bash
git add tests/test_multi_tone_generation.py README.md
git commit -m "docs: explain natural single-pass replies"
```

### Task 7: Verify Performance, Build, Install, Launch, and Push

**Files:**
- Modify only files required by failures discovered in this verification task.
- Build artifact: `jev-jarvis.app`
- Preserve: `~/Library/Application Support/jev-jarvis/conversations.sqlite3`
- Preserve: `~/.config/jev-jarvis/env`

- [ ] **Step 1: Run the post-change benchmark against the saved baseline**

Ensure the installed HUD remains stopped, then run:

```bash
uv run python -B probe/lifelike_reply_benchmark.py \
  --compare /tmp/jev-lifelike-baseline.json \
  --output /tmp/jev-lifelike-after.json \
  --show-text
```

Expected:

- `generation_target_met: true`
- `p95_target_met: true`
- `completion_target_met: true`
- `accepted: true`

Inspect the synthetic candidates for perspective reversal, invented
availability, customer-service openings, and humor during serious cases. Do
not proceed to packaging if the comparison exits nonzero.

- [ ] **Step 2: Run the complete offline and native smoke suites**

Run:

```bash
git diff --check
uv run python -B -m unittest discover -s tests -v
uv run python src/judge_zh_test.py
uv run python src/generate.py --check
uv run python -B probe/settings_smoke.py
```

Expected: all unit tests pass, judgment regression remains acceptable, the
configured generator resolves to local Ollama, and settings smoke completes
without personal credentials leaving the configuration file.

- [ ] **Step 3: Verify structural performance budgets**

Run:

```bash
rg -n "ThreadPoolExecutor|_one_tone|TONE_TEMPERATURES" src/generate.py src/styles.py
rg -n "max_tokens=max\\(64, 48 \\* len\\(active\\)\\)" src/generate.py
rg -n "hard_limit.: 1_800|recent_count.: 8|MAX_CONTEXT_CHARS = 1_200" \
  src/hud.py src/judge_openai.py
```

Expected: the first command has no matches; the second and third commands show
the bounded one-request implementation and unchanged context limits.

- [ ] **Step 4: Build the macOS app**

Run:

```bash
./packaging/build_app.sh
```

Expected: `jev-jarvis.app` is built successfully and its source bundle contains
`relationships.py` and `reply_quality.py`.

- [ ] **Step 5: Replace the installed app without retaining a local old copy**

The old source remains available in Git history on the Eden branch, so do not
create a local application backup. Run:

```bash
pkill -f '/Applications/jev-jarvis.app/Contents/Resources/app/src/hud.py' || true
pkill -f '/Applications/jev-jarvis.app/Contents/MacOS/jev-jarvis' || true
rm -rf /Applications/jev-jarvis.app
cp -R jev-jarvis.app /Applications/jev-jarvis.app
```

Do not modify or remove the conversation database or user env file.

- [ ] **Step 6: Launch and inspect the real runtime**

Run:

```bash
open /Applications/jev-jarvis.app
sleep 5
ps -axo pid,command | rg \
  '/Applications/jev-jarvis.app/Contents/(MacOS/jev-jarvis|Resources/app/src/hud.py)'
tail -80 "$HOME/Library/Logs/jev-jarvis.log"
```

Expected: launcher and HUD processes are present, no startup traceback appears,
and the log identifies the configured shared local model without downloading a
second model.

- [ ] **Step 7: Verify Git state and push only to Eden**

Run:

```bash
git status --short --branch
git log -8 --oneline --decorate
git push eden codex/persistent-sessions
```

Expected: the worktree is clean, all implementation commits are on
`codex/persistent-sessions`, and only
`github.com:EdenYan0213/jev-chat-jarvis-mac.git` is updated. Do not push
`origin`.

- [ ] **Step 8: Report measured results**

Report:

- baseline and new generation median;
- baseline and new end-to-end P95;
- two-candidate completion rate;
- total unit-test count;
- installed app path and active model;
- any live WeChat behavior that could not be verified without a new incoming
  message.
