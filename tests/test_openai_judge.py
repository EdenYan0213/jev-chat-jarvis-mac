"""Structured judgment through the shared local OpenAI-compatible model."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge_openai import MAX_CONTEXT_CHARS, OpenAIJudge
import judge as judge_module


class OpenAIJudgeTests(unittest.TestCase):
    def test_parses_structured_judgment_and_sends_context(self):
        judge = OpenAIJudge(
            base="http://127.0.0.1:11434/v1",
            key="ollama",
            model="qwen3.5:4b",
        )
        judge._post = Mock(return_value={"choices": [{"message": {"content": """
```json
{"intent":"催进度","confidence":82,"risk":4.5,"emotion":"焦虑",
 "emotion_confidence":0.75,"emotion_intensity":3,"emotion_trend":"升温"}
```
"""}}]})

        result = judge.judge(
            "今天能给吗", context="我: 正在处理\n对方: 已经等了两天")

        body = judge._post.call_args.args[0]
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertIn("已经等了两天", body["messages"][1]["content"])
        self.assertEqual(result["intent"], "催进度")
        self.assertEqual(result["confidence"], 0.82)
        self.assertEqual(result["risk"], 4.5)
        self.assertEqual(result["emotion"], "焦虑")
        self.assertEqual(result["emotion_intensity"], 3.0)
        self.assertEqual(result["emotion_trend"], "升温")
        self.assertEqual(result["backend"], "openjev-style/qwen3.5:4b")

    def test_invalid_values_are_normalized_without_loading_fallback(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        judge._post = Mock(return_value={"choices": [{"message": {"content": """
{"intent":"unknown","confidence":-2,"risk":99,"emotion":"angry",
 "emotion_confidence":5,"emotion_intensity":9,"emotion_trend":"other"}
"""}}]})

        result = judge.judge("hello")

        self.assertEqual(result["intent"], "闲聊")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["risk"], 9.0)
        self.assertEqual(result["emotion"], "平静")
        self.assertEqual(result["emotion_confidence"], 0.05)
        self.assertEqual(result["emotion_intensity"], 4.0)
        self.assertEqual(result["emotion_trend"], "稳定")

    def test_oversized_context_keeps_summary_head_and_recent_tail(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        judge._post = Mock(return_value={"choices": [{"message": {"content": """
{"intent":"闲聊","confidence":0.8,"risk":1,"emotion":"平静",
 "emotion_confidence":0.8,"emotion_intensity":1,"emotion_trend":"稳定"}
"""}}]})
        context = (
            "SUMMARY_HEAD\n"
            + ("较早内容" * MAX_CONTEXT_CHARS)
            + "\nRECENT_TAIL"
        )

        judge.judge("最新消息", context=context)

        body = judge._post.call_args.args[0]
        sent = body["messages"][1]["content"]
        self.assertIn("SUMMARY_HEAD", sent)
        self.assertIn("RECENT_TAIL", sent)
        self.assertIn("中间较早上下文已压缩省略", sent)

    def test_candidate_ranking_is_uniform_and_requires_no_model_call(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        judge._post = Mock()

        ranked = judge.rank_candidates(
            "message", "闲聊", ["one", "two"])

        self.assertEqual(ranked, [
            {"text": "one", "prob": 0.5},
            {"text": "two", "prob": 0.5},
        ])
        judge._post.assert_not_called()

    def test_configured_backend_never_falls_back_to_decider(self):
        with (
            patch("judge_openai.configured", return_value=True),
            patch(
                "judge_openai.OpenAIJudge",
                side_effect=RuntimeError("bad shared-model config"),
            ),
            patch.object(judge_module, "Judge") as local_judge,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "bad shared-model config"):
                judge_module.make_judge()
        local_judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
