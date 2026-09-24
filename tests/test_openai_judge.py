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
{"v":["工作","催进度",82,4.5,"焦虑",75,3,"升温"]}
```
"""}}]})

        result = judge.judge(
            "今天能给吗", context="我: 正在处理\n对方: 已经等了两天")

        body = judge._post.call_args.args[0]
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["reasoning_effort"], "none")
        sent = body["messages"][1]["content"]
        self.assertIn("已经等了两天", sent)
        self.assertIn(judge_module.INTENT_CHOICE_INSTRUCTION, sent)
        self.assertIn("同事聊生活也属于朋友场景", sent)
        self.assertIn("没有要求我执行任务，就绝不能选“派活”", sent)
        self.assertIn("求助帮忙=", sent)
        self.assertIn("朋友邀约=", sent)
        self.assertNotIn(judge_module.INTENTS["求助帮忙"], sent)
        self.assertEqual(result["intent"], "催进度")
        self.assertEqual(result["confidence"], 0.82)
        self.assertEqual(result["risk"], 4.5)
        self.assertEqual(result["emotion"], "焦虑")
        self.assertEqual(result["emotion_intensity"], 3.0)
        self.assertEqual(result["emotion_trend"], "升温")
        self.assertEqual(result["scene"], "工作")
        self.assertEqual(result["backend"], "openjev-style/qwen3.5:4b")

    def test_parses_social_intent_with_matching_action_guidance(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        judge._post = Mock(return_value={"choices": [{"message": {"content": """
{"p":"朋友","i":"倾诉求安慰","c":91,"r":2,"e":"委屈","ec":88,"s":3,"t":"升温"}
"""}}]})

        result = judge.judge(
            "今天真的好累，感觉谁都不理解我",
            context="朋友: 最近工作一直不顺\n我: 怎么了，愿意和我说说吗",
        )

        self.assertEqual(result["intent"], "倾诉求安慰")
        self.assertEqual(
            result["actions"],
            judge_module.ACTION_MAP["倾诉求安慰"],
        )
        self.assertEqual(result["emotion"], "委屈")
        self.assertEqual(result["scene"], "朋友")

    def test_private_scene_repairs_task_and_meeting_labels(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        responses = iter([
            {"choices": [{"message": {"content":
                '{"p":"私人社交","i":"派活","c":80,"r":1,"e":"平静",'
                '"ec":70,"s":0,"t":"稳定"}'}}]},
            {"choices": [{"message": {"content":
                '{"p":"朋友","i":"约会议","c":80,"r":1,"e":"期待",'
                '"ec":70,"s":1,"t":"稳定"}'}}]},
        ])
        judge._post = Mock(side_effect=lambda _body: next(responses))

        favor = judge.judge("能帮我取一下快递吗")
        invitation = judge.judge("周末一起吃饭吗")

        self.assertEqual(favor["scene"], "朋友")
        self.assertEqual(favor["intent"], "求助帮忙")
        self.assertEqual(invitation["intent"], "朋友邀约")
        self.assertEqual(
            judge_module.normalize_intent_for_scene("求助帮忙", "工作"),
            "求助帮忙",
        )

    def test_partial_compact_response_keeps_available_fields(self):
        judge = OpenAIJudge(base="http://local/v1", key="ollama", model="test")
        judge._post = Mock(return_value={"choices": [{"message": {"content":
            '{"v":["工作"],"i":["派活"],"c":90,"r":4,"e":["平静"],'
            '"ec":80,"s":1,"t":["稳定"]}'}}]})

        result = judge.judge("这个需求今天跟一下")

        self.assertEqual(result["scene"], "工作")
        self.assertEqual(result["intent"], "派活")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["emotion"], "平静")

    def test_every_intent_has_compact_criteria_and_action_guidance(self):
        self.assertEqual(
            set(judge_module.INTENTS),
            set(judge_module.INTENT_CHOICE_CRITERIA),
        )
        self.assertEqual(
            set(judge_module.INTENTS),
            set(judge_module.ACTION_MAP),
        )
        self.assertEqual(len(judge_module.INTENTS), 16)

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
        self.assertEqual(result["scene"], "不确定")

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

    def test_local_ollama_warm_loads_without_dummy_judgment(self):
        judge = OpenAIJudge(
            base="http://127.0.0.1:11434/v1",
            key="ollama",
            model="qwen3.5:4b",
        )
        with patch(
            "judge_openai.http_post_json",
            return_value={"done": True, "done_reason": "load"},
        ) as post:
            self.assertTrue(judge.warm())
        self.assertEqual(post.call_count, 1)
        first_url, _headers, first_body, timeout = post.call_args.args
        self.assertEqual(
            first_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(first_body["prompt"], "")
        self.assertEqual(first_body["keep_alive"], "30m")
        self.assertEqual(timeout, judge.timeout)

    def test_remote_openai_backend_does_not_try_ollama_warmup(self):
        judge = OpenAIJudge(
            base="https://example.invalid/v1",
            key="key",
            model="model",
        )
        with patch("judge_openai.http_post_json") as post:
            self.assertFalse(judge.warm())
        post.assert_not_called()

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
