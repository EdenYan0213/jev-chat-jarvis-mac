"""Emotion result contract shared by Jev and the local fallback judge."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotions import (
    EMOTIONS,
    EMOTION_INTENSITY_LEVELS,
    EMOTION_TRENDS,
    default_emotion_fields,
    format_emotion,
)
from judge import Judge
from judge_jev import JevJudge


class EmotionHelpersTests(unittest.TestCase):
    def test_default_fields_and_compact_display(self):
        self.assertEqual(default_emotion_fields(), {
            "emotion": "平静",
            "emotion_confidence": 0.0,
            "emotion_intensity": 0.0,
            "emotion_trend": "稳定",
        })
        self.assertEqual(format_emotion({
            "emotion": "焦虑",
            "emotion_intensity": 3.2,
            "emotion_trend": "升温",
        }), "焦虑 3/4 · 情绪升温")


class JevEmotionTests(unittest.TestCase):
    def judge(self, answers):
        judge = JevJudge(
            base="http://127.0.0.1:1", key="local", model="test")
        judge._post = Mock(return_value={"answers": answers})
        return judge

    def test_payload_and_valid_emotion_response(self):
        judge = self.judge({
            "emotion": {"choice": "焦虑", "confidence": 0.82},
            "emotion_intensity": {"score": 3.2},
            "emotion_trend": {"choice": "升温", "confidence": 0.73},
            "intent": {"choice": "催进度", "confidence": 0.8},
            "risk": {"score": 2.0},
        })

        result = judge.judge("今天能给吗", context="Alice 一直在等待")

        payload = judge._post.call_args.args[0]
        self.assertEqual(
            set(payload["questions"]),
            {"intent", "risk", "emotion",
             "emotion_intensity", "emotion_trend"})
        self.assertEqual(
            payload["questions"]["emotion"]["criteria"], EMOTIONS)
        self.assertEqual(
            payload["questions"]["emotion_intensity"]["criteria"],
            EMOTION_INTENSITY_LEVELS)
        self.assertEqual(
            payload["questions"]["emotion_trend"]["criteria"],
            EMOTION_TRENDS)
        self.assertIn("Alice 一直在等待", payload["state"])
        self.assertEqual(result["emotion"], "焦虑")
        self.assertEqual(result["emotion_confidence"], 0.82)
        self.assertEqual(result["emotion_intensity"], 3.2)
        self.assertEqual(result["emotion_trend"], "升温")
        self.assertEqual(result["intent"], "催进度")

    def test_malformed_emotion_fields_do_not_break_intent(self):
        judge = self.judge({
            "emotion": {"choice": "不是支持的情绪", "confidence": 9},
            "emotion_intensity": {"score": "strong"},
            "emotion_trend": {"choice": "继续变化"},
            "intent": {"choice": "派活", "confidence": 0.7},
            "risk": {"score": 4.0},
        })

        result = judge.judge("请处理一下")

        self.assertEqual(
            {key: result[key] for key in default_emotion_fields()},
            default_emotion_fields())
        self.assertEqual(result["intent"], "派活")
        self.assertEqual(result["risk"], 4.0)

    def test_numeric_emotion_fields_are_clamped(self):
        judge = self.judge({
            "emotion": {"choice": "开心", "confidence": 1.8},
            "emotion_intensity": {"score": 9},
            "emotion_trend": {"choice": "缓和"},
            "intent": {"choice": "闲聊", "confidence": 0.5},
            "risk": {"score": 1},
        })

        result = judge.judge("太好了")

        self.assertEqual(result["emotion_confidence"], 1.0)
        self.assertEqual(result["emotion_intensity"], 4.0)
        self.assertEqual(result["emotion_trend"], "缓和")


class LocalJudgeEmotionTests(unittest.TestCase):
    def test_local_judge_returns_same_emotion_shape(self):
        judge = object.__new__(Judge)
        judge._load = Mock()
        judge._forward = Mock(return_value=(
            list(range(5)), [0, 1, 2, 3, 4]))

        intent = np.zeros(8)
        intent[1] = 1.0
        risk = np.zeros(10)
        risk[3] = 1.0
        emotion = np.zeros(len(EMOTIONS))
        emotion[list(EMOTIONS).index("焦虑")] = 1.0
        intensity = np.zeros(len(EMOTION_INTENSITY_LEVELS))
        intensity[3] = 1.0
        trend = np.zeros(len(EMOTION_TRENDS))
        trend[list(EMOTION_TRENDS).index("升温")] = 1.0
        judge._slot_probs = Mock(side_effect=[
            intent, risk, emotion, intensity, trend])

        result = Judge.judge(
            judge, "还没好吗", context="对方已经连续询问两次")

        self.assertEqual(result["emotion"], "焦虑")
        self.assertEqual(result["emotion_confidence"], 1.0)
        self.assertEqual(result["emotion_intensity"], 3.0)
        self.assertEqual(result["emotion_trend"], "升温")
        prompt, slots = judge._forward.call_args.args
        self.assertEqual(slots, 5)
        self.assertIn("对方已经连续询问两次", prompt)
        self.assertIn("主情绪", prompt)
        self.assertIn("情绪趋势", prompt)


if __name__ == "__main__":
    unittest.main()
