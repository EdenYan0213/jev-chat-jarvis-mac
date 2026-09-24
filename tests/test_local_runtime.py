"""Regressions for the local-only runtime shipped in the installed app."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import builtin
import generate
import styles
from judge_jev import JevJudge


class LocalRuntimeTests(unittest.TestCase):
    def test_shared_credentials_are_disabled(self):
        self.assertEqual(builtin.API_KEY, "")
        self.assertEqual(builtin.BASE_URL, "")
        self.assertEqual(builtin.MODEL, "")
        self.assertEqual(tuple(builtin.MODEL_PREFERENCE), ())
        self.assertEqual(builtin.EXTRA_BODY, "")

    def test_generation_can_be_disabled_without_loading_credentials(self):
        with patch.object(generate.userconfig, "get", return_value="0"), \
             patch.object(generate, "load_credentials") as load:
            result = generate.Generator().generate("synthetic")
        self.assertEqual(
            result, {"groups": [], "disabled": True, "elapsed_s": 0.0})
        load.assert_not_called()

    def test_warm_sunshine_tone_is_available_and_its_label_is_stripped(self):
        self.assertIn("暖心阳光男孩", styles.BUILTIN)
        prompt = styles.BUILTIN["暖心阳光男孩"]
        self.assertIn("接住情绪", prompt)
        self.assertIn("不油腻", prompt)
        self.assertEqual(
            styles.strip_label("暖心阳光男孩：我在，慢慢说"),
            "我在，慢慢说",
        )

    def test_humorous_boy_tone_is_available_and_its_label_is_stripped(self):
        self.assertIn("幽默男孩", styles.BUILTIN)
        prompt = styles.BUILTIN["幽默男孩"]
        self.assertIn("轻巧反转", prompt)
        self.assertIn("生活化比喻", prompt)
        self.assertIn("不嘲笑对方", prompt)
        self.assertIn("严肃、难过或正在冲突时收住玩笑", prompt)
        self.assertEqual(
            styles.strip_label("幽默男孩：这个需求像早高峰，挤一挤还是能上车"),
            "这个需求像早高峰，挤一挤还是能上车",
        )

    def test_hud_local_generator_allows_cold_start(self):
        tree = ast.parse((ROOT / "src" / "hud.py").read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Generator"
        ]
        self.assertTrue(any(
            any(keyword.arg == "timeout"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value >= 90
                for keyword in call.keywords)
            for call in calls))

    def test_local_rank_uses_short_ids_and_maps_back_to_text(self):
        judge = JevJudge(
            base="http://127.0.0.1:1", key="local", model="test")
        judge._post = Mock(return_value={"answers": {"best": {
            "probabilities": {"reply_1": 0.2, "reply_2": 0.8}}}})
        replies = ["first " * 40, "second " * 40]

        ranked = judge.rank_candidates("message", "intent", replies)

        payload = judge._post.call_args.args[0]
        self.assertEqual(
            payload["questions"]["best"]["criteria"],
            {"reply_1": None, "reply_2": None})
        self.assertIn(replies[0], payload["state"])
        self.assertIn(replies[1], payload["state"])
        self.assertEqual(ranked, [
            {"text": replies[1], "prob": 0.8},
            {"text": replies[0], "prob": 0.2},
        ])

    def test_choice_only_rank_maps_short_id_back_to_reply(self):
        judge = JevJudge(
            base="http://127.0.0.1:1", key="local", model="test")
        judge._post = Mock(return_value={"answers": {"best": {
            "choice": "reply_2", "confidence": 0.7}}})

        self.assertEqual(
            judge.rank_candidates("message", "intent", ["one", "two"]),
            [{"text": "two", "prob": 0.7},
             {"text": "one", "prob": 0.0}])


if __name__ == "__main__":
    unittest.main()
