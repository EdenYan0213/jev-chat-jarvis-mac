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

    def test_high_eq_humorous_friend_tone_has_realistic_boundaries(self):
        self.assertIn("高情商幽默朋友", styles.BUILTIN)
        prompt = styles.BUILTIN["高情商幽默朋友"]
        self.assertIn("贴着原话", prompt)
        self.assertIn("只有气氛轻松时", prompt)
        self.assertIn("不像客服", prompt)
        self.assertIn("不新增原因、安排、细节", prompt)
        complaint = styles.guidance_for("高情商幽默朋友", "批评")
        favor = styles.guidance_for("高情商幽默朋友", "求助帮忙")
        apology = styles.guidance_for("高情商幽默朋友", "道歉和解")
        self.assertIn("必须以「是我……」开头", complaint)
        self.assertIn("口味、时间、地点等细节不能猜", favor)
        self.assertIn("明确说收到了对方的道歉或诚意", apology)
        self.assertEqual(
            styles.TONE_TEMPERATURES["高情商幽默朋友"],
            0.35,
        )
        self.assertEqual(
            styles.strip_label("高情商幽默朋友：行，今晚给快递上个专车"),
            "行，今晚给快递上个专车",
        )

    def test_high_eq_humorous_friend_uses_lower_sampling_temperature(self):
        generator = generate.Generator()
        generator._call = Mock(return_value="我收到啦，咱慢慢说")

        texts, error = generator._one_tone(
            "刚才是我语气不好，对不起",
            "道歉和解",
            "高情商幽默朋友",
        )

        self.assertEqual(texts, ["我收到啦，咱慢慢说"])
        self.assertEqual(error, "")
        self.assertEqual(generator._call.call_args.kwargs["temperature"], 0.35)
        prompt = generator._call.call_args.args[0]
        self.assertIn("本条最高优先规则（必须遵守）", prompt)
        self.assertTrue(prompt.rstrip().endswith(
            styles.guidance_for("高情商幽默朋友", "道歉和解")))

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
