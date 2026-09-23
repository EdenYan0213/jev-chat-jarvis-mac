"""Perception pure-function regressions on synthetic OCR: chat-title extraction and
message folding/side assignment. Run: python -B -m unittest discover -s tests -v.

No Cocoa, no screen read, no model calls; `block()` comes from tests/support_hud.py.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/: for support_hud
from perception import extract_chat_title, extract_messages
from support_hud import block


class PerceptionTests(unittest.TestCase):
    def test_short_chat_titles_survive_header_controls(self):
        for name in ('王', '张三', '李经理', 'A', '7', '项目讨论群'):
            with self.subTest(name=name):
                blocks = [block(name, .40, .94, .16, .025),
                          block('...', .92, .96, .03, .025),
                          block('口', .85, .94, .025, .025)]
                self.assertEqual(extract_chat_title(blocks), name)

    def test_title_fragments_join_without_distant_controls(self):
        blocks = [block('项目讨论', .40, .94, .12, .025),
                  block('组', .54, .945, .025, .025),
                  block('口口', .85, .95, .05, .025),
                  block('折叠聊天', .40, .905, .10, .025),
                  block('侧栏联系人', .10, .94, .15, .025),
                  block('下午开会', .40, .70, .15)]
        self.assertEqual(extract_chat_title(blocks), '项目讨论 组')
        self.assertEqual(extract_chat_title(blocks[2:]), '')

    def test_opposite_known_sides_never_fold_despite_close_edges(self):
        messages = extract_messages([block('收到的消息', .495, .70, .18),
                                     block('自己发出的消息', .51, .66, .34)])
        self.assertEqual([m.side for m in messages], ['them', 'me'])

    def test_incoming_wrapped_message_and_sender_preserved(self):
        messages = extract_messages([block('小王', .40, .80, .05, .020),
                                     block('第一行正文', .40, .65, .25),
                                     block('续行正文', .405, .61, .12)])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].side, 'them')
        self.assertEqual(messages[0].sender, '小王')
        self.assertEqual(len(messages[0].lines), 2)


if __name__ == '__main__':
    unittest.main()
