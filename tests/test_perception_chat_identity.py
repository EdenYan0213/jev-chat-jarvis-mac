"""Conversation-title extraction and fallback identity regressions."""
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception import TextBlock, chat_key_for_snapshot, extract_chat_title


class ChatIdentityTests(unittest.TestCase):
    def test_short_title_beats_right_side_header_glyphs(self):
        blocks = [
            TextBlock("•。。", 0.30, 0.946, 0.949, 0.03, 0.02),
            TextBlock("小海（", 0.30, 0.390, 0.939, 0.08, 0.03),
            TextBlock("搜索", 1.0, 0.113, 0.939, 0.05, 0.03),
        ]

        self.assertEqual(extract_chat_title(blocks), "小海（")

    def test_fallback_identity_separates_windows_and_headers(self):
        first = chat_key_for_snapshot("", 10, "header-a")
        same = chat_key_for_snapshot("", 10, "header-a")
        other_window = chat_key_for_snapshot("", 11, "header-a")
        other_header = chat_key_for_snapshot("", 10, "header-b")

        self.assertEqual(first, same)
        self.assertNotEqual(first, other_window)
        self.assertNotEqual(first, other_header)
        self.assertTrue(first.startswith("未命名聊天 · "))

    def test_readable_title_is_used_directly(self):
        self.assertEqual(
            chat_key_for_snapshot("  文件传输助手  ", 10, "ignored"),
            "文件传输助手")


if __name__ == "__main__":
    unittest.main()
