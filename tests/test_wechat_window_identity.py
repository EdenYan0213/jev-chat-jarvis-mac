"""Offline regressions for #50: which windows count as WeChat, in both directions.

Run: uv run python -B -m unittest discover -s tests

Synthetic window lists only — no screen read, no WeChat process, no permissions, no
model calls. The owner strings below were sampled live on macOS 26 + WeChat 4.1.13,
where the app reports owner='微信'; 微信读书 and 微信输入法 also own titled, window-sized
windows, which is what makes them reachable as false positives.
"""
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import perception
from perception import find_wechat_window

# Apps whose display names merely contain 微信 / WeChat but are not WeChat itself.
SIBLING_APPS = ("微信读书", "微信输入法", "企业微信", "WeChatWork")


def window(owner, title, w, h, wid=1, pid=1):
    return {'kCGWindowOwnerName': owner, 'kCGWindowName': title,
            'kCGWindowNumber': wid, 'kCGWindowOwnerPID': pid,
            'kCGWindowBounds': {'X': 0, 'Y': 0, 'Width': w, 'Height': h}}


def find_with(windows, previous_wid=None):
    with patch('Quartz.CGWindowListCopyWindowInfo', return_value=windows):
        return find_wechat_window(previous_wid)


class _FakeApp:
    def __init__(self, name, bundle):
        self._name, self._bundle = name, bundle

    def bundleIdentifier(self):
        return self._bundle

    def localizedName(self):
        return self._name


class _FakeWorkspace:
    def __init__(self, app):
        self._app = app

    def frontmostApplication(self):
        return self._app


def frontmost(name, bundle=""):
    """frontmost_app_is_wechat() against a synthetic NSWorkspace; queries no real app."""
    appkit = types.ModuleType('AppKit')
    appkit.NSWorkspace = SimpleNamespace(
        sharedWorkspace=lambda: _FakeWorkspace(_FakeApp(name, bundle)))
    with patch.dict(sys.modules, {'AppKit': appkit}):
        return perception.frontmost_app_is_wechat()


class WeChatWindowIdentityTests(unittest.TestCase):
    def test_weixin_alias_owner_is_accepted(self):
        """The #50 case: a 4.x build reporting 'Weixin' must still be found.

        The substring test this replaces matched on 'WeChat' and '微信', so an owner of
        'Weixin' matched neither and the window list came back empty — the panel then
        just disappeared with no stated reason.
        """
        win = find_with([window('Weixin', 'Weixin', 949, 862, wid=5)])
        self.assertIsNotNone(win)
        self.assertEqual(win.wid, 5)

    def test_sibling_app_window_is_never_selected(self):
        """A titled 微信读书 window used to pass the owner filter and be OCR'd as chat."""
        for previous in (None, 9):
            with self.subTest(previous_wid=previous):
                self.assertIsNone(
                    find_with([window('微信读书', '微信读书', 1728, 990, wid=9)], previous))

    def test_siblings_are_rejected_one_by_one(self):
        for owner in SIBLING_APPS:
            with self.subTest(owner=owner):
                self.assertIsNone(find_with([window(owner, owner, 1728, 990)]))

    def test_every_declared_name_is_accepted_as_owner(self):
        """The declared list is exactly the accepted set — no name works only by accident."""
        for name in perception.WECHAT_APP_NAMES:
            with self.subTest(name=name):
                self.assertIsNotNone(find_with([window(name, name, 949, 862)]))

    def test_main_window_still_wins_over_larger_detached(self):
        """A detached WeChat window is bigger but must not outrank the main chat window."""
        detached = window('WeChat', '微信 (窗口)', 947, 679, wid=2)
        main = window('微信', '微信', 754, 593, wid=1)
        self.assertEqual(find_with([detached, main]).wid, 1)
        self.assertEqual(find_with([detached, main], previous_wid=2).wid, 1)

    def test_detached_wechat_window_still_eligible_when_main_absent(self):
        """Exact owner matching filters the app, not the window: detached windows keep
        owner 'WeChat', so the main-window-absent fallback still resolves."""
        detached = window('WeChat', '微信 (窗口)', 947, 679, wid=2)
        self.assertEqual(find_with([detached]).wid, 2)

    def test_sibling_never_revives_via_previous_wid(self):
        main = window('微信', '微信', 949, 862, wid=1)
        sibling = window('微信读书', '微信读书', 1728, 990, wid=9)
        self.assertEqual(find_with([main, sibling], previous_wid=9).wid, 1)

    def test_foreground_check_and_window_filter_accept_the_same_names(self):
        """These two call sites are what drifted apart in #50; this pins them together."""
        for name in perception.WECHAT_APP_NAMES:
            with self.subTest(accepted=name):
                self.assertTrue(frontmost(name))
                self.assertIsNotNone(find_with([window(name, name, 949, 862)]))
        for name in SIBLING_APPS:
            with self.subTest(rejected=name):
                self.assertFalse(frontmost(name))
                self.assertIsNone(find_with([window(name, name, 1728, 990)]))

    def test_bundle_id_still_identifies_wechat_under_any_display_name(self):
        self.assertTrue(frontmost('WeChat', bundle='com.tencent.xinWeChat'))
        self.assertTrue(frontmost('Some Locale Name', bundle='com.tencent.xinWeChat'))
        self.assertFalse(frontmost('微信读书', bundle='com.tencent.weread'))


if __name__ == '__main__':
    unittest.main()
