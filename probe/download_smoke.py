"""Real local-model download + native HUD check, without reading/filling WeChat.

Run with an EMPTY, disposable HF cache (downloads ~3.8 GB):
  HF_HOME=/tmp/jev-download-check .venv/bin/python -B probe/download_smoke.py
Uses HF_HOME as-is; never deletes caches. Run again with --cached to check cache hits.
"""
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

if not os.environ.get('HF_HOME'):
    raise SystemExit('Set HF_HOME to a disposable cache directory first.')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import AppKit as A
from Foundation import NSDate
import userconfig
from judge import Judge, INTENTS

with patch.object(userconfig, 'load'), patch.object(userconfig, 'get', return_value=''):
    import hud
    app = A.NSApplication.sharedApplication()
    app.setActivationPolicy_(A.NSApplicationActivationPolicyRegular)
    judge = Judge()
    with patch.object(hud, 'make_judge', return_value=judge), patch.object(hud, 'Generator', return_value=Mock()):
        controller = hud.HudController.alloc().init()

controller._paused = True  # tick still updates model status, but never reads WeChat
controller.panel.center()
controller.panel.orderFrontRegardless()
controller._render('chat', '模型下载验证（不读取微信）')
results, errors = [], []


def load():
    try:
        judge.warm()
        result = judge.judge('现在进度怎么样了')
        results.append(result)
    except Exception as exc:
        errors.append(exc)


def capture(name):
    view = controller.panel.contentView()
    view.display()
    rep = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
    view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), rep)
    path = f'/tmp/jev-download-{name}.png'
    rep.representationUsingType_properties_(A.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(path, True)
    print('SCREENSHOT', path, flush=True)


worker = threading.Thread(target=load, daemon=True)
worker.start()
seen, captured = [], False
last_capture_percent = -10
while worker.is_alive():
    controller.tick_(None)
    text = str(controller.rows['status'].stringValue())
    if not seen or seen[-1] != text:
        seen.append(text)
        print('STATUS', text, flush=True)
    if '下载判断模型' in text:
        percent = int(text.split('%')[0].rsplit(' ', 1)[-1])
        if percent >= last_capture_percent + 10:
            capture('progress')
            last_capture_percent = percent
        captured = True
    A.NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(.1))
worker.join()
controller.tick_(None)
assert judge.load_status is None, 'load status was not cleared'
if errors:
    controller._render('status', '判断模型加载失败 · 请检查网络或磁盘空间', hud.PALETTE['red'])
    capture('failure')
    raise errors[0]
assert results and results[0]['intent'] in INTENTS
controller._render('status', '判断完成 · ' + results[0]['intent'], hud.PALETTE['green'])
capture('complete')
print('PASS: real model warm-up and judgment:', results[0]['intent'], flush=True)
assert captured == ('--cached' not in sys.argv), 'unexpected download visibility'
print('DOWNLOAD_VISIBLE', captured, flush=True)
controller.panel.orderOut_(None)
