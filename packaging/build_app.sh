#!/bin/zsh
# Build jev-jarvis.app — a native macOS bundle around the Python app.
#
# Why a launcher bundle instead of py2app/PyInstaller: freezing torch + transformers
# produces a 2-4 GB app. This bundle stays ~100 KB: it carries the Python source and
# bootstraps a uv-managed virtualenv under ~/Library/Application Support on first launch.
#
# What the bundle buys you (the reason to do this at all):
#   * double-click launch, no terminal
#   * its own TCC identity — Screen Recording / Accessibility are granted to
#     "jev-jarvis", not to whatever terminal happened to start it
#   * LSUIElement: a floating helper, no Dock icon, never steals focus
#
# Usage:  ./packaging/build_app.sh          -> builds ./jev-jarvis.app
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/jev-jarvis.app"
BUNDLE_ID="info.jevjarvis.app"
VERSION="0.1.0"

echo "==> 清理旧包"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"

echo "==> 拷贝 Python 源码"
cd "$ROOT"
cp -R src "$APP/Contents/Resources/app/src"
cp pyproject.toml uv.lock README.md "$APP/Contents/Resources/app/"
[ -f .env.example ] && cp .env.example "$APP/Contents/Resources/app/"
# never ship local secrets or caches
rm -rf "$APP/Contents/Resources/app/src/__pycache__"

echo "==> 写 Info.plist"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>              <string>jev-jarvis</string>
    <key>CFBundleDisplayName</key>       <string>jev-jarvis</string>
    <key>CFBundleIdentifier</key>        <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>           <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleExecutable</key>        <string>jev-jarvis</string>
    <key>CFBundleIconFile</key>          <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>    <string>13.0</string>
    <!-- floating helper: no Dock icon, never becomes the active app -->
    <key>LSUIElement</key>               <true/>
    <key>NSHighResolutionCapable</key>   <true/>
    <!-- permission prompts are shown by the system; these strings explain why -->
    <key>NSScreenCaptureUsageDescription</key>
    <string>jev-jarvis 需要读取微信窗口的画面，才能在本地识别消息文字（不上传）。</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>jev-jarvis 需要把选中的回复粘贴到微信输入框。</string>
</dict>
</plist>
PLIST

echo "==> 写启动器"
cat > "$APP/Contents/MacOS/jev-jarvis" <<'LAUNCHER'
#!/bin/zsh
# Launcher: bootstrap the uv environment once, then exec the app.
set -u

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SUPPORT="$HOME/Library/Application Support/jev-jarvis"
CONFIG="$HOME/.config/jev-jarvis"
VENV="$SUPPORT/venv"
LOG="$HOME/Library/Logs/jev-jarvis.log"
mkdir -p "$SUPPORT" "$(dirname "$LOG")"

# Finder launches have a minimal PATH; add the usual install locations for uv
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# user-level env (API keys). Lives OUTSIDE the repo so it can never be committed, in the
# one location the README documents — a Finder launch inherits no shell environment at all,
# so sourcing here is the only chance to pick the keys up before Python also reads them.
[ -f "$CONFIG/env" ] && source "$CONFIG/env"

log() { print -r -- "[$(date '+%F %T')] $*" >> "$LOG" }

die() {  # show a native dialog, then exit
    log "FATAL: $1"
    osascript -e "display alert \"jev-jarvis 启动失败\" message \"$1\n\n详情: $LOG\" as critical" >/dev/null 2>&1
    exit 1
}

if ! command -v uv >/dev/null 2>&1; then
    log "未找到 uv，正在用官方脚本安装"
    # non-blocking: a Finder launch has no terminal, and a silent multi-minute wait
    # for uv + deps is indistinguishable from "the app is broken"
    osascript -e 'display notification "首次启动：正在安装 uv（约 10 MB）" with title "jev-jarvis"' >/dev/null 2>&1
    curl -LsSf https://astral.sh/uv/install.sh >>"$LOG" 2>&1
fi
# re-check rather than trust the installer: PATH above already covers ~/.local/bin
if ! command -v uv >/dev/null 2>&1; then
    die "未找到 uv，自动安装失败。请手动安装：brew install uv（或 curl -LsSf https://astral.sh/uv/install.sh | sh）"
fi

export UV_PROJECT_ENVIRONMENT="$VENV"
export USE_TF=0                  # laya/transformers: skip the TensorFlow probe
export HF_HUB_DISABLE_TELEMETRY=1

if [ ! -x "$VENV/bin/python" ]; then
    log "首次启动：正在创建虚拟环境并安装依赖（需要几分钟，请保持联网）"
    if ! uv sync --project "$RES/app" --quiet >>"$LOG" 2>&1; then
        die "依赖安装失败，请查看日志"
    fi
    log "依赖安装完成"
fi

log "启动 hud.py"
exec "$VENV/bin/python" "$RES/app/src/hud.py" >>"$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/jev-jarvis"

echo "==> 生成图标"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" "$ROOT/packaging/make_icon.py" "$APP/Contents/Resources/AppIcon.iconset" 2>/dev/null \
  && iconutil -c icns "$APP/Contents/Resources/AppIcon.iconset" \
       -o "$APP/Contents/Resources/AppIcon.icns" \
  && rm -rf "$APP/Contents/Resources/AppIcon.iconset" \
  && echo "    图标已生成" \
  || echo "    跳过图标（生成失败，不影响使用）"

echo "==> 完成"
du -sh "$APP" | awk '{print "    包体积: " $1}'
echo "    路径: $APP"
echo "    双击即可启动；首次启动会装依赖（几分钟）"
