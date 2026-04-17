#!/usr/bin/env bash
# 生成 voiceScribe.app，双击即可打开 GUI
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DEST="$HOME/Applications/voiceScribe.app"
MACOS_DIR="$APP_DEST/Contents/MacOS"
RES_DIR="$APP_DEST/Contents/Resources"
VENV="$PROJECT_DIR/.venv"
PYTHON="$VENV/bin/python3"

# ── 检查虚拟环境 ─────────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "❌ 未找到虚拟环境，请先执行："
  echo "   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# ── 创建目录 ─────────────────────────────────────────────────────────────────
mkdir -p "$HOME/Applications"
rm -rf "$APP_DEST"
mkdir -p "$MACOS_DIR" "$RES_DIR"

# ── 启动脚本 ─────────────────────────────────────────────────────────────────
cat > "$MACOS_DIR/voiceScribe" << LAUNCHER
#!/usr/bin/env bash
export SSL_CERT_FILE=/usr/local/etc/openssl@1.1/cert.pem
export REQUESTS_CA_BUNDLE=/usr/local/etc/openssl@1.1/cert.pem
export HF_ENDPOINT=https://hf-mirror.com
exec "$PYTHON" "$PROJECT_DIR/app.py"
LAUNCHER
chmod +x "$MACOS_DIR/voiceScribe"

# ── Info.plist ────────────────────────────────────────────────────────────────
cat > "$APP_DEST/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>voiceScribe</string>
  <key>CFBundleDisplayName</key>       <string>voiceScribe</string>
  <key>CFBundleIdentifier</key>        <string>com.voicescribe.app</string>
  <key>CFBundleVersion</key>           <string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>voiceScribe</string>
  <key>CFBundleIconFile</key>          <string>AppIcon</string>
  <key>NSMicrophoneUsageDescription</key>
    <string>voiceScribe 需要访问麦克风进行实时语音识别</string>
  <key>LSMinimumSystemVersion</key>    <string>12.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
</dict>
</plist>
PLIST

# ── 生成图标（纯 Python，无需 Xcode）────────────────────────────────────────
"$PYTHON" - << 'PYICON'
import os, struct, zlib, pathlib

# 用最简方式生成一个 512×512 的 PNG 当图标
def make_png(size=512):
    def chunk(tag, data):
        c = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)

    pixels = []
    for y in range(size):
        row = b"\x00"  # filter byte
        for x in range(size):
            cx, cy = x - size // 2, y - size // 2
            r2 = cx * cx + cy * cy
            R = (size // 2) ** 2
            mic_r = (size // 5) ** 2
            body_h = size * 2 // 5
            if r2 <= R:                       # 圆形背景 (深蓝)
                row += bytes([15, 25, 60, 255])
            else:
                row += bytes([0, 0, 0, 0])
            # 简单麦克风形状留给系统默认，这里只做纯色圆
        pixels.append(row)

    raw = b"".join(pixels)
    compressed = zlib.compress(raw, 9)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")
    return header + ihdr + idat + iend

res = pathlib.Path(os.environ["HOME"]) / "Applications/voiceScribe.app/Contents/Resources"
res.mkdir(parents=True, exist_ok=True)
(res / "AppIcon.png").write_bytes(make_png())
print("  图标已生成")
PYICON

echo ""
echo "✅ voiceScribe.app 已创建：$APP_DEST"
echo ""
echo "   双击打开，或运行："
echo "   open ~/Applications/voiceScribe.app"
