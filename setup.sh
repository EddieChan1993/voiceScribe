#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export SSL_CERT_FILE=/usr/local/etc/openssl@1.1/cert.pem
export REQUESTS_CA_BUNDLE=/usr/local/etc/openssl@1.1/cert.pem
export HF_ENDPOINT=https://hf-mirror.com

# 检测芯片
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    PYTHON_BIN=$(which python3.11 || which python3.12 || which python3.10 || echo "")
    CHIP="Apple Silicon"
else
    PYTHON_BIN="/usr/local/bin/python3.10"
    CHIP="Intel"
fi

echo "→ 检测到芯片：$CHIP ($ARCH)"
echo "→ Python：$PYTHON_BIN"

if [ -z "$PYTHON_BIN" ] || [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 未找到合适的 Python，请先安装 Python 3.10+"
    exit 1
fi

echo "→ 删除旧虚拟环境..."
rm -rf .venv

echo "→ 创建虚拟环境..."
"$PYTHON_BIN" -m venv .venv

echo "→ 升级 pip..."
.venv/bin/pip install --upgrade pip -q

echo "→ 安装基础依赖..."
.venv/bin/pip install sounddevice numpy customtkinter faster-whisper -q

if [ "$ARCH" = "arm64" ]; then
    echo "→ Apple Silicon 检测到，安装 mlx-whisper..."
    .venv/bin/pip install mlx-whisper -q
    echo "   MLX 引擎安装完成，将使用 large-v3-turbo 模型"
else
    echo "→ Intel 芯片，使用 faster-whisper + medium.en"
fi

echo ""
echo "✅ 安装完成！"
echo "   运行：.venv/bin/python3 app.py"
echo "   或：   bash make_app.sh && open ~/Applications/voiceScribe.app"
