# voiceScribe

macOS 本地实时语音转文字工具，基于 Whisper，完全离线，支持麦克风输入。

## 特性

- 🎙 **流式实时字幕**：边说边刷新，停顿后定稿换行，蓝色预览 / 白色定稿
- ⚡ **自动适配硬件**：Apple Silicon 使用 MLX 引擎（Neural Engine 加速），Intel 使用 faster-whisper（CPU int8）
- 🔒 **完全本地**：模型在本机运行，音频不上传
- 🛡 **幻觉过滤**：自动过滤 Whisper 在低质量音频下产生的重复词

## 硬件对应

| 硬件 | 引擎 | 推荐模型 |
|------|------|---------|
| Apple Silicon (M系列) | mlx-whisper | large-v3-turbo |
| Intel Mac | faster-whisper | medium.en |

## 安装

```bash
cd voiceScribe
bash setup.sh          # 自动检测芯片、创建 venv、安装依赖
```

### 手动安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
# 直接运行
.venv/bin/python3 app.py

# 或生成可双击的 .app（推荐）
bash make_app.sh
open ~/Applications/voiceScribe.app
```

## 界面说明

| 控件 | 说明 |
|------|------|
| 模型下拉 | 切换 Whisper 模型（MLX 仅显示已下载的模型）|
| 语言下拉 | en / zh / ja / ko / auto |
| 设备下拉 | 选择麦克风输入设备，↻ 刷新列表 |
| ▶ 开始 | 首次启动会加载模型（几秒），之后复用 |
| ⏹ 停止 | 停止录音，文字面板保留，不清空 |
| 复制 | 复制全部字幕到剪贴板 |
| 清空 | 手动清空文字面板（首次开始自动清一次）|
| 关闭窗口 | 隐藏到 Dock，模型保留在内存 |
| 退出 | 真正退出，释放内存 |

## VAD 参数（在 app.py 顶部调整）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SILENCE_RMS` | 80 | 低于此值视为静音（噪音大时调高）|
| `SILENCE_SEC` | 1.0 | 停顿多少秒提交定稿 |
| `MAX_WIN_SEC` | 8.0 | 单句最大时长（超过自动截断）|
| `REFRESH_SEC` | 0.35s / 1.0s | partial 刷新频率（MLX / Intel）|

## 目录结构

```
voiceScribe/
├── app.py            # 主程序（GUI）
├── system_audio.py   # ScreenCaptureKit 系统音频捕获（备用，暂未启用）
├── main.py           # CLI 版本
├── setup.sh          # 一键安装脚本
├── make_app.sh       # 生成 ~/Applications/voiceScribe.app
└── requirements.txt
```

## 已知问题 / TODO

- [ ] 系统音频（内录）功能代码已完成，待 ScreenCaptureKit 权限调试后启用
- [ ] Intel Mac 延迟约 1 秒，受 CPU 算力限制
- [ ] MLX 模型首次下载需要 hf-mirror 或 VPN（`HF_ENDPOINT=https://hf-mirror.com`）
- [ ] make_app.sh 生成的图标为纯色圆形，待替换为精美图标
