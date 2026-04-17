# voiceScribe

实时语音转文字工具，基于本地 Whisper 模型，无需联网，支持麦克风和系统音频（视频字幕）。

## 特性

- 🎙 **流式实时字幕**：边说边纠正，句子未完成前持续刷新，定稿后换行
- 🖥 **系统音频捕获**：macOS 13+ 原生 ScreenCaptureKit，无需安装 BlackHole
- ⚡ **自动适配硬件**：Apple Silicon 使用 MLX 引擎（极速），Intel 使用 faster-whisper（CPU int8）
- 🔒 **完全本地**：模型在本机运行，音频不上传

## 硬件对应

| 硬件 | 引擎 | 默认模型 | 视频延迟 |
|------|------|---------|---------|
| Apple Silicon (M系列) | mlx-whisper | large-v3-turbo | < 0.5 秒 |
| Intel Mac | faster-whisper | medium.en | 1-2 秒 |

## 安装

```bash
# 1. 安装 ffmpeg
brew install ffmpeg

# 2. 一键安装（自动识别芯片）
cd voiceScribe
bash setup.sh

# 3. 运行
.venv/bin/python3 app.py
```

## 生成可点击的 App 图标

```bash
bash make_app.sh
open ~/Applications/voiceScribe.app
```

## 使用

### 麦克风模式（默认）
- 来源选「麦克风」，选择对应设备，点「▶ 开始」
- 对着麦克风说话，停顿后自动输出

### 系统音频模式（视频字幕）
- 来源选「系统音频 (ScreenCaptureKit)」
- 首次使用系统会弹出屏幕录制权限请求，允许即可
- 开启「视频模式」开关，点「▶ 开始」
- 播放任意视频，字幕自动输出

> macOS 12 及以下需要安装 [BlackHole](https://github.com/ExistingForge/BlackHole) 作为替代方案

## 界面说明

| 控件 | 说明 |
|------|------|
| 模型下拉 | 切换 Whisper 模型（首次使用自动下载）|
| 语言下拉 | en / zh / ja / ko / auto |
| 视频模式 | 开：固定窗口切片（连续音频）；关：VAD 停顿检测（说话）|
| 复制 | 复制全部字幕到剪贴板 |
| 清空 | 清空文本区 |
| 关闭窗口 | 隐藏到 Dock，模型保留在内存 |
| 退出按钮 | 真正退出，释放内存 |

## 目录结构

```
voiceScribe/
├── app.py            # 主程序（GUI）
├── system_audio.py   # ScreenCaptureKit 系统音频捕获
├── main.py           # CLI 版本
├── setup.sh          # 一键安装脚本
├── make_app.sh       # 生成 .app 图标
└── requirements.txt
```
