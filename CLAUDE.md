# voiceScribe 项目上下文

## 项目简介
macOS 本地实时语音转文字工具，支持麦克风和系统音频（视频字幕），自动适配 Apple Silicon / Intel。

## 技术栈
- **语言**：Python 3.10+
- **GUI**：customtkinter + tkinter（深色主题）
- **推理引擎**：
  - Apple Silicon → `mlx-whisper`（MLX / Neural Engine）
  - Intel → `faster-whisper`（CTranslate2 / CPU int8）
- **音频采集**：
  - 麦克风：`sounddevice`
  - 系统音频：macOS `ScreenCaptureKit`（无需 BlackHole）
- **VAD**：能量 RMS 检测 + 最大段时长兜底

## 目录结构
```
app.py            主程序，GUI + 控制逻辑
system_audio.py   ScreenCaptureKit 系统音频捕获模块
main.py           CLI 版本（无 GUI）
setup.sh          安装脚本，自动检测芯片
make_app.sh       生成 ~/Applications/voiceScribe.app
requirements.txt  Python 依赖
```

## 核心架构
```
sounddevice / SystemAudioCapture
        ↓ int16 chunks (16kHz)
    audio_queue (Queue)
        ↓
    VAD Loop (_vad_mic / _vad_video)
        ↓ numpy arrays
    TranscriptionWorker (单线程)
        ↓ partial / final
    UI (主线程 via after())
```

## 两种工作模式
- **麦克风模式（_vad_mic）**：RMS 能量检测说话开始/停顿，每 0.8s 发 partial 中间结果，停顿 1.2s 发 final 定稿
- **视频模式（_vad_video）**：固定时间窗口切片（Apple Silicon 2s，Intel 3s），无 VAD，适合连续音频

## 后端自动选择
```python
IS_APPLE_SILICON = (platform.processor() == "" and platform.machine() == "arm64")
BACKEND = "mlx" if IS_APPLE_SILICON and mlx_whisper installed else "faster_whisper"
```

## 关键参数
| 参数 | 值 | 说明 |
|------|----|------|
| SAMPLE_RATE | 16000 | Whisper 要求的采样率 |
| SILENCE_RMS | 200 | 低于此值视为静音，噪音大时调高 |
| SILENCE_TRIGGER | 40 | 连续静音块数触发断句（~1.2s）|
| PARTIAL_INTERVAL | 0.8s | 中间结果刷新频率 |
| VIDEO_WINDOW_SEC | 2s(M) / 3s(Intel) | 视频模式切片长度 |
| CPU_THREADS | 6 | Intel 推理线程数 |

## 模型对应
### MLX（Apple Silicon）
- `mlx-community/whisper-{name}-mlx`
- 推荐：`large-v3-turbo`（速度+精度最佳）

### faster-whisper（Intel）
- `Systran/faster-whisper-{name}`
- 推荐：`medium.en`（英语最优，int8 量化）

## 开发约定
- 所有 UI 更新必须通过 `self.after(0, ...)` 回到主线程
- TranscriptionWorker 只保留最新的 partial，旧的直接丢弃防止积压
- 关闭窗口 = `withdraw()`（隐藏），「退出」按钮才真正 `destroy()`
- 模型下载使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像

## 已知问题 / TODO
- [ ] ScreenCaptureKit 在 macOS 12 以下不可用，需降级到 BlackHole 方案
- [ ] Intel Mac 视频模式延迟约 1-2 秒，受 CPU 算力限制
- [ ] 视频模式下音量调节失效（多输出设备限制），建议用应用内音量
- [ ] MLX 模型首次下载也需要 hf-mirror 或 VPN
- [ ] make_app.sh 生成的图标为纯色圆形，待替换为精美图标
