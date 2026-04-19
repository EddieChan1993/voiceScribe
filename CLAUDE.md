# voiceScribe 项目上下文

## 项目简介
macOS 本地实时语音转文字工具，麦克风输入，自动适配 Apple Silicon / Intel。

## 技术栈
- **语言**：Python 3.10+
- **GUI**：customtkinter + tkinter（深色主题）
- **推理引擎**：
  - Apple Silicon → `mlx-whisper`（MLX / Neural Engine）
  - Intel → `faster-whisper`（CTranslate2 / CPU int8）
- **音频采集**：`sounddevice`（麦克风）
- **VAD**：能量 RMS 检测 + 最大段时长强制截断

## 目录结构
```
app.py            主程序，GUI + 控制逻辑
system_audio.py   ScreenCaptureKit 系统音频捕获（内录，暂时关闭）
main.py           CLI 版本（无 GUI）
setup.sh          安装脚本，自动检测芯片
make_app.sh       生成 ~/Applications/voiceScribe.app
requirements.txt  Python 依赖
```

## 核心架构
```
sounddevice（麦克风）
        ↓ int16 chunks (16kHz)
    LiveTranscriber._q (Queue)
        ↓
    _vad_loop（VAD 状态机）
        ↓ numpy arrays
    _infer_loop（partial 定时推理）/ _do_final（定稿推理）
        ↓ on_partial / on_final 回调
    UI（主线程 via after()）
```

## VAD 逻辑
- **RMS 检测**：连续 `SPEECH_TRIGGER=3` 块超过 `SILENCE_RMS=80` → 进入说话状态
- **定稿**：连续 `SILENCE_SEC=1.0s` 静音 → 提交 final
- **强制截断**：单句超过 `MAX_WIN_SEC=8s` → 强制 final，防止 "掐头"
- **Partial 刷新**：每 `REFRESH_SEC=0.35s(MLX)/1.0s(Intel)` 推理当前积累音频

## 幻觉过滤
```python
def _is_hallucination(text):
    # 任意单词占比超过 55% → 丢弃（Whisper 在低质量音频下产生重复词）
```

## 后端自动选择
```python
IS_APPLE_SILICON = platform.machine() == "arm64"
# 检测 ~/.cache/huggingface/hub 是否有 mlx 模型缓存
BACKEND = "mlx" if IS_APPLE_SILICON and _has_mlx else "faster_whisper"
```

## 关键参数
| 参数 | 值 | 说明 |
|------|----|------|
| SAMPLE_RATE | 16000 | Whisper 要求的采样率 |
| SILENCE_RMS | 80 | 低于此值视为静音 |
| SILENCE_SEC | 1.0 | 停顿触发定稿（秒）|
| MAX_WIN_SEC | 8.0 | 单句最大时长 |
| REFRESH_SEC | 0.35s/1.0s | partial 刷新频率（MLX/Intel）|
| CPU_THREADS | 6 | Intel 推理线程数 |

## UI 结构
- **bar1**（顶栏）：标题、后端标签、模型选择、语言选择、复制/清空/退出
- **bar2**（设备栏）：设备下拉 + 刷新（来源选择器已注释，待内录功能启用）
- **tf**（文字区）：
  - `_text`（上方，可滚动）：定稿文字，白色
  - `_partial_box`（下方固定 3 行）：实时识别中的文字，蓝色
- **sbar**（底栏）：开始/停止按钮、状态、音量条、句数

## 重要约定
- 所有 UI 更新必须通过 `self.after(0, ...)` 回到主线程
- `_initial_clear_done`：整个 App 生命周期只自动清屏一次（首次检测到声音）；之后只有用户点「清空」才清
- 停止 → 再开始：文字面板不自动清空，继续追加
- 关闭窗口 = `withdraw()`（隐藏），退出按钮才真正 `destroy()`
- 模型下载使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像

## 系统音频内录（暂时关闭）
- 代码在 `system_audio.py`（`SystemAudioCapture` 类）和 `app.py` 中已注释的 `_run_sck` 方法
- 启用步骤：
  1. 取消注释 `system_audio` import
  2. 取消注释 bar2 来源选择器 UI
  3. 取消注释 `_on_source_change` 方法
  4. 取消注释 `_init_and_run` 中的路由逻辑
  5. 取消注释 `_run_sck` 方法
  6. 在「系统设置 → 隐私与安全性 → 屏幕与系统录音」授权，重启应用

## 已知问题 / TODO
- [ ] 系统音频内录待权限调试后启用
- [ ] Intel Mac 视频模式延迟约 1-2 秒，受 CPU 算力限制
- [ ] MLX 模型首次下载需要 hf-mirror 或 VPN
- [ ] make_app.sh 生成的图标为纯色圆形，待替换为精美图标
