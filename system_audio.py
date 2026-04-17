#!/usr/bin/env python3
"""
macOS 原生系统音频捕获（ScreenCaptureKit, macOS 12.3+）
无需 BlackHole，首次使用时系统会弹出「屏幕录制」权限请求。
"""

import queue
import threading
import numpy as np
import objc
from Foundation import NSObject, NSRunLoop, NSDate

try:
    import ScreenCaptureKit as SCK
    import CoreMedia
    SCK_AVAILABLE = True
except ImportError:
    SCK_AVAILABLE = False


TARGET_SR   = 16_000      # 目标采样率
CAPTURE_SR  = 48_000      # SCK 默认输出采样率
CHUNK_SEC   = 0.03        # 每块 30ms
CHUNK_FRAMES = int(TARGET_SR * CHUNK_SEC)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    n_out = int(len(samples) * dst_rate / src_rate)
    x_old = np.arange(len(samples))
    x_new = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _sampleBuffer_to_float32(sampleBuffer) -> np.ndarray | None:
    """从 CMSampleBuffer 提取 float32 PCM 数据"""
    try:
        import ctypes
        blockBuf = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
        if blockBuf is None:
            return None
        length = CoreMedia.CMBlockBufferGetDataLength(blockBuf)
        if length == 0:
            return None
        raw = (ctypes.c_byte * length)()
        status = CoreMedia.CMBlockBufferCopyDataBytes(blockBuf, 0, length, raw)
        if status != 0:
            return None
        return np.frombuffer(bytes(raw), dtype=np.float32).copy()
    except Exception:
        return None


# ── Delegate（接收 SCStream 回调）────────────────────────────────────────────
SCStreamOutput = objc.protocolNamed("SCStreamOutput") if SCK_AVAILABLE else None


class _AudioDelegate(NSObject if SCK_AVAILABLE else object,
                     protocols=[SCStreamOutput] if SCK_AVAILABLE else []):

    @objc.python_method
    def _init_delegate(self, audio_queue: queue.Queue,
                       src_rate: int, channel_count: int):
        self._queue        = audio_queue
        self._src_rate     = src_rate
        self._channel_count = channel_count
        self._leftover     = np.array([], dtype=np.int16)
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, type_):
        if int(type_) != 1:   # SCStreamOutputTypeAudio == 1
            return
        samples = _sampleBuffer_to_float32(sampleBuffer)
        if samples is None or len(samples) == 0:
            return

        # 多声道 → 单声道
        if self._channel_count > 1:
            samples = samples.reshape(-1, self._channel_count).mean(axis=1)

        # 重采样到 16kHz
        resampled = _resample(samples, self._src_rate, TARGET_SR)

        # float32 → int16
        int16 = (resampled * 32768).clip(-32768, 32767).astype(np.int16)

        # 拼接并切成固定大小块
        buf = np.concatenate([self._leftover, int16])
        while len(buf) >= CHUNK_FRAMES:
            self._queue.put(buf[:CHUNK_FRAMES].copy())
            buf = buf[CHUNK_FRAMES:]
        self._leftover = buf


# ── 主捕获类 ─────────────────────────────────────────────────────────────────
class SystemAudioCapture:
    """
    用法:
        cap = SystemAudioCapture(audio_queue)
        cap.start()   # 异步，弹出权限请求
        ...
        cap.stop()
    """

    def __init__(self, audio_queue: queue.Queue):
        if not SCK_AVAILABLE:
            raise RuntimeError("ScreenCaptureKit 不可用，请升级到 macOS 12.3+")
        self._queue   = audio_queue
        self._stream  = None
        self._delegate = None
        self._running  = False
        self._started  = threading.Event()
        self._error    = None

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        # 等待流启动（最多 5 秒）
        self._started.wait(timeout=5.0)
        if self._error:
            raise RuntimeError(f"启动系统音频捕获失败：{self._error}")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stopCaptureWithCompletionHandler_(lambda e: None)
            self._stream = None

    def _run_loop(self):
        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(
            self._on_content
        )
        loop = NSRunLoop.currentRunLoop()
        while self._running:
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))

    def _on_content(self, content, error):
        if error or not content:
            self._error = str(error) if error else "无法获取屏幕内容"
            self._started.set()
            return

        displays = content.displays()
        if not displays:
            self._error = "未找到显示器"
            self._started.set()
            return

        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setCapturesAudio_(True)
        cfg.setExcludesCurrentProcessAudio_(True)   # 不录自身进程的声音
        cfg.setSampleRate_(CAPTURE_SR)
        cfg.setChannelCount_(2)

        # 捕获整个屏幕的音频
        filt = SCK.SCContentFilter.alloc() \
            .initWithDisplay_excludingApplications_exceptingWindows_(
                displays[0], [], []
            )

        self._delegate = _AudioDelegate.alloc().init()
        self._delegate._init_delegate(self._queue, CAPTURE_SR, 2)

        self._stream = SCK.SCStream.alloc() \
            .initWithFilter_configuration_delegate_(filt, cfg, None)

        err_ptr = objc.nil
        self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._delegate,
            SCK.SCStreamOutputTypeAudio,
            None,
            err_ptr,
        )

        self._stream.startCaptureWithCompletionHandler_(self._on_start)

    def _on_start(self, error):
        if error:
            self._error = str(error)
        self._started.set()
