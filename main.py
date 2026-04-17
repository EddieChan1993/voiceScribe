#!/usr/bin/env python3
"""
voiceScribe — 实时语音转文字 (CLI)
按 Ctrl+C 退出
"""

import sys
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ── 配置 ────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16_000
CHANNELS          = 1
CHUNK_DURATION    = 0.03
CHUNK_SIZE        = int(SAMPLE_RATE * CHUNK_DURATION)

SILENCE_RMS       = 200
SPEECH_TRIGGER    = 3
SILENCE_TRIGGER   = 40
MIN_SPEECH_CHUNKS = 10

MODEL_NAME        = "medium.en"
LANGUAGE          = "en"
# ────────────────────────────────────────────────────────────────────────────


def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def load_model() -> WhisperModel:
    print(f"[voiceScribe] 加载 faster-whisper {MODEL_NAME} 模型...", flush=True)
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    print("[voiceScribe] 模型就绪，开始监听麦克风 🎙\n", flush=True)
    return model


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    audio_f32 = audio.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_f32,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    return "".join(s.text for s in segments).strip()


def listen(audio_queue: queue.Queue):
    def callback(indata, frames, time_info, status):
        if status:
            print(f"[警告] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    ):
        threading.Event().wait()


def vad_loop(audio_queue: queue.Queue, model: WhisperModel):
    speech_buf: list[np.ndarray] = []
    silent_count = speech_count = 0
    in_speech = False

    while True:
        chunk = audio_queue.get()
        level = rms(chunk)

        if not in_speech:
            if level > SILENCE_RMS:
                speech_count += 1
                speech_buf.append(chunk)
                if speech_count >= SPEECH_TRIGGER:
                    in_speech, silent_count = True, 0
                    print("[...说话中]", end="\r", flush=True)
            else:
                speech_count = 0
                speech_buf.clear()
        else:
            speech_buf.append(chunk)
            if level < SILENCE_RMS:
                silent_count += 1
                if silent_count >= SILENCE_TRIGGER:
                    if len(speech_buf) >= MIN_SPEECH_CHUNKS:
                        audio = np.concatenate(speech_buf)
                        print(" " * 20, end="\r")
                        text = transcribe(model, audio)
                        if text:
                            print(f"▶ {text}")
                    speech_buf.clear()
                    speech_count = silent_count = 0
                    in_speech = False
            else:
                silent_count = 0


def main():
    model = load_model()
    audio_queue: queue.Queue = queue.Queue()
    threading.Thread(target=listen, args=(audio_queue,), daemon=True).start()
    try:
        vad_loop(audio_queue, model)
    except KeyboardInterrupt:
        print("\n[voiceScribe] 已退出。")


if __name__ == "__main__":
    main()
