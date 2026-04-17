#!/usr/bin/env python3
"""
voiceScribe — 流式实时字幕 GUI
自动检测硬件：
  Apple Silicon → mlx-whisper（Neural Engine，极速）
  Intel         → faster-whisper（CPU int8，已优化）
"""

import os
import subprocess
import platform
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import queue
import threading
import time
import numpy as np
import sounddevice as sd
import customtkinter as ctk
import tkinter as tk

try:
    from system_audio import SystemAudioCapture, SCK_AVAILABLE
except Exception:
    SCK_AVAILABLE = False
    SystemAudioCapture = None

# ── 硬件检测 ─────────────────────────────────────────────────────────────────
IS_APPLE_SILICON = (platform.processor() == "" and platform.machine() == "arm64")

# ── 后端导入 ─────────────────────────────────────────────────────────────────
if IS_APPLE_SILICON:
    try:
        import mlx_whisper as _mlx_whisper
        BACKEND = "mlx"
    except ImportError:
        BACKEND = "faster_whisper"
else:
    BACKEND = "faster_whisper"

if BACKEND == "faster_whisper":
    from faster_whisper import WhisperModel

# ── 配置 ────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16_000
CHANNELS          = 1
CHUNK_DURATION    = 0.03
CHUNK_SIZE        = int(SAMPLE_RATE * CHUNK_DURATION)

SILENCE_RMS       = 200
SPEECH_TRIGGER    = 3
SILENCE_TRIGGER   = 40
MIN_SPEECH_CHUNKS = 8
PARTIAL_INTERVAL  = 0.8
MAX_SEGMENT_SEC   = 20
CPU_THREADS       = 6

# 视频模式窗口：M 系列用 2 秒，Intel 用 3 秒
VIDEO_WINDOW_SEC  = 2 if IS_APPLE_SILICON else 3

# ── 模型列表 ─────────────────────────────────────────────────────────────────
# MLX 模型映射（HuggingFace repo）
MLX_MODELS = {
    "tiny.en":         "mlx-community/whisper-tiny.en-mlx",
    "base.en":         "mlx-community/whisper-base.en-mlx",
    "small.en":        "mlx-community/whisper-small.en-mlx",
    "medium.en":       "mlx-community/whisper-medium.en-mlx",
    "large-v3":        "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo-mlx",
}
MLX_DEFAULT    = "large-v3-turbo"
MLX_OPTIONS    = ["small.en", "medium.en", "large-v3-turbo", "large-v3"]

FW_DEFAULT     = "medium.en"
FW_OPTIONS     = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]

MODEL_SIZES_MB = {
    "tiny.en": 80,   "tiny": 80,
    "base.en": 150,  "base": 150,
    "small.en": 500, "small": 500,
    "medium.en": 900,"medium": 900,
    "large-v3": 1700,
    "large-v3-turbo": 900,
}
# ────────────────────────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLOR_FINAL   = "#e8e8ff"
COLOR_PARTIAL = "#5599ee"


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _process_rss_mb() -> int:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        )
        return int(out.strip()) // 1024
    except Exception:
        return 0


def _hf_cache_size_mb(model_name: str) -> int:
    import pathlib
    for prefix in ["models--Systran--faster-whisper-",
                   "models--mlx-community--whisper-"]:
        suffix = model_name if "turbo" not in model_name else f"{model_name}"
        repo   = f"{prefix}{suffix.replace('.', '-') if 'mlx' in prefix else suffix}-mlx" \
                 if "mlx" in prefix else f"{prefix}{model_name}"
        cache  = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / repo
        if cache.exists():
            return sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) // (1024 * 1024)
    return 0


def _is_model_cached(model_name: str) -> bool:
    return _hf_cache_size_mb(model_name) >= MODEL_SIZES_MB.get(model_name, 300) * 0.9


def list_input_devices() -> list[tuple[int, str]]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append((i, d["name"]))
    return devices


# ── 统一转录后端 ──────────────────────────────────────────────────────────────
class WhisperBackend:
    """统一封装 mlx-whisper 和 faster-whisper，对外接口相同"""

    def __init__(self, model_name: str):
        self._name    = model_name
        self._backend = BACKEND
        self._model   = None

        if self._backend == "faster_whisper":
            self._model = WhisperModel(
                model_name, device="cpu",
                compute_type="int8",
                cpu_threads=CPU_THREADS,
            )
        # mlx 懒加载，第一次 transcribe 时自动下载/缓存

    def transcribe(self, audio_f32: np.ndarray, lang: str,
                   video_mode: bool = False) -> str:
        language = None if lang == "auto" else lang

        if self._backend == "mlx":
            repo   = MLX_MODELS.get(self._name, f"mlx-community/whisper-{self._name}-mlx")
            result = _mlx_whisper.transcribe(
                audio_f32,
                path_or_hf_repo=repo,
                language=language,
                verbose=False,
            )
            return result["text"].strip()
        else:
            kwargs = dict(
                language=lang if lang != "auto" else None,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
            )
            if video_mode:
                kwargs["vad_filter"] = False
            else:
                kwargs["vad_filter"] = True
                kwargs["vad_parameters"] = {"min_silence_duration_ms": 200}
            segments, _ = self._model.transcribe(audio_f32, **kwargs)
            return "".join(s.text for s in segments).strip()


# ── 转录工作线程 ──────────────────────────────────────────────────────────────
class TranscriptionWorker:
    def __init__(self, backend: WhisperBackend, lang: str,
                 on_partial, on_final, on_discard, video_mode: bool = False):
        self._backend    = backend
        self._lang       = lang
        self._video_mode = video_mode
        self._on_partial = on_partial
        self._on_final   = on_final
        self._on_discard = on_discard

        self._lock        = threading.Lock()
        self._partial_buf = None
        self._final_buf   = None
        self._event       = threading.Event()
        self._stop        = False

        threading.Thread(target=self._run, daemon=True).start()

    def submit_partial(self, audio: np.ndarray):
        with self._lock:
            self._partial_buf = audio.copy()
        self._event.set()

    def submit_final(self, audio: np.ndarray):
        with self._lock:
            self._final_buf   = audio.copy()
            self._partial_buf = None
        self._event.set()

    def submit_discard(self):
        with self._lock:
            self._partial_buf = None
            self._final_buf   = None
        self._on_discard()

    def shutdown(self):
        self._stop = True
        self._event.set()

    def _run(self):
        while not self._stop:
            self._event.wait()
            self._event.clear()
            with self._lock:
                final   = self._final_buf
                partial = self._partial_buf
                if final is not None:
                    self._final_buf = None
                elif partial is not None:
                    self._partial_buf = None
            if final is not None:
                text = self._backend.transcribe(final, self._lang, self._video_mode)
                if text:
                    self._on_final(text)
                else:
                    self._on_discard()
            elif partial is not None:
                text = self._backend.transcribe(partial, self._lang, False)
                self._on_partial(text)


# ── 主应用 ────────────────────────────────────────────────────────────────────
class VoiceScribeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("voiceScribe")
        self.geometry("860x610")
        self.minsize(620, 460)

        self._backend: WhisperBackend | None = None
        self._loaded_key  = ""
        self._worker: TranscriptionWorker | None = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._running     = False
        self._has_partial = False
        self._count       = 0
        self._devices     = list_input_devices()
        self._device_map  = {name: idx for idx, name in self._devices}

        self._build_ui()

    # ── 界面 ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 顶部工具栏
        bar1 = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#141428")
        bar1.grid(row=0, column=0, sticky="ew")
        bar1.grid_columnconfigure(1, weight=1)

        # 品牌 + 芯片标识
        chip_tag  = "M系列 ⚡" if IS_APPLE_SILICON else "Intel"
        chip_color = "#60cc60" if IS_APPLE_SILICON else "#aaaaaa"
        ctk.CTkLabel(
            bar1, text="🎙 voiceScribe",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#c8c8ff",
        ).grid(row=0, column=0, padx=(16, 4), pady=10)
        ctk.CTkLabel(
            bar1, text=chip_tag,
            font=ctk.CTkFont(size=11),
            text_color=chip_color,
        ).grid(row=0, column=1, padx=(0, 10), sticky="w")

        # 模型选择（按芯片给不同选项）
        model_opts    = MLX_OPTIONS if IS_APPLE_SILICON else FW_OPTIONS
        default_model = MLX_DEFAULT if IS_APPLE_SILICON else FW_DEFAULT
        self._model_var = ctk.StringVar(value=default_model)
        ctk.CTkOptionMenu(
            bar1, values=model_opts,
            variable=self._model_var, width=145,
            fg_color="#1e1e3e", button_color="#0f3460", button_hover_color="#1a4a7a",
        ).grid(row=0, column=2, padx=6)

        self._lang_var = ctk.StringVar(value="en")
        ctk.CTkOptionMenu(
            bar1, values=["en", "zh", "ja", "ko", "auto"],
            variable=self._lang_var, width=80,
            fg_color="#1e1e3e", button_color="#0f3460", button_hover_color="#1a4a7a",
        ).grid(row=0, column=3, padx=6)

        self._video_mode = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            bar1, text="视频模式",
            variable=self._video_mode,
            font=ctk.CTkFont(size=12), width=100,
            button_color="#0f3460", progress_color="#1a5090",
        ).grid(row=0, column=4, padx=10)

        ctk.CTkButton(
            bar1, text="复制", width=60,
            fg_color="#2a2a44", hover_color="#3a3a5e",
            command=self._copy_all,
        ).grid(row=0, column=5, padx=6)

        ctk.CTkButton(
            bar1, text="清空", width=60,
            fg_color="#2a2a44", hover_color="#3a3a5e",
            command=self._clear,
        ).grid(row=0, column=6, padx=6)

        ctk.CTkButton(
            bar1, text="退出", width=60,
            fg_color="#3e1010", hover_color="#5e1818",
            command=self.quit_app,
        ).grid(row=0, column=7, padx=(0, 14))

        # 设备选择栏
        bar2 = ctk.CTkFrame(self, height=40, corner_radius=0, fg_color="#0f0f20")
        bar2.grid(row=1, column=0, sticky="ew")
        bar2.grid_columnconfigure(1, weight=1)

        # 来源：麦克风 or 系统音频
        self._source_var = ctk.StringVar(value="麦克风")
        src_options = ["麦克风", "系统音频 (ScreenCaptureKit)"] if SCK_AVAILABLE else ["麦克风"]
        ctk.CTkOptionMenu(
            bar2, values=src_options,
            variable=self._source_var, width=220,
            fg_color="#1a1a30", button_color="#0f3460", button_hover_color="#1a4a7a",
            command=self._on_source_change,
        ).grid(row=0, column=0, padx=(16, 8), pady=8)

        ctk.CTkLabel(
            bar2, text="设备",
            font=ctk.CTkFont(size=12), text_color="#6666aa",
        ).grid(row=0, column=1, padx=(0, 4), pady=8)

        names = [n for _, n in self._devices]
        self._device_var = ctk.StringVar(value=self._pick_default_device(names))
        self._device_menu = ctk.CTkOptionMenu(
            bar2, values=names or ["（无设备）"],
            variable=self._device_var, width=360,
            fg_color="#1a1a30", button_color="#0f3460", button_hover_color="#1a4a7a",
            dynamic_resizing=False,
        )
        self._device_menu.grid(row=0, column=2, padx=6, sticky="w")

        ctk.CTkButton(
            bar2, text="↻", width=32, height=26,
            fg_color="#1e1e3e", hover_color="#2e2e5e",
            command=self._refresh_devices,
            font=ctk.CTkFont(size=14),
        ).grid(row=0, column=3, padx=(0, 14))

        # 文本区
        tf = ctk.CTkFrame(self, fg_color="#080814", corner_radius=0)
        tf.grid(row=2, column=0, sticky="nsew")
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)

        self._text = tk.Text(
            tf, bg="#080814", fg=COLOR_FINAL,
            font=("Helvetica Neue", 15), wrap=tk.WORD,
            state=tk.DISABLED, relief=tk.FLAT, bd=0,
            padx=18, pady=12, cursor="arrow",
        )
        self._text.tag_configure("final",   foreground=COLOR_FINAL)
        self._text.tag_configure("partial", foreground=COLOR_PARTIAL)
        self._text.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(tf, command=self._text.yview, bg="#141428")
        sb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=sb.set)

        # 状态栏
        sbar = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color="#0e0e22")
        sbar.grid(row=3, column=0, sticky="ew")
        sbar.grid_columnconfigure(1, weight=1)

        self._toggle_btn = ctk.CTkButton(
            sbar, text="▶  开始", width=110, height=34,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0f3460", hover_color="#1a5090",
            command=self._toggle,
        )
        self._toggle_btn.grid(row=0, column=0, padx=16, pady=11)

        mid = ctk.CTkFrame(sbar, fg_color="transparent")
        mid.grid(row=0, column=1, sticky="ew", padx=8)
        mid.grid_columnconfigure(0, weight=1)

        self._status_lbl = ctk.CTkLabel(
            mid, text="就绪", font=ctk.CTkFont(size=13), text_color="#6666aa",
        )
        self._status_lbl.grid(row=0, column=0, sticky="w")

        self._rms_lbl = ctk.CTkLabel(
            mid, text="", font=ctk.CTkFont(size=11), text_color="#444466",
        )
        self._rms_lbl.grid(row=0, column=1, padx=(12, 0), sticky="w")

        self._level_bar = ctk.CTkProgressBar(
            mid, height=5, fg_color="#1a1a36", progress_color="#3a70c0",
        )
        self._level_bar.set(0)
        self._level_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        self._count_lbl = ctk.CTkLabel(
            sbar, text="0 句", font=ctk.CTkFont(size=12), text_color="#444466",
        )
        self._count_lbl.grid(row=0, column=2, padx=16)

    # ── 工具 ─────────────────────────────────────────────────────────────────

    def _pick_default_device(self, names):
        for n in names:
            if "macbook" in n.lower() or "built-in" in n.lower() or "mac mini" in n.lower():
                return n
        return names[0] if names else ""

    def _on_source_change(self, val):
        is_sys = "ScreenCaptureKit" in val
        state  = "disabled" if is_sys else "normal"
        self._device_menu.configure(state=state)

    def _refresh_devices(self):
        sd._terminate(); sd._initialize()
        self._devices    = list_input_devices()
        self._device_map = {name: idx for idx, name in self._devices}
        names = [n for _, n in self._devices]
        self._device_menu.configure(values=names)

    def _selected_device_id(self):
        return self._device_map.get(self._device_var.get())

    # ── 控制 ─────────────────────────────────────────────────────────────────

    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self._running = True
        self._toggle_btn.configure(text="⏹  停止", fg_color="#5e1010", hover_color="#7e1818")
        self._set_status("加载模型…", "#c0903a")
        threading.Thread(target=self._init_and_run, daemon=True).start()

    def _stop(self):
        self._running = False
        if self._worker:
            self._worker.shutdown()
            self._worker = None
        self._toggle_btn.configure(text="▶  开始", fg_color="#0f3460", hover_color="#1a5090")
        self._set_status("已停止", "#6666aa")
        self._level_bar.set(0)

    def _init_and_run(self):
        name = self._model_var.get()
        key  = f"{BACKEND}:{name}:{self._lang_var.get()}"

        if self._backend is None or self._loaded_key != key:
            cached   = _is_model_cached(name)
            total_mb = MODEL_SIZES_MB.get(name, 500)
            self._append_log(f"{'加载' if cached else '下载'}模型 {name}"
                             f"{'（已缓存）' if cached else f'（约 {total_mb} MB）'}…\n")
            self._append_log(f"后端：{'MLX / Apple Silicon ⚡' if BACKEND == 'mlx' else 'faster-whisper / Intel CPU'}\n")

            result, err = [None], [None]

            def do_load():
                try:
                    result[0] = WhisperBackend(name)
                except Exception as e:
                    err[0] = e

            t = threading.Thread(target=do_load, daemon=True)
            t.start()

            base_mb = _process_rss_mb()
            while t.is_alive():
                if cached:
                    loaded = max(_process_rss_mb() - base_mb, 0)
                    pct    = min(int(loaded / total_mb * 100), 99)
                    self._set_status(f"加载中 {loaded}/{total_mb} MB  {pct}%", "#c0903a")
                    self.after(0, self._level_bar.set, min(loaded / total_mb, 0.99))
                else:
                    mb  = _hf_cache_size_mb(name)
                    pct = min(int(mb / total_mb * 100), 99)
                    if mb >= total_mb * 0.95:
                        self._set_status("初始化模型，请稍候…", "#c0903a")
                        self.after(0, self._level_bar.set, 0.99)
                    else:
                        self._set_status(f"下载中 {mb}/{total_mb} MB  {pct}%", "#c0903a")
                        self.after(0, self._level_bar.set, min(mb / total_mb, 0.98))
                time.sleep(0.8)

            t.join()
            if err[0]:
                self._append_log(f"加载失败：{err[0]}\n")
                self._stop()
                return

            self._backend     = result[0]
            self._loaded_key  = key
            self._append_log(f"模型 {name} 就绪 ✓\n")
            self.after(0, self._level_bar.set, 0)

        self._worker = TranscriptionWorker(
            backend    = self._backend,
            lang       = self._lang_var.get(),
            on_partial = lambda t: self.after(0, self._show_partial, t),
            on_final   = lambda t: self.after(0, self._show_final, t),
            on_discard = lambda:   self.after(0, self._discard_partial),
            video_mode = self._video_mode.get(),
        )

        self._audio_queue = queue.Queue()
        self._set_status("监听中…", "#40a040")
        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._vad_loop,    daemon=True).start()

    # ── 录音 ─────────────────────────────────────────────────────────────────

    def _listen_loop(self):
        use_sck = "ScreenCaptureKit" in self._source_var.get()

        if use_sck and SCK_AVAILABLE:
            cap = SystemAudioCapture(self._audio_queue)
            try:
                cap.start()
                self._set_status("系统音频监听中…", "#40a040")
                while self._running:
                    time.sleep(0.1)
            finally:
                cap.stop()
        else:
            def callback(indata, frames, time_info, status):
                if self._running:
                    self._audio_queue.put(indata[:, 0].copy())

            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype="int16", blocksize=CHUNK_SIZE,
                device=self._selected_device_id(),
                callback=callback,
            ):
                while self._running:
                    time.sleep(0.05)

    # ── VAD 状态机 ────────────────────────────────────────────────────────────

    def _vad_loop(self):
        if self._video_mode.get():
            self._vad_video()
        else:
            self._vad_mic()

    def _vad_video(self):
        """视频模式：固定窗口，无需静音检测"""
        buf: list[np.ndarray] = []
        win_start = time.time()

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            level = rms(chunk)
            self.after(0, self._level_bar.set, min(level / 2500.0, 1.0))
            self.after(0, self._rms_lbl.configure, {"text": f"RMS {int(level)}"})
            buf.append(chunk)

            if time.time() - win_start >= VIDEO_WINDOW_SEC:
                if buf and self._worker:
                    self._worker.submit_final(np.concatenate(buf))
                buf.clear()
                win_start = time.time()

    def _vad_mic(self):
        """麦克风模式：VAD 检测停顿，流式纠正"""
        speech_buf: list[np.ndarray] = []
        silent_count = speech_count = 0
        in_speech     = False
        last_partial  = 0.0
        segment_start = 0.0

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            level = rms(chunk)
            self.after(0, self._level_bar.set, min(level / 2500.0, 1.0))
            self.after(0, self._rms_lbl.configure, {"text": f"RMS {int(level)}"})

            if not in_speech:
                if level > SILENCE_RMS:
                    speech_count += 1
                    speech_buf.append(chunk)
                    if speech_count >= SPEECH_TRIGGER:
                        in_speech     = True
                        silent_count  = 0
                        last_partial  = time.time()
                        segment_start = time.time()
                        self._set_status("录音中…", "#c04040")
                else:
                    speech_count = 0
                    speech_buf.clear()
            else:
                speech_buf.append(chunk)
                now = time.time()

                if now - segment_start >= MAX_SEGMENT_SEC:
                    if self._worker:
                        self._worker.submit_final(np.concatenate(speech_buf))
                    speech_buf.clear()
                    speech_count = silent_count = 0
                    in_speech = False
                    segment_start = last_partial = now
                    if self._running:
                        self._set_status("监听中…", "#40a040")
                    continue

                if level < SILENCE_RMS:
                    silent_count += 1
                    if now - last_partial >= PARTIAL_INTERVAL and self._worker:
                        self._worker.submit_partial(np.concatenate(speech_buf))
                        last_partial = now
                    if silent_count >= SILENCE_TRIGGER:
                        if len(speech_buf) >= MIN_SPEECH_CHUNKS and self._worker:
                            self._worker.submit_final(np.concatenate(speech_buf))
                        elif self._worker:
                            self._worker.submit_discard()
                        speech_buf.clear()
                        speech_count = silent_count = 0
                        in_speech = False
                        if self._running:
                            self._set_status("监听中…", "#40a040")
                else:
                    silent_count = 0
                    if now - last_partial >= PARTIAL_INTERVAL and self._worker:
                        self._worker.submit_partial(np.concatenate(speech_buf))
                        last_partial = now

    # ── UI 更新 ──────────────────────────────────────────────────────────────

    def _show_partial(self, text: str):
        self._text.configure(state=tk.NORMAL)
        if self._has_partial:
            self._text.delete("end-1l linestart", "end")
        self._text.insert(tk.END, text, "partial")
        self._has_partial = True
        self._text.configure(state=tk.DISABLED)
        self._text.see(tk.END)

    def _show_final(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self._count += 1
        self._count_lbl.configure(text=f"{self._count} 句")
        self._text.configure(state=tk.NORMAL)
        if self._has_partial:
            self._text.delete("end-1l linestart", "end")
        self._text.insert(tk.END, f"[{ts}]  {text}\n", "final")
        self._has_partial = False
        self._text.configure(state=tk.DISABLED)
        self._text.see(tk.END)

    def _discard_partial(self):
        if not self._has_partial:
            return
        self._text.configure(state=tk.NORMAL)
        self._text.delete("end-1l linestart", "end")
        self._has_partial = False
        self._text.configure(state=tk.DISABLED)

    def _append_log(self, msg: str):
        self.after(0, lambda: (
            self._text.configure(state=tk.NORMAL),
            self._text.insert(tk.END, msg, "final"),
            self._text.configure(state=tk.DISABLED),
            self._text.see(tk.END),
        ))

    def _copy_all(self):
        text = self._text.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

    def _clear(self):
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
        self._has_partial = False
        self._count = 0
        self._count_lbl.configure(text="0 句")

    def _set_status(self, text: str, color: str = "#6666aa"):
        self.after(0, lambda: self._status_lbl.configure(text=text, text_color=color))

    def on_close(self):
        if self._running:
            self._stop()
        self.withdraw()

    def quit_app(self):
        self._running = False
        if self._worker:
            self._worker.shutdown()
        self.destroy()


def main():
    app = VoiceScribeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.createcommand("tk::mac::ReopenApplication", app.deiconify)
    app.mainloop()


if __name__ == "__main__":
    main()
