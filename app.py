#!/usr/bin/env python3
"""
voiceScribe — 实时字幕
麦克风：说话时实时刷新当前句，停顿后换行
后端优先级：mlx-whisper (Neural Engine) > faster-whisper (CPU)
"""

import os, platform, queue, threading, time, pathlib
import numpy as np
import sounddevice as sd
import customtkinter as ctk
import tkinter as tk

# ── 系统音频（内录）功能暂时关闭，代码保留供后续启用 ──────────────────────────
# try:
#     from system_audio import SystemAudioCapture, SCK_AVAILABLE
# except Exception:
#     SCK_AVAILABLE = False
#     SystemAudioCapture = None
SCK_AVAILABLE = False
SystemAudioCapture = None


# ── 后端检测 ──────────────────────────────────────────────────────────────────
IS_APPLE_SILICON = platform.machine() == "arm64"

_mlx_cache = pathlib.Path.home() / ".cache/huggingface/hub"
_has_mlx   = any(_mlx_cache.glob("models--mlx-community--whisper-*")) if _mlx_cache.exists() else False

if IS_APPLE_SILICON and _has_mlx:
    import mlx_whisper as _mlx
    BACKEND = "mlx"
else:
    from faster_whisper import WhisperModel as _FWModel
    BACKEND = "faster_whisper"

# ── 音频参数 ──────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CHUNK_SEC    = 0.03
CHUNK_SIZE   = int(SAMPLE_RATE * CHUNK_SEC)

# ── VAD 参数 ──────────────────────────────────────────────────────────────────
SILENCE_RMS     = 80
SPEECH_TRIGGER  = 3          # 连续 N 块有声才算说话开始
SILENCE_SEC     = 1.0        # 停顿多少秒提交定稿
MAX_WIN_SEC     = 8.0        # 最长单句

# ── 刷新率 ────────────────────────────────────────────────────────────────────
REFRESH_SEC = 0.35 if BACKEND == "mlx" else 1.0

# ── 模型列表 ──────────────────────────────────────────────────────────────────
if BACKEND == "mlx":
    MLX_MODELS = {
        "tiny.en":        "mlx-community/whisper-tiny.en-mlx",
        "base.en":        "mlx-community/whisper-base.en-mlx",
        "small.en":       "mlx-community/whisper-small.en-mlx",
        "small":          "mlx-community/whisper-small-mlx",
        "medium.en":      "mlx-community/whisper-medium.en-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-mlx",
        "large-v3":       "mlx-community/whisper-large-v3-mlx",
    }
    # 只列出已缓存的模型
    MODEL_OPTIONS = [
        k for k, repo in MLX_MODELS.items()
        if any(_mlx_cache.glob(f"models--mlx-community--whisper-{k.replace('.', '-')}-mlx*"))
        or any(_mlx_cache.glob(f"models--mlx-community--whisper-{k.replace('.', '-')}*"))
    ]
    if not MODEL_OPTIONS:
        MODEL_OPTIONS = ["small.en"]
    MODEL_DEFAULT = "small.en" if "small.en" in MODEL_OPTIONS else MODEL_OPTIONS[0]
else:
    MODEL_OPTIONS = ["tiny.en", "base.en", "small.en", "medium.en", "medium", "large-v3"]
    MODEL_DEFAULT = "medium.en"

CPU_THREADS = 6

# ── 外观 ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
COLOR_FINAL   = "#e8e8ff"
COLOR_PARTIAL = "#5599ee"


def rms(a): return float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))

def _is_hallucination(text: str) -> bool:
    """Whisper 有时在低质量音频下产生大量重复词，检测并过滤掉。"""
    words = text.split()
    if len(words) < 5:
        return False
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    max_repeat = max(counts.values())
    # 任意单词占比超过 55% → 幻觉
    return max_repeat / len(words) > 0.55

def list_input_devices():
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


# ── 转录后端 ──────────────────────────────────────────────────────────────────
class Transcriber:
    def __init__(self, model_name: str):
        self.backend = BACKEND
        self.model_name = model_name
        self._lock = threading.Lock()   # 防止并发推理
        if BACKEND == "faster_whisper":
            self._fw = _FWModel(model_name, device="cpu", compute_type="int8", cpu_threads=CPU_THREADS)

    def transcribe(self, audio_f32: np.ndarray, lang: str) -> str:
        language = None if lang == "auto" else lang
        with self._lock:
            if self.backend == "mlx":
                repo = MLX_MODELS.get(self.model_name, f"mlx-community/whisper-{self.model_name}-mlx")
                result = _mlx.transcribe(audio_f32, path_or_hf_repo=repo,
                                         language=language, verbose=False)
                return result["text"].strip()
            else:
                segs, _ = self._fw.transcribe(audio_f32, language=language,
                                              beam_size=1, best_of=1,
                                              condition_on_previous_text=False,
                                              vad_filter=False)
                return "".join(s.text for s in segs).strip()


# ── 流式 VAD + 推理线程 ───────────────────────────────────────────────────────
class LiveTranscriber:
    """
    采集线程维护 VAD 状态，推理线程每 REFRESH_SEC 刷新当前句。
    on_partial(text)  : 当前句更新（刷新显示，替换当前行）
    on_final(text)    : 当前句定稿（换行）
    on_level(rms_val) : 音量回调
    """
    def __init__(self, transcriber: Transcriber, lang: str,
                 on_partial, on_final, on_level):
        self._tr         = transcriber
        self._lang       = lang
        self._on_partial = on_partial
        self._on_final   = on_final
        self._on_level   = on_level

        self._q          = queue.Queue()
        self._lock       = threading.Lock()
        self._stop       = False

        # VAD 状态
        self._speech_buf    = []
        self._in_speech     = False
        self._speech_count  = 0
        self._silence_count = 0
        self._silence_trig  = int(SILENCE_SEC / CHUNK_SEC)
        self._max_chunks    = int(MAX_WIN_SEC / CHUNK_SEC)

        threading.Thread(target=self._vad_loop,   daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def push(self, chunk): self._q.put(chunk)
    def shutdown(self):    self._stop = True

    def _vad_loop(self):
        while not self._stop:
            try: chunk = self._q.get(timeout=0.1)
            except queue.Empty: continue

            level = rms(chunk)
            self._on_level(level)

            with self._lock:
                if not self._in_speech:
                    if level >= SILENCE_RMS:
                        self._speech_count += 1
                        self._speech_buf.append(chunk)
                        if self._speech_count >= SPEECH_TRIGGER:
                            self._in_speech    = True
                            self._silence_count = 0
                    else:
                        self._speech_count = 0
                        self._speech_buf.clear()
                else:
                    self._speech_buf.append(chunk)
                    if len(self._speech_buf) >= self._max_chunks:
                        # 达到最大句长：强制定稿，开新句，不丢音频
                        buf = list(self._speech_buf)
                        self._in_speech     = False
                        self._speech_buf    = []
                        self._speech_count  = 0
                        self._silence_count = 0
                        threading.Thread(
                            target=self._do_final, args=(buf,), daemon=True
                        ).start()
                        continue

                    if level < SILENCE_RMS:
                        self._silence_count += 1
                        if self._silence_count >= self._silence_trig:
                            # 定稿：提交最终音频
                            buf = list(self._speech_buf)
                            self._in_speech     = False
                            self._speech_buf    = []
                            self._speech_count  = 0
                            self._silence_count = 0
                            threading.Thread(
                                target=self._do_final, args=(buf,), daemon=True
                            ).start()
                    else:
                        self._silence_count = 0

    def _infer_loop(self):
        """每 REFRESH_SEC 推理当前积累的音频，刷新 partial"""
        while not self._stop:
            time.sleep(REFRESH_SEC)
            with self._lock:
                if not self._in_speech or len(self._speech_buf) < int(0.5 / CHUNK_SEC):
                    continue
                buf = list(self._speech_buf)
            self._do_partial(buf)

    def _do_partial(self, buf):
        # 若 final 正在推理则跳过，避免堆积
        if not self._tr._lock.acquire(blocking=False):
            return
        self._tr._lock.release()
        audio = np.concatenate(buf).astype(np.float32) / 32768.0
        try:
            text = self._tr.transcribe(audio, self._lang)
        except Exception:
            return
        if text and not _is_hallucination(text):
            self._on_partial(text)

    def _do_final(self, buf):
        if not buf:
            return
        audio = np.concatenate(buf).astype(np.float32) / 32768.0
        try:
            text = self._tr.transcribe(audio, self._lang)
        except Exception:
            return
        if text and _is_hallucination(text):
            text = ""
        self._on_final(text or "")


# ── 主应用 ────────────────────────────────────────────────────────────────────
class VoiceScribeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("voiceScribe")
        self.geometry("860x610")
        self.minsize(620, 460)

        self._running             = False
        self._initial_clear_done  = False   # 只在整个 App 生命周期内清屏一次
        self._count               = 0
        self._shown_text          = ""
        self._transcriber         = None
        self._live                = None
        self._loaded_key          = ""
        self._devices             = list_input_devices()
        self._device_map          = {n: i for i, n in self._devices}
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        bar1 = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#141428")
        bar1.grid(row=0, column=0, sticky="ew")
        bar1.grid_columnconfigure(1, weight=1)

        backend_tag   = f"M系列 ⚡ MLX" if BACKEND == "mlx" else ("M系列 CPU" if IS_APPLE_SILICON else "Intel CPU")
        backend_color = "#60cc60" if BACKEND == "mlx" else "#aaaaff"
        ctk.CTkLabel(bar1, text="🎙 voiceScribe",
            font=ctk.CTkFont(size=17, weight="bold"), text_color="#c8c8ff",
        ).grid(row=0, column=0, padx=(16,4), pady=10)
        ctk.CTkLabel(bar1, text=backend_tag,
            font=ctk.CTkFont(size=11), text_color=backend_color,
        ).grid(row=0, column=1, padx=(0,10), sticky="w")

        self._model_var = ctk.StringVar(value=MODEL_DEFAULT)
        ctk.CTkOptionMenu(bar1, values=MODEL_OPTIONS, variable=self._model_var, width=145,
            fg_color="#1e1e3e", button_color="#0f3460", button_hover_color="#1a4a7a",
        ).grid(row=0, column=2, padx=6)

        self._lang_var = ctk.StringVar(value="en")
        ctk.CTkOptionMenu(bar1, values=["en","zh","ja","ko","auto"],
            variable=self._lang_var, width=80,
            fg_color="#1e1e3e", button_color="#0f3460", button_hover_color="#1a4a7a",
        ).grid(row=0, column=3, padx=6)

        ctk.CTkButton(bar1, text="复制", width=60, fg_color="#2a2a44", hover_color="#3a3a5e",
            command=self._copy_all).grid(row=0, column=4, padx=6)
        ctk.CTkButton(bar1, text="清空", width=60, fg_color="#2a2a44", hover_color="#3a3a5e",
            command=self._clear).grid(row=0, column=5, padx=6)
        ctk.CTkButton(bar1, text="退出", width=60, fg_color="#3e1010", hover_color="#5e1818",
            command=self.quit_app).grid(row=0, column=6, padx=(0,14))

        bar2 = ctk.CTkFrame(self, height=40, corner_radius=0, fg_color="#0f0f20")
        bar2.grid(row=1, column=0, sticky="ew")
        bar2.grid_columnconfigure(1, weight=1)

        # ── 内录来源选择器（SCK_AVAILABLE 时启用，暂时关闭）─────────────────────
        # source_options = ["麦克风"]
        # if SCK_AVAILABLE:
        #     source_options.append("系统音频")
        # self._source_var = ctk.StringVar(value="麦克风")
        # ctk.CTkLabel(bar2, text="来源", ...).grid(...)
        # self._source_menu = ctk.CTkOptionMenu(bar2, values=source_options, ...).grid(...)
        self._source_var = ctk.StringVar(value="麦克风")   # 固定麦克风

        ctk.CTkLabel(bar2, text="设备", font=ctk.CTkFont(size=12),
            text_color="#6666aa").grid(row=0, column=0, padx=(16,4), pady=8)

        names = [n for _, n in self._devices]
        self._device_var = ctk.StringVar(value=self._pick_default(names))
        self._device_menu = ctk.CTkOptionMenu(bar2, values=names or ["（无设备）"],
            variable=self._device_var, width=400,
            fg_color="#1a1a30", button_color="#0f3460", button_hover_color="#1a4a7a",
            dynamic_resizing=False)
        self._device_menu.grid(row=0, column=1, padx=6, sticky="w")
        ctk.CTkButton(bar2, text="↻", width=32, height=26, font=ctk.CTkFont(size=14),
            fg_color="#1e1e3e", hover_color="#2e2e5e",
            command=self._refresh_devices).grid(row=0, column=2, padx=(0,14))

        tf = ctk.CTkFrame(self, fg_color="#080814", corner_radius=0)
        tf.grid(row=2, column=0, sticky="nsew")
        tf.grid_columnconfigure(0, weight=1)
        tf.grid_rowconfigure(0, weight=1)

        self._text = tk.Text(tf, bg="#080814", fg=COLOR_FINAL,
            font=("Helvetica Neue", 15), wrap=tk.WORD,
            state=tk.DISABLED, relief=tk.FLAT, bd=0,
            padx=18, pady=12, cursor="arrow")
        self._text.tag_configure("final", foreground=COLOR_FINAL)
        self._text.grid(row=0, column=0, sticky="nsew")
        sb = tk.Scrollbar(tf, command=self._text.yview, bg="#141428")
        sb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=sb.set)

        # partial 独立只读 Text：固定 3 行高，wrap=WORD，不在滚动区内
        self._partial_box = tk.Text(tf, height=3,
            bg="#080814", fg=COLOR_PARTIAL,
            font=("Helvetica Neue", 15), wrap=tk.WORD,
            state=tk.DISABLED, relief=tk.FLAT, bd=0,
            padx=18, pady=6, cursor="arrow")
        self._partial_box.grid(row=1, column=0, columnspan=2, sticky="ew")

        sbar = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color="#0e0e22")
        sbar.grid(row=3, column=0, sticky="ew")
        sbar.grid_columnconfigure(1, weight=1)

        self._toggle_btn = ctk.CTkButton(sbar, text="▶  开始", width=110, height=34,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0f3460", hover_color="#1a5090", command=self._toggle)
        self._toggle_btn.grid(row=0, column=0, padx=16, pady=11)

        mid = ctk.CTkFrame(sbar, fg_color="transparent")
        mid.grid(row=0, column=1, sticky="ew", padx=8)
        mid.grid_columnconfigure(0, weight=1)

        self._status_lbl = ctk.CTkLabel(mid, text="就绪",
            font=ctk.CTkFont(size=13), text_color="#6666aa")
        self._status_lbl.grid(row=0, column=0, sticky="w")

        self._rms_lbl = ctk.CTkLabel(mid, text="",
            font=ctk.CTkFont(size=11), text_color="#444466")
        self._rms_lbl.grid(row=0, column=1, padx=(12,0), sticky="w")

        self._level_bar = ctk.CTkProgressBar(mid, height=5,
            fg_color="#1a1a36", progress_color="#3a70c0")
        self._level_bar.set(0)
        self._level_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2,0))

        self._count_lbl = ctk.CTkLabel(sbar, text="0 句",
            font=ctk.CTkFont(size=12), text_color="#444466")
        self._count_lbl.grid(row=0, column=2, padx=16)

    def _pick_default(self, names):
        for n in names:
            if any(k in n.lower() for k in ("macbook","built-in","mac mini","rode")):
                return n
        return names[0] if names else ""

    # def _on_source_change(self, value):   # 内录功能暂时关闭
    #     is_mic = (value == "麦克风")
    #     self._device_menu.configure(state="normal" if is_mic else "disabled")

    def _refresh_devices(self):
        sd._terminate(); sd._initialize()
        self._devices    = list_input_devices()
        self._device_map = {n: i for i, n in self._devices}
        self._device_menu.configure(values=[n for _, n in self._devices])

    def _selected_device(self):
        return self._device_map.get(self._device_var.get())

    # ── 控制 ─────────────────────────────────────────────────────────────────

    def _toggle(self):
        self._stop() if self._running else self._start()

    def _start(self):
        self._running = True
        self._toggle_btn.configure(text="⏹  停止", fg_color="#5e1010", hover_color="#7e1818")
        self._set_status("加载模型…", "#c0903a")
        threading.Thread(target=self._init_and_run, daemon=True).start()

    def _stop(self):
        self._running    = False
        self._shown_text = ""
        if self._live:
            self._live.shutdown()
            self._live = None
        self._toggle_btn.configure(text="▶  开始", fg_color="#0f3460", hover_color="#1a5090")
        self._set_status("已停止", "#6666aa")
        self._level_bar.set(0)
        self._set_partial_box("")          # 清空实时蓝字
        self._rms_lbl.configure(text="")

    def _init_and_run(self):
        model = self._model_var.get()
        lang  = self._lang_var.get()
        key   = f"{model}:{lang}"

        if self._transcriber is None or self._loaded_key != key:
            self._append_log(f"加载模型 {model}（{BACKEND}）…\n")
            err = [None]
            done = threading.Event()
            def load():
                try:
                    self._transcriber = Transcriber(model)
                    self._loaded_key  = key
                except Exception as e:
                    err[0] = e
                finally:
                    done.set()
            threading.Thread(target=load, daemon=True).start()
            dots = 0
            while not done.wait(0.5):
                dots = (dots+1) % 4
                self._set_status(f"加载 {model}{'.'*dots}", "#c0903a")
            if err[0]:
                self._append_log(f"❌ 加载失败：{err[0]}\n")
                self.after(0, self._stop); return
            self._append_log(f"模型 {model} 就绪 ✓\n")

        # if self._source_var.get() == "系统音频":   # 内录功能暂时关闭
        #     self._run_sck()
        # else:
        self._run_mic()

    # ── 麦克风模式 ────────────────────────────────────────────────────────────

    def _run_mic(self):
        lang = self._lang_var.get()

        def on_partial(text):
            if not self._running: return
            self.after(0, self._show_partial, text)
            self._set_status("录音中…", "#c04040")

        def on_final(text):
            if not self._running: return
            self.after(0, self._show_final, text)
            self._set_status("监听中…", "#40a040")

        def on_level(level):
            if not self._running: return
            self.after(0, self._level_bar.set, min(level/2000.0, 1.0))
            self.after(0, self._rms_lbl.configure, {"text": f"RMS {int(level)}"})
            if level > SILENCE_RMS:
                self.after(0, self._clear_log_once)

        self._live = LiveTranscriber(self._transcriber, lang,
                                     on_partial, on_final, on_level)

        def callback(indata, frames, t, status):
            if self._running and self._live:
                self._live.push(indata[:, 0].copy())

        self._set_status("监听中…", "#40a040")
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                 dtype="int16", blocksize=CHUNK_SIZE,
                                 device=self._selected_device(),
                                 callback=callback):
                while self._running:
                    time.sleep(0.05)
        except Exception as e:
            self._append_log(f"❌ 麦克风错误：{e}\n")
            self.after(0, self._stop)

    # ── 系统音频模式（ScreenCaptureKit）—— 暂时关闭，待调试后启用 ─────────────
    #
    # def _run_sck(self):
    #     lang = self._lang_var.get()
    #     sck_q: queue.Queue = queue.Queue()
    #     audio_arrived = threading.Event()
    #
    #     def on_partial(text):
    #         if not self._running: return
    #         self.after(0, self._show_partial, text)
    #         self._set_status("录音中…", "#c04040")
    #
    #     def on_final(text):
    #         if not self._running: return
    #         self.after(0, self._show_final, text)
    #         self._set_status("监听中…", "#40a040")
    #
    #     def on_level(level):
    #         if not self._running: return
    #         self.after(0, self._level_bar.set, min(level / 2000.0, 1.0))
    #         self.after(0, self._rms_lbl.configure, {"text": f"RMS {int(level)}"})
    #         if level > SILENCE_RMS:
    #             self.after(0, self._clear_log_once)
    #
    #     self._live = LiveTranscriber(self._transcriber, lang,
    #                                  on_partial, on_final, on_level)
    #
    #     def _bridge():
    #         while self._running:
    #             try:
    #                 chunk = sck_q.get(timeout=0.1)
    #                 audio_arrived.set()
    #                 if self._live:
    #                     self._live.push(chunk)
    #             except queue.Empty:
    #                 continue
    #     threading.Thread(target=_bridge, daemon=True).start()
    #
    #     try:
    #         cap = SystemAudioCapture(sck_q)
    #         cap.start()
    #     except Exception as e:
    #         self._append_log(
    #             f"❌ 系统音频启动失败：{e}\n"
    #             f"请前往「系统设置 › 隐私与安全性 › 屏幕与系统录音」授权后重启应用。\n"
    #         )
    #         self.after(0, self._stop); return
    #
    #     self._append_log("✓ 系统音频捕获已启动（请确保其他应用正在播放音频）\n")
    #     self._set_status("监听中…", "#40a040")
    #
    #     def _check_audio():
    #         time.sleep(3.0)
    #         if self._running and not audio_arrived.is_set():
    #             self._append_log(
    #                 "⚠ 3 秒内未收到数据。\n原因：屏幕录制权限未授权 或 当前无音频播放。\n"
    #             )
    #     threading.Thread(target=_check_audio, daemon=True).start()
    #
    #     try:
    #         while self._running:
    #             time.sleep(0.05)
    #     finally:
    #         cap.stop()

    # ── UI 更新 ───────────────────────────────────────────────────────────────

    def _set_partial_box(self, text: str):
        self._partial_box.configure(state=tk.NORMAL)
        self._partial_box.delete("1.0", tk.END)
        if text:
            self._partial_box.insert("1.0", text)
        self._partial_box.configure(state=tk.DISABLED)

    def _show_partial(self, text: str):
        if not text or text == self._shown_text:
            return
        self._clear_log_once()
        self._set_partial_box(text)
        self._shown_text = text

    def _show_final(self, text: str):
        self._count += 1
        self._count_lbl.configure(text=f"{self._count} 句")
        self._clear_log_once()
        self._set_partial_box("")
        self._shown_text = ""
        if text:
            self._text.configure(state=tk.NORMAL)
            self._text.insert(tk.END, text + "\n", "final")
            self._text.configure(state=tk.DISABLED)
            self._text.see(tk.END)

    def _clear_log_once(self):
        """只在整个 App 首次正式监听时清屏一次，之后永不自动清屏。"""
        if self._initial_clear_done:
            return
        self._initial_clear_done = True
        self._shown_text = ""
        self._set_partial_box("")
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)

    def _append_log(self, msg: str):
        self.after(0, lambda: (
            self._text.configure(state=tk.NORMAL),
            self._text.insert(tk.END, msg, "final"),
            self._text.configure(state=tk.DISABLED),
            self._text.see(tk.END),
        ))

    def _copy_all(self):
        parts = [self._text.get("1.0", tk.END).strip(),
                 self._partial_box.get("1.0", tk.END).strip()]
        t = "\n".join(p for p in parts if p)
        if t: self.clipboard_clear(); self.clipboard_append(t)

    def _clear(self):
        self._shown_text = ""
        self._count = 0
        self._count_lbl.configure(text="0 句")
        self._set_partial_box("")
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)

    def _set_status(self, text, color="#6666aa"):
        self.after(0, lambda: self._status_lbl.configure(text=text, text_color=color))

    def on_close(self):
        if self._running: self._stop()
        self.withdraw()

    def quit_app(self):
        self._running = False
        if self._live: self._live.shutdown()
        self.destroy()


if __name__ == "__main__":
    app = VoiceScribeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.createcommand("tk::mac::ReopenApplication", app.deiconify)
    app.mainloop()
