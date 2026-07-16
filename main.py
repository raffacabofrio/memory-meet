import sys, site
if site.getusersitepackages() not in sys.path:
    sys.path.insert(0, site.getusersitepackages())

import tkinter as tk
import customtkinter as ctk
import pyaudiowpatch as pyaudio
import numpy as np
import threading
import queue
import datetime
import os
import time
import logging
import lameenc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from transcribers import get_transcriber
from transcribers.base import SpeakerRef

def _get_documents_dir() -> Path:
    try:
        import ctypes, ctypes.wintypes
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(0, 5, 0, 0, buf)  # CSIDL_PERSONAL
        return Path(buf.value)
    except Exception:
        return Path.home() / "Documents"

APP_DIR = _get_documents_dir() / "MemoryMeet"
APP_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    filename=APP_DIR / "memorymeet.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CHUNK          = 1024
FORMAT         = pyaudio.paInt16
CHANNELS       = 2
MP3_BITRATE    = 128
CHUNK_SEGUNDOS = 5 * 60
WORKERS        = int(os.getenv("MEMORYMEET_WORKERS", "1"))  # N>1 exige transcriber thread-safe (ver processar_chunk)
SPEAKER_MIC    = os.getenv("MEMORYMEET_SPEAKER", "Me")               # rótulo do canal do mic (você)
SPEAKER_SYS    = os.getenv("MEMORYMEET_INTERLOCUTOR", "Interlocutor")  # rótulo do canal do sistema (o outro lado)
REF_ALVO_SEG    = 8         # duração da janela de referência de voz (2-10s)
REF_MIN_ENERGIA = 150       # piso de energia (mean abs int16) p/ considerar que o canal tem fala (não ancorar silêncio)

BG       = "#1a1a2e"
RED      = "#e05050"
GREEN    = "#50c878"
ORANGE   = "#f0a030"
TEXT_DIM = "#55557a"
BLUE     = "#4a9eff"
VU_W     = 220


# ── áudio ─────────────────────────────────────────────────────────────────────

def mix_frames(frames_a, frames_b):
    min_len = min(len(frames_a), len(frames_b))
    if min_len == 0:
        return np.concatenate(frames_a) if frames_a else np.concatenate(frames_b)
    a = np.concatenate(frames_a[:min_len]).astype(np.int32)
    b = np.concatenate(frames_b[:min_len]).astype(np.int32)
    s = min(len(a), len(b))
    return np.clip((a[:s] + b[:s]) // 2, -32768, 32767).astype(np.int16)


def audio_para_mp3(audio, rate):
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(rate)
    enc.set_channels(CHANNELS)
    enc.set_quality(2)
    return enc.encode(audio.tobytes()) + enc.flush()


def best_speech_window(audio, win_len, step):
    """Retorna (trecho, energia) da janela de maior energia — provável fala mais limpa do canal."""
    a = np.abs(audio.astype(np.int32))
    if len(audio) <= win_len:
        return audio, (float(a.mean()) if len(a) else 0.0)
    best_start, best_energy = 0, -1.0
    for start in range(0, len(audio) - win_len + 1, step):
        energy = float(a[start:start + win_len].mean())
        if energy > best_energy:
            best_energy, best_start = energy, start
    return audio[best_start:best_start + win_len], best_energy


def _seg_attr(seg, key, default=""):
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)


def format_segments(segments):
    """diarized_json -> texto com [speaker] por turno, mesclando turnos consecutivos do mesmo falante."""
    blocos = []
    for seg in segments:
        speaker = _seg_attr(seg, "speaker", "") or "SPEAKER"
        text = (_seg_attr(seg, "text", "") or "").strip()
        if not text:
            continue
        if blocos and blocos[-1][0] == speaker:
            blocos[-1][1] += " " + text
        else:
            blocos.append([speaker, text])
    return "\n\n".join(f"[{spk}] {txt}" for spk, txt in blocos)


# ── pipeline de chunks ────────────────────────────────────────────────────────
# Cortador (produtor) → fila de jobs → workers (função pura) → fila de resultados
# → orquestrador (consumidor único dos side effects: MP3, TXT e UI).

@dataclass(frozen=True)
class ChunkJob:
    """Trabalho imutável: frames crus + snapshot das referências de voz."""
    index: int
    mic: tuple
    sys: tuple
    refs: tuple          # SpeakerRefs já ancoradas — worker não muta nada
    is_final: bool

    @property
    def label(self):
        return "final" if self.is_final else str(self.index)


@dataclass(frozen=True)
class ChunkResult:
    index: int
    is_final: bool
    mp3: Optional[bytes]
    texto: str
    erro: Optional[str] = None


def processar_chunk(job: ChunkJob, transcriber, rate: int) -> ChunkResult:
    """Função pura: não escreve arquivo, não toca UI, não muta estado compartilhado.
       É o que permite N workers em paralelo — mas N>1 exige transcriber thread-safe;
       o WhisperX local compartilha o modelo entre chamadas, então manter WORKERS=1
       até rodar em GPU ou com um modelo por worker."""
    try:
        if not job.mic and not job.sys:
            return ChunkResult(job.index, job.is_final, None, "")
        audio = mix_frames(job.mic, job.sys) if (job.mic and job.sys) else (
                np.concatenate(job.mic) if job.mic else np.concatenate(job.sys))
        logging.info("Chunk %s — samples: %d", job.label, len(audio))
        mp3 = audio_para_mp3(audio, rate)
        logging.info("Chunk %s — MP3 %.1f MB", job.label, len(mp3) / 1024 / 1024)
        if transcriber is None:   # modelo falhou ao carregar — preserva o áudio mesmo assim
            logging.warning("Chunk %s — sem transcriber, salvando só o MP3", job.label)
            return ChunkResult(job.index, job.is_final, mp3, "")
        segments = transcriber.transcribe_and_diarize(audio, rate, list(job.refs))
        texto = format_segments(segments)
        logging.info("Chunk %s — %d segmentos, %d chars", job.label, len(segments), len(texto))
        return ChunkResult(job.index, job.is_final, mp3, texto)
    except Exception as e:
        logging.error("Erro chunk %s: %s", job.label, e, exc_info=True)
        return ChunkResult(job.index, job.is_final, None, "", str(e))


# ── app ───────────────────────────────────────────────────────────────────────

class MemoryMeet:
    def __init__(self, root):
        self.root = root
        self.root.title("MemoryMeet")
        self.root.resizable(False, False)
        self.root.geometry("320x360")
        self.root.configure(fg_color=BG)

        icon = Path(__file__).parent / "assets" / "MemoryMeet.ico"
        if icon.exists():
            self.root.iconbitmap(str(icon))

        self.gravando         = False
        self.stop_event       = threading.Event()
        self.start_time       = None
        self.timer_job        = None
        self._thread_mic      = None
        self._thread_sys      = None
        self._thread_keepalive = None
        self._ultimo_arquivo  = None
        self._duracao_gravada = 0
        self._mp3_path        = None
        self._txt_path        = None
        self._ref_mic         = None
        self._ref_sys         = None
        # Modelo carrega em background: whisperx puxa torch + 3 modelos (~dezenas de
        # segundos em CPU) e travava o mainloop antes da janela abrir. A gravação não
        # depende dele — só o worker espera o evento antes de transcrever.
        self.transcriber        = None
        self._transcriber_ready = threading.Event()

        self.p = pyaudio.PyAudio()
        try:
            self.loopback      = self.p.get_default_wasapi_loopback()
            self.rate          = int(self.loopback["defaultSampleRate"])
            self.loop_channels = min(int(self.loopback["maxInputChannels"]), 2)
            self.sample_width  = self.p.get_sample_size(FORMAT)
            logging.info("Loopback: %s | rate: %d", self.loopback["name"], self.rate)
        except Exception as e:
            logging.error("Falha ao inicializar loopback: %s", e)
            self.loopback = None

        self._build_ui()
        threading.Thread(target=self._load_transcriber, daemon=True).start()

    def _load_transcriber(self):
        try:
            t0 = time.time()
            self.transcriber = get_transcriber()
            logging.info("Transcriber carregado em %.1fs", time.time() - t0)
            self.root.after(0, lambda: self.lbl_modelo.configure(text=""))
        except Exception as e:
            logging.error("Falha ao carregar transcriber: %s", e, exc_info=True)
            self.root.after(0, self._modelo_falhou)
        finally:
            self._transcriber_ready.set()

    def _modelo_falhou(self):
        self.btn_gravar.configure(state="disabled")
        self.lbl_modelo.configure(text="Erro ao carregar modelo (ver log)", text_color=RED)

    # ── build ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._logo_img = self._load_logo()

        # --- Tela Idle ---
        self.frame_idle = ctk.CTkFrame(self.root, fg_color="transparent")

        if self._logo_img:
            ctk.CTkLabel(self.frame_idle, image=self._logo_img, text="").pack(pady=(32, 0))
        else:
            ctk.CTkLabel(self.frame_idle, text="MemoryMeet",
                         font=ctk.CTkFont(size=24, weight="bold"),
                         text_color="#e0e0f8").pack(pady=(50, 0))

        # spacer elástico empurra o botão pro rodapé
        ctk.CTkFrame(self.frame_idle, fg_color="transparent", height=0).pack(expand=True, fill="y")

        self.btn_gravar = ctk.CTkButton(
            self.frame_idle, text="● Gravar",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=200, height=50, corner_radius=25,
            fg_color=RED, hover_color="#c03030",
            command=self.iniciar
        )
        self.btn_gravar.pack(pady=(0, 8))

        self.lbl_modelo = ctk.CTkLabel(
            self.frame_idle, text="Carregando modelo…",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, height=14
        )
        self.lbl_modelo.pack(pady=(0, 10))

        if self.loopback is None:
            self.btn_gravar.configure(state="disabled")
            ctk.CTkLabel(self.frame_idle, text="Erro: loopback não encontrado",
                         text_color=RED, font=ctk.CTkFont(size=11)).pack()

        # --- Tela Gravando ---
        self.frame_recording = ctk.CTkFrame(self.root, fg_color="transparent")

        self.lbl_timer = ctk.CTkLabel(
            self.frame_recording, text="00:00",
            font=ctk.CTkFont(family="Courier", size=52, weight="bold"),
            text_color="#e0e0f8"
        )
        self.lbl_timer.pack(pady=(24, 0))

        self.vu_canvas = tk.Canvas(
            self.frame_recording, width=VU_W, height=5,
            bg="#13131f", highlightthickness=0
        )
        self.vu_canvas.pack(pady=(12, 0))

        self.lbl_status = ctk.CTkLabel(
            self.frame_recording, text="Gravando...",
            font=ctk.CTkFont(size=12), text_color=RED
        )
        self.lbl_status.pack(pady=(10, 0))

        self.spinner = ctk.CTkProgressBar(
            self.frame_recording, width=160, height=4,
            mode="indeterminate", indeterminate_speed=1.2,
            progress_color=ORANGE, fg_color="#2a2a3a"
        )
        self.spinner.pack(pady=(6, 0))
        self.spinner.pack_forget()

        ctk.CTkFrame(self.frame_recording, fg_color="transparent", height=0).pack(expand=True, fill="y")

        self.btn_parar = ctk.CTkButton(
            self.frame_recording, text="■ Parar",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=200, height=50, corner_radius=25,
            fg_color="#333344", hover_color="#444455",
            command=self.parar
        )
        self.btn_parar.pack(pady=(0, 32))

        # --- Tela Resultado ---
        self.frame_result = ctk.CTkFrame(self.root, fg_color="transparent")

        # link no rodapé — packed primeiro com side="bottom"
        self.lbl_nova = ctk.CTkLabel(
            self.frame_result, text="Nova gravação →",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM, cursor="hand2"
        )
        self.lbl_nova.pack(side="bottom", pady=(0, 32))
        self.lbl_nova.bind("<Button-1>", lambda _: self._show_idle())

        # spacer inferior — empurra conteúdo pra cima
        ctk.CTkFrame(self.frame_result, fg_color="transparent", height=0).pack(
            side="bottom", expand=True, fill="y")

        # spacer superior — empurra conteúdo pra baixo (juntos, centralizam)
        ctk.CTkFrame(self.frame_result, fg_color="transparent", height=0).pack(
            side="top", expand=True, fill="y")

        # conteúdo central
        self.doc_canvas = tk.Canvas(
            self.frame_result, width=56, height=70,
            bg=BG, highlightthickness=0
        )
        self.doc_canvas.pack(pady=(0, 10))
        self.doc_canvas.bind("<Button-1>", self._abrir_arquivo)
        self.doc_canvas.configure(cursor="hand2")
        self._draw_doc_icon()

        self.lbl_resultado = ctk.CTkLabel(
            self.frame_result, text="",
            font=ctk.CTkFont(size=13, underline=True),
            text_color=BLUE, cursor="hand2"
        )
        self.lbl_resultado.pack()
        self.lbl_resultado.bind("<Button-1>", self._abrir_arquivo)

        self.lbl_preview = ctk.CTkLabel(
            self.frame_result, text="",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM, wraplength=270, justify="left"
        )
        self.lbl_preview.pack(pady=(10, 0), padx=20)


        self._show_idle()

    def _load_logo(self):
        try:
            from PIL import Image
            logo_path = Path(__file__).parent / "assets" / "logo.png"
            if not logo_path.exists():
                return None
            img = Image.open(logo_path).convert("RGBA").resize((160, 160), Image.LANCZOS)
            return ctk.CTkImage(light_image=img, dark_image=img, size=(160, 160))
        except Exception as e:
            logging.warning("Logo não carregado: %s", e)
        return None

    def _draw_doc_icon(self):
        c, w, h, fold = self.doc_canvas, 56, 70, 14
        c.create_polygon(0, 0, w-fold, 0, w, fold, w, h, 0, h,
                         fill="#e8e8f4", outline="#8888aa", width=1)
        c.create_polygon(w-fold, 0, w, fold, w-fold, fold,
                         fill="#c0c0d8", outline="#8888aa", width=1)
        for y in range(h//3, h-6, 11):
            c.create_line(9, y, w-9, y, fill="#9999bb", width=2)

    # ── navegação entre telas ──────────────────────────────────────────────────

    def _show_idle(self):
        self.frame_recording.pack_forget()
        self.frame_result.pack_forget()
        self.frame_idle.pack(fill="both", expand=True)

    def _show_recording(self):
        self.frame_idle.pack_forget()
        self.frame_result.pack_forget()
        self.lbl_status.configure(text="Gravando...", text_color=RED,
                                  font=ctk.CTkFont(size=12))
        self.frame_recording.pack(fill="both", expand=True)

    def _show_result(self):
        arquivo = str(self._txt_path) if self._txt_path and self._txt_path.exists() \
                  else str(self._mp3_path)
        self._ultimo_arquivo = arquivo

        self.lbl_resultado.configure(text=Path(arquivo).name)

        preview = ""
        if self._txt_path and self._txt_path.exists():
            try:
                text = self._txt_path.read_text(encoding="utf-8")
                preview = text[:160].strip()
                if len(text) > 160:
                    preview += "…"
            except Exception:
                pass
        self.lbl_preview.configure(text=preview)

        self.frame_idle.pack_forget()
        self.frame_recording.pack_forget()
        self.frame_result.pack(fill="both", expand=True)

    # ── controle ──────────────────────────────────────────────────────────────

    def iniciar(self):
        self.gravando = True
        self.stop_event.clear()
        self.mic_frames   = []
        self.sys_frames   = []
        self._chunk_index = 0
        self._ultimo_arquivo = None
        self._ref_mic     = None
        self._ref_sys     = None
        self._chunks_emitidos   = 0
        self._chunks_concluidos = 0
        self._txt_escrito       = False
        self.job_queue    = queue.Queue()
        self.result_queue = queue.Queue()

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._mp3_path = APP_DIR / f"meet_{ts}.mp3"
        self._txt_path = APP_DIR / f"meet_{ts}.txt"

        self._show_recording()
        self.start_time = time.time()
        self._tick()
        self._vu_tick()

        logging.info("Gravação iniciada")
        # keep-alive PRIMEIRO: mantém a saída ativa pro loopback BT não suspender no silêncio
        # (senão o canal do sistema perde frames e dessincroniza do mic — vira mix sobreposto)
        self._thread_keepalive = threading.Thread(target=self._keep_output_alive, daemon=True)
        self._thread_keepalive.start()
        self._thread_mic = threading.Thread(target=self._record_mic, daemon=True)
        self._thread_sys = threading.Thread(target=self._record_system, daemon=True)
        self._thread_mic.start()
        self._thread_sys.start()
        for _ in range(WORKERS):
            threading.Thread(target=self._worker_loop, daemon=True).start()
        threading.Thread(target=self._orchestrator_loop, daemon=True).start()
        threading.Thread(target=self._cutter_loop, daemon=True).start()

    def parar(self):
        self.gravando = False
        self._duracao_gravada = int(time.time() - self.start_time)
        self.stop_event.set()
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.btn_parar.configure(fg_color="#2a2a3a", hover_color="#2a2a3a",
                                 text_color="#44445a", state="disabled")
        self._vu_clear()
        self.root.after(0, lambda: self._start_spinner("Finalizando", ORANGE))

    def _tick(self):
        if self.gravando:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            self.lbl_timer.configure(text=f"{m:02d}:{s:02d}")
            self.timer_job = self.root.after(1000, self._tick)

    # ── VU meter ──────────────────────────────────────────────────────────────

    def _vu_tick(self):
        if not self.gravando:
            return
        if self.mic_frames:
            rms   = float(np.sqrt(np.mean(self.mic_frames[-1].astype(np.float32) ** 2)))
            level = min(rms / 6000, 1.0)
            self._vu_draw(level)
        self.root.after(80, self._vu_tick)

    def _vu_draw(self, level):
        self.vu_canvas.delete("all")
        w = max(1, int(VU_W * level))
        color = GREEN if level < 0.6 else (ORANGE if level < 0.85 else RED)
        self.vu_canvas.create_rectangle(0, 0, w, 5, fill=color, outline="")

    def _vu_clear(self):
        self.vu_canvas.delete("all")

    # ── spinner ───────────────────────────────────────────────────────────────

    def _start_spinner(self, msg, color):
        self.lbl_status.configure(text=msg, text_color=color, font=ctk.CTkFont(size=12))
        self.btn_parar.pack_forget()
        self.spinner.pack(after=self.lbl_status, pady=(70, 0))
        self.spinner.start()

    def _stop_spinner(self):
        self.spinner.stop()
        self.spinner.pack_forget()

    # ── gravação ──────────────────────────────────────────────────────────────

    def _keep_output_alive(self):
        """Toca silêncio inaudível contínuo na saída padrão enquanto grava. Mantém o endpoint
           ativo pro loopback WASAPI (sobretudo Bluetooth) não suspender durante o silêncio —
           o que faria o canal do sistema perder frames e dessincronizar do mic."""
        try:
            out = self.p.open(format=FORMAT, channels=CHANNELS, rate=self.rate,
                              output=True, frames_per_buffer=CHUNK)
            silencio = np.zeros(CHUNK * CHANNELS, dtype=np.int16).tobytes()
            while not self.stop_event.is_set():
                out.write(silencio)               # bloqueia no ritmo real → mantém a saída viva
            out.stop_stream(); out.close()
            logging.info("Keep-alive de saída encerrado")
        except Exception as e:
            logging.error("Erro keep-alive (gravação segue sem ele): %s", e, exc_info=True)

    def _record_mic(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=self.rate,
                                 input=True, frames_per_buffer=CHUNK)
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.mic_frames.append(np.frombuffer(data, dtype=np.int16).copy())
            stream.stop_stream(); stream.close()
            logging.info("Mic encerrado. Frames: %d", len(self.mic_frames))
        except Exception as e:
            logging.error("Erro mic: %s", e, exc_info=True)

    def _record_system(self):
        try:
            stream = self.p.open(format=FORMAT, channels=self.loop_channels, rate=self.rate,
                                 input=True, input_device_index=self.loopback["index"],
                                 frames_per_buffer=CHUNK)
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                arr  = np.frombuffer(data, dtype=np.int16).copy()
                if self.loop_channels == 1:
                    arr = np.repeat(arr, 2)
                self.sys_frames.append(arr)
            stream.stop_stream(); stream.close()
            logging.info("Sistema encerrado. Frames: %d", len(self.sys_frames))
        except Exception as e:
            logging.error("Erro sistema: %s", e, exc_info=True)

    def _swap_frames(self):
        mic, sys = self.mic_frames, self.sys_frames
        self.mic_frames, self.sys_frames = [], []
        return mic, sys

    def _cutter_loop(self):
        """Produtor: corta a cada CHUNK_SEGUNDOS e enfileira — nunca espera transcrição.
           (Antes cortava só depois de transcrever, e os chunks cresciam de 5 pra 18 min.)"""
        while True:
            parou = self.stop_event.wait(timeout=CHUNK_SEGUNDOS)
            if parou:
                for t in (self._thread_mic, self._thread_sys):
                    if t:
                        t.join(timeout=10)
            mic, sys = self._swap_frames()
            if mic or sys or parou:
                self._anchor_references(mic, sys)
                self._chunk_index += 1
                self._chunks_emitidos = self._chunk_index
                self.job_queue.put(ChunkJob(self._chunk_index, tuple(mic), tuple(sys),
                                            self._speaker_refs(), parou))
                self._update_progress()
            if parou:
                return

    def _worker_loop(self):
        self._transcriber_ready.wait()   # modelo carrega em background no startup
        while True:
            job = self.job_queue.get()
            if job is None:
                return
            self.result_queue.put(processar_chunk(job, self.transcriber, self.rate))

    def _orchestrator_loop(self):
        """Consumidor único dos side effects: recebe resultados em qualquer ordem,
           reordena por índice e só ele escreve MP3/TXT e atualiza a UI."""
        pendentes, proximo = {}, 1
        while True:
            r = self.result_queue.get()
            pendentes[r.index] = r
            while proximo in pendentes:
                r = pendentes.pop(proximo)
                self._gravar_resultado(r)
                proximo += 1
                self._chunks_concluidos = r.index
                if r.is_final:
                    for _ in range(WORKERS):
                        self.job_queue.put(None)
                    self.root.after(0, self._finalizar)
                    return
                self._update_progress()

    def _gravar_resultado(self, r: ChunkResult):
        if r.erro:
            return                      # já logado no worker; os próximos chunks seguem
        if r.mp3:
            with open(self._mp3_path, "ab") as f:
                f.write(r.mp3)
        if r.texto:
            with open(self._txt_path, "a", encoding="utf-8") as f:
                if self._txt_escrito:
                    f.write("\n\n")
                f.write(r.texto)
            self._txt_escrito = True

    def _anchor_references(self, mic, sys):
        # Ancora os dois falantes no cortador (barato — só varre energia): o job leva
        # um snapshot imutável e o worker fica puro. Montada no 1º chunk com fala
        # suficiente e reusada nos seguintes — o diarize renumera falantes por chamada,
        # a âncora estabiliza os rótulos.
        if self._ref_mic is None:
            self._ref_mic = self._build_reference(mic)
            if self._ref_mic is not None:
                logging.info("Referência [%s] montada", SPEAKER_MIC)
        if self._ref_sys is None:
            self._ref_sys = self._build_reference(sys)
            if self._ref_sys is not None:
                logging.info("Referência [%s] montada", SPEAKER_SYS)

    def _speaker_refs(self):
        refs = []
        if self._ref_mic is not None:
            refs.append(SpeakerRef(name=SPEAKER_MIC, audio=self._ref_mic, rate=self.rate))
        if self._ref_sys is not None:
            refs.append(SpeakerRef(name=SPEAKER_SYS, audio=self._ref_sys, rate=self.rate))
        return tuple(refs)

    def _update_progress(self):
        feitos, total = self._chunks_concluidos, self._chunks_emitidos
        def aplicar():
            if self.gravando:
                texto = "Gravando..."
                if feitos < total:
                    texto += f"   ·   transcrevendo {feitos + 1} de {total}"
                self.lbl_status.configure(text=texto, text_color=RED)
            elif feitos < total:
                self.lbl_status.configure(text=f"Finalizando  ·  trecho {feitos + 1} de {total}",
                                          text_color=ORANGE)
        self.root.after(0, aplicar)

    def _build_reference(self, frames):
        """Acha os ~8s de fala mais limpa do canal (varre tudo, pega a janela de maior energia) e
           devolve o clip cru (mesmo dtype/rate da captura). None se o canal não tiver fala
           suficiente (não ancora silêncio). Cada transcriber decide como codificar/usar o clip."""
        try:
            if not frames:
                return None
            audio = np.concatenate(frames)
            spc = self.rate * CHANNELS                 # samples por segundo (stereo interleaved)
            if len(audio) < 2 * spc:                   # menos de 2s não serve
                return None
            win = min(REF_ALVO_SEG * spc, len(audio))
            clip, energia = best_speech_window(audio, win, step=spc)
            if energia < REF_MIN_ENERGIA:              # só silêncio/ruído — não ancora
                return None
            return clip
        except Exception as e:
            logging.error("Falha ao montar referência: %s", e, exc_info=True)
            return None

    def _finalizar(self):
        try:
            if not self._mp3_path or not self._mp3_path.exists():
                self._stop_spinner()
                self.lbl_status.configure(text="Nenhum áudio capturado.", text_color=RED)
                self.btn_parar.configure(state="normal", fg_color="#333344",
                                         hover_color="#444455", text_color=("white", "white"))
                return

            self._stop_spinner()
            self.root.after(0, self._show_result)

        except Exception as e:
            logging.error("Erro em _finalizar: %s", e, exc_info=True)
            self._stop_spinner()
            self.lbl_status.configure(text=f"Erro: {e}", text_color=RED)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _abrir_arquivo(self, _e=None):
        if self._ultimo_arquivo and Path(self._ultimo_arquivo).exists():
            os.startfile(self._ultimo_arquivo)

    def fechar(self):
        self.stop_event.set()
        self.root.after(100, self._shutdown)

    def _shutdown(self):
        self.p.terminate()
        self.root.destroy()


if __name__ == "__main__":
    try:
        root = ctk.CTk()
        app  = MemoryMeet(root)
        root.protocol("WM_DELETE_WINDOW", app.fechar)
        root.mainloop()
    except Exception as e:
        import traceback
        crash_log = APP_DIR / "crash.log"
        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            traceback.print_exc(file=f)
        raise
