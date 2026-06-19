import sys, site
if site.getusersitepackages() not in sys.path:
    sys.path.insert(0, site.getusersitepackages())

import tkinter as tk
import customtkinter as ctk
import pyaudiowpatch as pyaudio
import numpy as np
import threading
import datetime
import os
import io
import time
import logging
import lameenc
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

APP_DIR = Path.home() / "Documents" / "MemoryMeet"
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

CHUNK         = 1024
FORMAT        = pyaudio.paInt16
CHANNELS      = 2
MP3_BITRATE   = 128
CHUNK_SEGUNDOS = 5 * 60

BG      = "#1a1a2e"
RED     = "#e05050"
GREEN   = "#50c878"
ORANGE  = "#f0a030"
TEXT_DIM = "#55557a"
VU_W    = 220


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


# ── app ───────────────────────────────────────────────────────────────────────

class MemoryMeet:
    def __init__(self, root):
        self.root = root
        self.root.title("MemoryMeet")
        self.root.resizable(False, False)
        self.root.geometry("320x260")
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
        self._ultimo_arquivo  = None
        self._duracao_gravada = 0
        self._anim_job        = None
        self._anim_base       = ""
        self._anim_dots       = 0
        self._anim_color      = ORANGE

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

    def _build_ui(self):
        self.lbl_timer = ctk.CTkLabel(
            self.root, text="00:00",
            font=ctk.CTkFont(family="Courier", size=52, weight="bold"),
            text_color="#e0e0f8"
        )
        self.lbl_timer.pack(pady=(24, 0))

        # VU meter — mantém Canvas para controle de cor dinâmica
        self.vu_canvas = tk.Canvas(
            self.root, width=VU_W, height=5,
            bg="#13131f", highlightthickness=0
        )
        self.vu_canvas.pack(pady=(12, 0))

        self.lbl_status = ctk.CTkLabel(
            self.root, text="Pronto",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM, cursor="arrow"
        )
        self.lbl_status.pack(pady=(10, 0))
        self.lbl_status.bind("<Button-1>", self._abrir_arquivo)

        self.btn = ctk.CTkButton(
            self.root,
            text="● Gravar",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=200, height=50,
            corner_radius=25,
            fg_color=RED,
            hover_color="#c03030",
            command=self.toggle
        )
        self.btn.pack(pady=(20, 0))

        if self.loopback is None:
            self._set_status("Erro: loopback não encontrado", RED)
            self.btn.configure(state="disabled")

    # ── controle ──────────────────────────────────────────────────────────────

    def toggle(self):
        if self.gravando:
            self.parar()
        else:
            self.iniciar()

    def iniciar(self):
        self.gravando = True
        self.stop_event.clear()
        self.mic_frames    = []
        self.sys_frames    = []
        self._mp3_chunks   = []
        self._transcricoes = []
        self._chunk_index  = 0
        self._ultimo_arquivo = None

        self.btn.configure(text="■ Parar", fg_color="#333344", hover_color="#444455")
        self._set_status("Gravando...", RED)

        self.start_time = time.time()
        self._tick()
        self._vu_tick()

        logging.info("Gravação iniciada")
        self._thread_mic = threading.Thread(target=self._record_mic, daemon=True)
        self._thread_sys = threading.Thread(target=self._record_system, daemon=True)
        self._thread_mic.start()
        self._thread_sys.start()
        threading.Thread(target=self._chunk_loop, daemon=True).start()

    def parar(self):
        self.gravando = False
        self._duracao_gravada = int(time.time() - self.start_time)
        self.stop_event.set()
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.btn.configure(text="● Gravar", fg_color="#2a2a3a",
                           hover_color="#2a2a3a", text_color="#44445a",
                           state="disabled")
        self._vu_clear()
        self._start_dot_animation("Finalizando", ORANGE)

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

    # ── dot animation ─────────────────────────────────────────────────────────

    def _start_dot_animation(self, base, color):
        self._stop_dot_animation()
        self._anim_base  = base
        self._anim_color = color
        self._anim_dots  = 0
        self._animate_dots()

    def _animate_dots(self):
        self._anim_dots = (self._anim_dots + 1) % 4
        self.lbl_status.configure(
            text=self._anim_base + "." * self._anim_dots,
            text_color=self._anim_color,
            cursor="arrow",
            font=ctk.CTkFont(size=12)
        )
        self._anim_job = self.root.after(400, self._animate_dots)

    def _stop_dot_animation(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    # ── gravação ──────────────────────────────────────────────────────────────

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

    def _chunk_loop(self):
        while True:
            parou = self.stop_event.wait(timeout=CHUNK_SEGUNDOS)
            if parou:
                for t in (self._thread_mic, self._thread_sys):
                    if t:
                        t.join(timeout=10)
            mic, sys = self._swap_frames()
            if mic or sys:
                self._chunk_index += 1
                self._processar_chunk(mic, sys, is_final=parou)
            if parou:
                break
        self._finalizar()

    def _processar_chunk(self, mic, sys, is_final=False):
        label = "final" if is_final else str(self._chunk_index)
        try:
            audio = mix_frames(mic, sys) if (mic and sys) else (
                    np.concatenate(mic) if mic else np.concatenate(sys))

            logging.info("Chunk %s — samples: %d", label, len(audio))
            mp3 = audio_para_mp3(audio, self.rate)
            self._mp3_chunks.append(mp3)
            logging.info("Chunk %s — MP3 %.1f MB", label, len(mp3) / 1024 / 1024)

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return

            resultado = OpenAI(api_key=api_key).audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("audio.mp3", io.BytesIO(mp3)),
                timeout=120,
            )
            self._transcricoes.append(resultado.text)
            logging.info("Chunk %s — %d chars", label, len(resultado.text))

            if not is_final and self.gravando:
                self._flash_status("✓ trecho processado", GREEN, restore_after_ms=3000)

        except Exception as e:
            logging.error("Erro chunk %s: %s", label, e, exc_info=True)

    def _flash_status(self, msg, color, restore_after_ms=3000):
        self._stop_dot_animation()
        self.lbl_status.configure(text=msg, text_color=color, cursor="arrow",
                                  font=ctk.CTkFont(size=12))
        self.root.after(restore_after_ms,
                        lambda: self._set_status("Gravando...", RED) if self.gravando else None)

    def _finalizar(self):
        try:
            if not self._mp3_chunks:
                self._stop_dot_animation()
                self._set_status("Nenhum áudio capturado.", RED)
                self._reativar_btn()
                return

            ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            base = str(APP_DIR / f"meet_{ts}")

            with open(base + ".mp3", "wb") as f:
                for c in self._mp3_chunks:
                    f.write(c)

            self._stop_dot_animation()
            m, s    = divmod(self._duracao_gravada, 60)
            duracao = f"{m}min {s:02d}s"

            if self._transcricoes:
                txt   = base + ".txt"
                texto = "\n\n".join(self._transcricoes)
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(texto)
                palavras = len(texto.split())
                self._set_status(f"✓ {duracao} · {palavras:,} palavras — clique para abrir",
                                 GREEN, arquivo=txt)
            else:
                mp3 = base + ".mp3"
                self._set_status(f"✓ {duracao} · MP3 salvo — clique para abrir",
                                 GREEN, arquivo=mp3)

        except Exception as e:
            logging.error("Erro em _finalizar: %s", e, exc_info=True)
            self._stop_dot_animation()
            self._set_status(f"Erro: {e}", RED)
        finally:
            self._reativar_btn()

    # ── helpers de UI ─────────────────────────────────────────────────────────

    def _set_status(self, msg, color=None, arquivo=None):
        self._ultimo_arquivo = arquivo
        cursor = "hand2" if arquivo else "arrow"
        fg     = color or TEXT_DIM
        font   = ctk.CTkFont(size=12, underline=True) if arquivo else ctk.CTkFont(size=12)
        self.root.after(0, lambda: self.lbl_status.configure(
            text=msg, text_color=fg, cursor=cursor, font=font))

    def _abrir_arquivo(self, _e=None):
        if self._ultimo_arquivo and Path(self._ultimo_arquivo).exists():
            os.startfile(self._ultimo_arquivo)

    def _reativar_btn(self):
        self.root.after(0, lambda: self.btn.configure(
            state="normal", fg_color=RED, hover_color="#c03030",
            text_color=("white", "white"), text="● Gravar"
        ))

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
        crash_log = Path.home() / "Documents" / "MemoryMeet" / "crash.log"
        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            traceback.print_exc(file=f)
        raise
