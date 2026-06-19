import tkinter as tk
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

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
MP3_BITRATE = 128
CHUNK_MINUTOS = 20  # ~18 MB em MP3 128kbps — seguro abaixo do limite de 25 MB da API


def mix_frames(frames_a, frames_b):
    min_len = min(len(frames_a), len(frames_b))
    if min_len == 0:
        return np.concatenate(frames_a) if frames_a else np.concatenate(frames_b)
    a = np.concatenate(frames_a[:min_len]).astype(np.int32)
    b = np.concatenate(frames_b[:min_len]).astype(np.int32)
    min_size = min(len(a), len(b))
    return np.clip((a[:min_size] + b[:min_size]) // 2, -32768, 32767).astype(np.int16)


def audio_para_mp3(audio, rate):
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(CHANNELS)
    encoder.set_quality(2)
    mp3 = encoder.encode(audio.tobytes())
    mp3 += encoder.flush()
    return mp3


class MemoryMeet:
    def __init__(self, root):
        self.root = root
        self.root.title("MemoryMeet")
        self.root.resizable(False, False)
        self.root.geometry("300x200")

        self.gravando = False
        self.stop_event = threading.Event()
        self.start_time = None
        self.timer_job = None
        self._thread_mic = None
        self._thread_sys = None

        self.p = pyaudio.PyAudio()
        try:
            self.loopback = self.p.get_default_wasapi_loopback()
            self.rate = int(self.loopback["defaultSampleRate"])
            self.loop_channels = min(int(self.loopback["maxInputChannels"]), 2)
            self.sample_width = self.p.get_sample_size(FORMAT)
            logging.info("Inicializado. Loopback: %s | rate: %d", self.loopback["name"], self.rate)
        except Exception as e:
            logging.error("Falha ao inicializar loopback: %s", e)
            self.loopback = None

        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg="#1e1e1e")

        self.lbl_timer = tk.Label(
            self.root, text="00:00", font=("Courier", 48, "bold"),
            bg="#1e1e1e", fg="#ffffff"
        )
        self.lbl_timer.pack(pady=(20, 0))

        self.lbl_status = tk.Label(
            self.root, text="Pronto", font=("Segoe UI", 11),
            bg="#1e1e1e", fg="#888888"
        )
        self.lbl_status.pack(pady=(4, 0))

        self.btn = tk.Button(
            self.root, text="● Gravar", font=("Segoe UI", 13, "bold"),
            bg="#c0392b", fg="white", relief="flat", cursor="hand2",
            padx=20, pady=8, command=self.toggle
        )
        self.btn.pack(pady=(20, 0))

        if self.loopback is None:
            self.lbl_status.config(text="Erro: loopback não encontrado", fg="#e74c3c")
            self.btn.config(state="disabled")

    def toggle(self):
        if self.gravando:
            self.parar()
        else:
            self.iniciar()

    def iniciar(self):
        self.gravando = True
        self.stop_event.clear()
        self.mic_frames = []
        self.sys_frames = []

        self.btn.config(text="■ Parar", bg="#555555")
        self.lbl_status.config(text="Gravando...", fg="#e74c3c")

        self.start_time = time.time()
        self._tick()

        logging.info("Gravação iniciada")
        self._thread_mic = threading.Thread(target=self._record_mic, daemon=True)
        self._thread_sys = threading.Thread(target=self._record_system, daemon=True)
        self._thread_mic.start()
        self._thread_sys.start()

    def parar(self):
        self.gravando = False
        self.stop_event.set()
        if self.timer_job:
            self.root.after_cancel(self.timer_job)

        self.btn.config(state="disabled", text="● Gravar", bg="#c0392b")
        self.lbl_status.config(text="Processando...", fg="#f39c12")

        threading.Thread(target=self._processar, daemon=True).start()

    def _tick(self):
        if self.gravando:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            self.lbl_timer.config(text=f"{m:02d}:{s:02d}")
            self.timer_job = self.root.after(1000, self._tick)

    def _record_mic(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=self.rate,
                                 input=True, frames_per_buffer=CHUNK)
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.mic_frames.append(np.frombuffer(data, dtype=np.int16).copy())
            stream.stop_stream()
            stream.close()
            logging.info("Mic encerrado. Frames: %d", len(self.mic_frames))
        except Exception as e:
            logging.error("Erro mic: %s", e, exc_info=True)
            self._set_status(f"Erro mic: {e}", "#e74c3c")

    def _record_system(self):
        try:
            stream = self.p.open(format=FORMAT, channels=self.loop_channels, rate=self.rate,
                                 input=True, input_device_index=self.loopback["index"],
                                 frames_per_buffer=CHUNK)
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).copy()
                if self.loop_channels == 1:
                    arr = np.repeat(arr, 2)
                self.sys_frames.append(arr)
            stream.stop_stream()
            stream.close()
            logging.info("Sistema encerrado. Frames: %d", len(self.sys_frames))
        except Exception as e:
            logging.error("Erro sistema: %s", e, exc_info=True)
            self._set_status(f"Erro sistema: {e}", "#e74c3c")

    def _processar(self):
        for t in (self._thread_mic, self._thread_sys):
            if t:
                t.join(timeout=10)

        try:
            if not self.mic_frames and not self.sys_frames:
                self._set_status("Nenhum áudio capturado.", "#e74c3c")
                self._reativar_btn()
                return

            if self.mic_frames and self.sys_frames:
                audio = mix_frames(self.mic_frames, self.sys_frames)
            elif self.mic_frames:
                audio = np.concatenate(self.mic_frames)
            else:
                audio = np.concatenate(self.sys_frames)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename_base = str(APP_DIR / f"meet_{timestamp}")

            self._set_status("Convertendo para MP3...", "#f39c12")
            logging.info("Convertendo para MP3. Samples: %d", len(audio))
            mp3 = audio_para_mp3(audio, self.rate)
            mp3_file = filename_base + ".mp3"
            with open(mp3_file, "wb") as f:
                f.write(mp3)
            logging.info("MP3 salvo: %s (%.1f MB)", mp3_file, len(mp3) / 1024 / 1024)

            self._set_status("Transcrevendo...", "#f39c12")
            try:
                txt_file = self._transcrever(audio, filename_base)
                if txt_file:
                    logging.info("Transcrição concluída: %s", txt_file)
                    self._set_status(f"Salvo: {Path(txt_file).name}", "#2ecc71")
                else:
                    self._set_status(f"MP3 salvo (sem transcrição): {Path(mp3_file).name}", "#2ecc71")
            except Exception as e:
                logging.error("Transcrição falhou: %s", e, exc_info=True)
                self._set_status(f"Transcrição falhou: {e}", "#e74c3c")

        except Exception as e:
            logging.error("Erro em _processar: %s", e, exc_info=True)
            self._set_status(f"Erro: {e}", "#e74c3c")
        finally:
            self._reativar_btn()

    def _transcrever(self, audio, filename_base):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        client = OpenAI(api_key=api_key)
        frames_por_chunk = self.rate * CHANNELS * 60 * CHUNK_MINUTOS
        chunks = [audio[i:i + frames_por_chunk] for i in range(0, len(audio), frames_por_chunk)]

        transcricoes = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                self._set_status(f"Transcrevendo {i+1}/{len(chunks)}...", "#f39c12")
            mp3 = audio_para_mp3(chunk, self.rate)
            logging.info("Enviando chunk %d/%d para API (%.1f MB MP3)", i + 1, len(chunks),
                         len(mp3) / 1024 / 1024)
            resultado = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("audio.mp3", io.BytesIO(mp3)),
                timeout=120,
            )
            logging.info("Chunk %d/%d transcrito. Chars: %d", i + 1, len(chunks), len(resultado.text))
            transcricoes.append(resultado.text)

        texto = "\n\n".join(transcricoes)
        txt_file = filename_base + ".txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(texto)
        return txt_file

    def _set_status(self, msg, color="#888888"):
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color))

    def _reativar_btn(self):
        self.root.after(0, lambda: self.btn.config(state="normal"))

    def fechar(self):
        self.stop_event.set()
        self.p.terminate()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MemoryMeet(root)
    root.protocol("WM_DELETE_WINDOW", app.fechar)
    root.mainloop()
