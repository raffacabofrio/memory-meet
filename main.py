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
CHUNK_SEGUNDOS = 5 * 60  # processa um pedaço a cada 5 min durante a gravação


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

        icon = Path(__file__).parent / "assets" / "MemoryMeet.ico"
        if icon.exists():
            self.root.iconbitmap(str(icon))

        self.gravando = False
        self.stop_event = threading.Event()
        self.start_time = None
        self.timer_job = None
        self._thread_mic = None
        self._thread_sys = None
        self._ultimo_arquivo = None

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
            bg="#1e1e1e", fg="#888888", cursor="arrow"
        )
        self.lbl_status.pack(pady=(4, 0))
        self.lbl_status.bind("<Button-1>", self._abrir_arquivo)

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
        self._mp3_chunks = []
        self._transcricoes = []
        self._chunk_index = 0
        self._ultimo_arquivo = None

        self.btn.config(text="■ Parar", bg="#555555")
        self.lbl_status.config(text="Gravando...", fg="#e74c3c", cursor="arrow")

        self.start_time = time.time()
        self._tick()

        logging.info("Gravação iniciada")
        self._thread_mic = threading.Thread(target=self._record_mic, daemon=True)
        self._thread_sys = threading.Thread(target=self._record_system, daemon=True)
        self._thread_mic.start()
        self._thread_sys.start()
        threading.Thread(target=self._chunk_loop, daemon=True).start()

    def parar(self):
        self.gravando = False
        self.stop_event.set()
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.btn.config(state="disabled", text="● Gravar", bg="#c0392b")
        self.lbl_status.config(text="Finalizando...", fg="#f39c12", cursor="arrow")

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

    def _swap_frames(self):
        # Troca atômica — recording threads passam a escrever na nova lista vazia
        mic = self.mic_frames
        sys = self.sys_frames
        self.mic_frames = []
        self.sys_frames = []
        return mic, sys

    def _chunk_loop(self):
        while True:
            parou = self.stop_event.wait(timeout=CHUNK_SEGUNDOS)

            if parou:
                # Aguarda threads de gravação terminarem antes do chunk final
                for t in (self._thread_mic, self._thread_sys):
                    if t:
                        t.join(timeout=10)

            mic, sys = self._swap_frames()

            if mic or sys:
                self._chunk_index += 1
                label = "final" if parou else f"{self._chunk_index}"
                self._processar_chunk(mic, sys, label)

            if parou:
                break

        self._finalizar()

    def _processar_chunk(self, mic, sys, label):
        try:
            if mic and sys:
                audio = mix_frames(mic, sys)
            elif mic:
                audio = np.concatenate(mic)
            else:
                audio = np.concatenate(sys)

            logging.info("Chunk %s — mixing. Samples: %d", label, len(audio))
            mp3 = audio_para_mp3(audio, self.rate)
            self._mp3_chunks.append(mp3)
            logging.info("Chunk %s — MP3 pronto (%.1f MB)", label, len(mp3) / 1024 / 1024)

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return

            client = OpenAI(api_key=api_key)
            resultado = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("audio.mp3", io.BytesIO(mp3)),
                timeout=120,
            )
            self._transcricoes.append(resultado.text)
            logging.info("Chunk %s — transcrito. Chars: %d", label, len(resultado.text))

        except Exception as e:
            logging.error("Erro no chunk %s: %s", label, e, exc_info=True)

    def _finalizar(self):
        try:
            if not self._mp3_chunks:
                self._set_status("Nenhum áudio capturado.", "#e74c3c")
                self._reativar_btn()
                return

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename_base = str(APP_DIR / f"meet_{timestamp}")

            mp3_file = filename_base + ".mp3"
            with open(mp3_file, "wb") as f:
                for chunk in self._mp3_chunks:
                    f.write(chunk)
            logging.info("MP3 salvo: %s", mp3_file)

            if self._transcricoes:
                txt_file = filename_base + ".txt"
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(self._transcricoes))
                logging.info("Transcrição salva: %s", txt_file)
                self._set_status(f"Salvo: {Path(txt_file).name}", "#2ecc71", arquivo=txt_file)
            else:
                self._set_status(f"MP3 salvo (sem transcrição): {Path(mp3_file).name}", "#2ecc71", arquivo=mp3_file)

        except Exception as e:
            logging.error("Erro em _finalizar: %s", e, exc_info=True)
            self._set_status(f"Erro: {e}", "#e74c3c")
        finally:
            self._reativar_btn()

    def _set_status(self, msg, color="#888888", arquivo=None):
        self._ultimo_arquivo = arquivo
        cursor = "hand2" if arquivo else "arrow"
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color, cursor=cursor))

    def _abrir_arquivo(self, _event=None):
        if self._ultimo_arquivo and Path(self._ultimo_arquivo).exists():
            os.startfile(self._ultimo_arquivo)

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
