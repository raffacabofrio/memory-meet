import logging
import os
import time
from math import gcd
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from .base import Segment, Transcriber

SAMPLE_RATE = 16000  # whisperx.audio.SAMPLE_RATE — todo o pipeline espera mono float32 nessa taxa


def _to_whisperx_audio(audio: np.ndarray, rate: int) -> np.ndarray:
    """int16 estéreo interleaved (rate Hz) -> mono float32 16kHz. Feito com resample_poly em vez
       de ffmpeg: o áudio já está em memória como PCM cru, não precisa de arquivo/decoder externo."""
    stereo = audio.reshape(-1, 2).astype(np.float32) / 32768.0
    mono = stereo.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, rate // g)
    return mono.astype(np.float32)


class WhisperXTranscriber(Transcriber):
    """100% local: só faster-whisper (via whisperx) — sem alinhamento (wav2vec2) e sem
       diarização (pyannote). Modelo carregado uma vez aqui no __init__ e reusado por chunk.

       Antes, o áudio ia mixado (mic+sistema) pra um pipeline de diarizar (identificar quem
       fala por voz) e depois mapear pro nome real por similaridade com as referências
       ancoradas. Medido em produção (14/08/2026): alinhamento + diarização somavam bem mais
       tempo que a própria transcrição (ex.: 130s transcrição vs 163s + 515s dos outros dois
       estágios, num chunk de 5 min) — o gargalo real não era o modelo de fala, era esse par.

       MemoryMeet já sabe quem é quem sem precisar inferir nada: mic = você, loopback do
       sistema = o outro lado, dois canais físicos capturados separadamente. Então cada canal
       é transcrito isoladamente e rotulado direto pelo canal — elimina os dois estágios caros
       por completo, sem trocar a precisão da atribuição (é física, não estimada por voz).

       Ordem dos textos: cada segmento carrega o `start` (segundos desde o início do chunk)
       devolvido pelo próprio whisper. Os segmentos dos dois canais são intercalados por esse
       `start` antes de formatar (`transcribe_and_diarize` devolve a lista já mesclada) — nunca
       "tudo do mic, depois tudo do sistema". Isso depende dos dois canais começarem no mesmo
       instante do chunk, o que já é garantido pelo keep-alive de captura (ver histórico do
       bug do "zíper" no BACKLOG) — sem ele, o `start` de cada canal não seria comparável."""

    def __init__(self):
        import whisperx

        self._whisperx = whisperx
        self.device = "cpu"
        self.language = os.getenv("WHISPERX_LANGUAGE", "pt")
        compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
        model_size = os.getenv("WHISPERX_MODEL", "small")
        hf_token = os.getenv("HF_TOKEN")

        self.model = whisperx.load_model(
            model_size, self.device, compute_type=compute_type,
            language=self.language, use_auth_token=hf_token,
        )

    def _transcribe_channel(self, audio: Optional[np.ndarray], rate: int, speaker_name: str):
        """Transcreve um canal isolado. Devolve [(start, Segment)] — vazio se o canal não
           teve áudio neste chunk ou não teve fala. `start` é a chave de ordenação usada em
           transcribe_and_diarize pra intercalar os dois canais na ordem real da fala."""
        if audio is None:
            return []
        wx_audio = _to_whisperx_audio(audio, rate)
        result = self.model.transcribe(wx_audio, batch_size=8, language=self.language)
        out = []
        for seg in result.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            out.append((seg.get("start", 0.0), Segment(speaker=speaker_name, text=text)))
        return out

    def transcribe_and_diarize(self, mic_audio, sys_audio, rate, mic_name, sys_name, refs):
        t0 = time.perf_counter()
        mic_segs = self._transcribe_channel(mic_audio, rate, mic_name)
        sys_segs = self._transcribe_channel(sys_audio, rate, sys_name)
        t_transcricao = time.perf_counter() - t0

        logging.info(
            "WhisperX — transcrição por canal: %.1fs (mic: %d segmentos, sistema: %d segmentos)",
            t_transcricao, len(mic_segs), len(sys_segs),
        )

        # Merge estável por start: preserva a ordem real da conversa mesmo com os dois lados
        # falando dentro do mesmo chunk de 5 min. sorted() é estável, então empates de start
        # (raro, mas possível) mantêm mic antes de sistema — decisão arbitrária, não crítica.
        merged = sorted(mic_segs + sys_segs, key=lambda item: item[0])
        return [seg for _, seg in merged]
