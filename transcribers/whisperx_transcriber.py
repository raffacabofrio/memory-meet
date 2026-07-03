import os
from math import gcd

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
    """100% local: faster-whisper (transcrição) + wav2vec2 (alinhamento) + pyannote (diarização),
       empacotados pelo whisperx. Modelos carregados uma vez aqui no __init__ e reusados por chunk.

       Diarização ainda é genérica (SPEAKER_00/SPEAKER_01...) — mapear esses rótulos pro nome real
       (Raffa/Interlocutor) a partir das referências ancoradas (`refs`) é o próximo passo, ainda não
       implementado: por ora o parâmetro é aceito pra cumprir o contrato mas não é usado."""

    def __init__(self):
        import whisperx
        from whisperx.diarize import DiarizationPipeline

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
        self.diarize_model = DiarizationPipeline(token=hf_token, device=self.device)
        self.align_model, self.align_meta = whisperx.load_align_model(
            language_code=self.language, device=self.device)

    def transcribe_and_diarize(self, audio, rate, refs):
        wx_audio = _to_whisperx_audio(audio, rate)

        result = self.model.transcribe(wx_audio, batch_size=8, language=self.language)
        result = self._whisperx.align(result["segments"], self.align_model, self.align_meta,
                                       wx_audio, self.device)

        diarize_df = self.diarize_model(wx_audio)
        result = self._whisperx.assign_word_speakers(diarize_df, result)

        return [
            Segment(speaker=seg.get("speaker", "") or "SPEAKER", text=(seg.get("text") or ""))
            for seg in result.get("segments", [])
        ]
