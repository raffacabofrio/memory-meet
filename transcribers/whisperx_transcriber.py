import logging
import os
import time
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .base import Segment, SpeakerRef, Transcriber

SAMPLE_RATE = 16000  # whisperx.audio.SAMPLE_RATE — todo o pipeline espera mono float32 nessa taxa

# Similaridade de cosseno mínima pra aceitar o match SPEAKER_XX -> nome real. Abaixo disso,
# melhor manter o rótulo genérico do que arriscar nomear errado (ver _map_speakers).
REF_SIM_THRESHOLD = float(os.getenv("WHISPERX_REF_SIM_THRESHOLD", "0.5"))


def _to_whisperx_audio(audio: np.ndarray, rate: int) -> np.ndarray:
    """int16 estéreo interleaved (rate Hz) -> mono float32 16kHz. Feito com resample_poly em vez
       de ffmpeg: o áudio já está em memória como PCM cru, não precisa de arquivo/decoder externo."""
    stereo = audio.reshape(-1, 2).astype(np.float32) / 32768.0
    mono = stereo.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, rate // g)
    return mono.astype(np.float32)


def _cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class WhisperXTranscriber(Transcriber):
    """100% local: faster-whisper (transcrição) + wav2vec2 (alinhamento) + pyannote (diarização),
       empacotados pelo whisperx. Modelos carregados uma vez aqui no __init__ e reusados por chunk.

       Diarização sai genérica (SPEAKER_00/SPEAKER_01...) do pyannote — pra virar nome real
       (Raffa/Interlocutor), usamos `return_embeddings=True` da DiarizationPipeline (embedding por
       SPEAKER_XX detectado, sem custo extra) e comparamos por similaridade de cosseno com o
       embedding de cada referência ancorada (`refs`), extraído rodando a mesma pipeline no clip
       de referência com `num_speakers=1`. Mesmo espaço vetorial dos dois lados, sem depender de
       modelo de embedding adicional."""

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

        # Embedding de referência é caro (roda a pipeline de diarize inteira) e as refs são
        # montadas uma vez em main.py e reusadas em todos os chunks — cachear por identidade
        # evita recalcular a cada chunk de 5 min.
        self._ref_embedding_cache: dict[int, list | None] = {}

    def _ref_embedding(self, ref: SpeakerRef):
        """Embedding do clip de referência (fala de um só canal, já filtrado pra janela de maior
           energia em main.py) — roda a mesma DiarizationPipeline com num_speakers=1 e pega o
           único embedding devolvido. None se a pipeline não conseguir extrair nada do clip."""
        cache_key = id(ref)
        if cache_key in self._ref_embedding_cache:
            return self._ref_embedding_cache[cache_key]

        emb = None
        try:
            wx_ref_audio = _to_whisperx_audio(ref.audio, ref.rate)
            _, embeddings = self.diarize_model(wx_ref_audio, num_speakers=1, return_embeddings=True)
            if embeddings:
                emb = next(iter(embeddings.values()))
        except Exception:
            logging.exception("Falha ao extrair embedding da referência [%s]", ref.name)

        self._ref_embedding_cache[cache_key] = emb
        return emb

    def _map_speakers(self, refs, speaker_embeddings):
        """SPEAKER_XX (pyannote) -> nome real (Raffa/Interlocutor) por similaridade de cosseno
           com as referências ancoradas. Sem refs ou sem embeddings, devolve mapeamento vazio —
           os rótulos genéricos seguem intactos (compatibilidade com o comportamento atual).
           Se a melhor similaridade ficar abaixo de REF_SIM_THRESHOLD, também não mapeia: é
           melhor manter SPEAKER_XX do que atribuir um nome errado com baixa confiança."""
        if not refs or not speaker_embeddings:
            return {}

        ref_embs = [(r.name, self._ref_embedding(r)) for r in refs]
        ref_embs = [(name, emb) for name, emb in ref_embs if emb is not None]
        if not ref_embs:
            return {}

        mapping = {}
        for spk, emb in speaker_embeddings.items():
            best_name, best_sim = None, -1.0
            for name, ref_emb in ref_embs:
                sim = _cosine_sim(emb, ref_emb)
                if sim > best_sim:
                    best_sim, best_name = sim, name
            if best_name is not None and best_sim >= REF_SIM_THRESHOLD:
                mapping[spk] = best_name
            else:
                logging.info("Speaker %s sem match confiável (melhor sim=%.2f) — mantendo rótulo genérico",
                             spk, best_sim)
        return mapping

    def transcribe_and_diarize(self, audio, rate, refs):
        wx_audio = _to_whisperx_audio(audio, rate)

        t0 = time.perf_counter()
        result = self.model.transcribe(wx_audio, batch_size=8, language=self.language)
        t_transcricao = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = self._whisperx.align(result["segments"], self.align_model, self.align_meta,
                                       wx_audio, self.device)
        t_alinhamento = time.perf_counter() - t0

        t0 = time.perf_counter()
        diarize_df, speaker_embeddings = self.diarize_model(wx_audio, return_embeddings=True)
        result = self._whisperx.assign_word_speakers(diarize_df, result)
        speaker_map = self._map_speakers(refs, speaker_embeddings)
        t_diarizacao = time.perf_counter() - t0

        logging.info(
            "WhisperX — transcrição: %.1fs | alinhamento: %.1fs | diarização: %.1fs",
            t_transcricao, t_alinhamento, t_diarizacao,
        )

        segments = []
        for seg in result.get("segments", []):
            raw_speaker = seg.get("speaker", "") or "SPEAKER"
            speaker = speaker_map.get(raw_speaker, raw_speaker)
            segments.append(Segment(speaker=speaker, text=(seg.get("text") or "")))
        return segments
