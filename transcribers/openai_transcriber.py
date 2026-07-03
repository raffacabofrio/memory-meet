import base64
import io
import os

import lameenc
from openai import OpenAI

from .base import Segment, SpeakerRef, Transcriber

TRANSCRIBE_MODEL = "gpt-4o-transcribe-diarize"
MP3_BITRATE = 128
CHANNELS = 2


def _audio_para_mp3(audio, rate):
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(rate)
    enc.set_channels(CHANNELS)
    enc.set_quality(2)
    return enc.encode(audio.tobytes()) + enc.flush()


def _seg_attr(seg, key, default=""):
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)


def _ref_to_data_url(ref: SpeakerRef) -> str:
    mp3 = _audio_para_mp3(ref.audio, ref.rate)
    return "data:audio/mp3;base64," + base64.b64encode(mp3).decode("ascii")


class OpenAITranscriber(Transcriber):
    """Usa known_speaker_references da API pra ancorar os nomes — o server faz o
       casamento de voz por conta própria, não precisamos de nenhuma lógica de
       identificação aqui."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def transcribe_and_diarize(self, audio, rate, refs):
        if not self.api_key:
            return []

        mp3 = _audio_para_mp3(audio, rate)

        extra = {}
        if refs:
            extra["known_speaker_names"] = [r.name for r in refs]
            extra["known_speaker_references"] = [_ref_to_data_url(r) for r in refs]

        resultado = OpenAI(api_key=self.api_key).audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=("audio.mp3", io.BytesIO(mp3)),
            response_format="diarized_json",
            chunking_strategy="auto",
            extra_body=extra,
            timeout=180,
        )
        segments = getattr(resultado, "segments", None) or []
        return [
            Segment(speaker=_seg_attr(s, "speaker", "") or "SPEAKER",
                    text=_seg_attr(s, "text", "") or "")
            for s in segments
        ]
