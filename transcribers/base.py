from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpeakerRef:
    """Referência de voz ancorada de um canal — clip cru (mesmo dtype/rate da captura)."""
    name: str
    audio: np.ndarray
    rate: int


@dataclass
class Segment:
    """Um turno de fala: quem falou e o que foi dito."""
    speaker: str
    text: str


def mix_audio(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Mixa dois arrays de áudio já concatenados (mesmo dtype/rate), alinhando pelo menor
       tamanho quando os dois existem. Usado só onde é preciso um único trecho combinado
       (MP3 final, ou um backend que diariza no áudio mixado) — a transcrição em si mantém
       os canais separados (ver Transcriber.transcribe_and_diarize)."""
    if a is None:
        return b
    if b is None:
        return a
    n = min(len(a), len(b))
    x = a[:n].astype(np.int32)
    y = b[:n].astype(np.int32)
    return np.clip((x + y) // 2, -32768, 32767).astype(np.int16)


class Transcriber(ABC):
    @abstractmethod
    def transcribe_and_diarize(
        self,
        mic_audio: Optional[np.ndarray],
        sys_audio: Optional[np.ndarray],
        rate: int,
        mic_name: str,
        sys_name: str,
        refs: list[SpeakerRef],
    ) -> list[Segment]:
        """Transcreve um chunk de áudio com os canais mic/sistema mantidos separados (não
           mixados) — cada implementação decide como usar essa separação física de falantes.
           mic_audio/sys_audio: np.ndarray (mesmo dtype/rate da captura) ou None se o canal
           não teve nenhum frame neste chunk. mic_name/sys_name: rótulos configurados (env),
           sempre disponíveis (não dependem de referência ancorada). refs: referências de
           voz já ancoradas (nome + clip) — usadas por backends que fazem diarização real
           (ex. OpenAI); pode vir vazia e/ou ser ignorada por quem já identifica o falante
           pelo canal físico (ex. WhisperX)."""
        ...
