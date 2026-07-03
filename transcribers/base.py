from abc import ABC, abstractmethod
from dataclasses import dataclass

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


class Transcriber(ABC):
    @abstractmethod
    def transcribe_and_diarize(self, audio: np.ndarray, rate: int, refs: list[SpeakerRef]) -> list[Segment]:
        """Transcreve e diariza um chunk de áudio mixado (mic+sistema).
           refs: referências de voz já ancoradas (nome + clip), uma por canal identificado
           até agora — pode vir vazia se nenhum canal teve fala suficiente ainda."""
        ...
