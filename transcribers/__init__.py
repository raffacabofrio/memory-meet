import os


def get_transcriber():
    """Lê TRANSCRIBER do .env (default: openai) e devolve a implementação correspondente.
       Import lazy dentro de cada branch: quem usa 'openai' nunca carrega dependências
       de outros transcribers (ex.: torch/whisperx)."""
    nome = os.getenv("TRANSCRIBER", "openai").lower()

    if nome == "openai":
        from .openai_transcriber import OpenAITranscriber
        return OpenAITranscriber()

    if nome == "whisperx":
        from .whisperx_transcriber import WhisperXTranscriber
        return WhisperXTranscriber()

    raise ValueError(f"TRANSCRIBER desconhecido: {nome!r}")
