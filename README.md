# MemoryMeet

<p align="center">
  <img src="assets/logo.png" alt="MemoryMeet" width="280"/>
</p>

Memory fades. Conversations matter.

MemoryMeet is a lightweight, open-source meeting recorder and transcription tool that helps you preserve discussions exactly as they happened.

No subscriptions. No time limits. No distractions.

**Your meetings. Your memory.**

---

## Features

- Records microphone and system audio simultaneously
- Transcribes automatically using OpenAI Whisper
- Processes audio progressively in the background — no waiting at the end
- Saves MP3 + TXT to `~/Documents/MemoryMeet/`

## Requirements

- Windows 10/11
- Python 3.10+
- An OpenAI API key (default backend) — or nothing at all, if you use the local WhisperX backend

## Setup

```bash
git clone https://github.com/raffacabofrio/memory-meet.git
cd memory-meet
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
python main.py
```

## Transcription backend

Transcription + diarization is a pluggable layer (`transcribers/`). Pick one via `.env`:

```
TRANSCRIBER=openai      # default — calls OpenAI's gpt-4o-transcribe-diarize API, costs per minute
TRANSCRIBER=whisperx    # 100% local — faster-whisper + wav2vec2 + pyannote, no API calls, no cost
```

### WhisperX (local) setup

Runs entirely on your CPU — no audio ever leaves your machine, no per-minute cost. Requires **Python 3.12**
specifically: `whisperx`'s dependency chain (`faster-whisper`/`ctranslate2`) doesn't have wheels for very new
Python versions yet (checked against 3.14 on 03/07/2026). If your global Python is newer, create a dedicated venv:

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-whisperx.txt
.venv\Scripts\python main.py
```

The diarization model is gated on Hugging Face — one-time setup:

1. Create a Hugging Face account: https://huggingface.co/join
2. Accept the license at https://hf.co/pyannote/speaker-diarization-community-1 ("Agree and access repository")
   — if a dependent model also comes back gated, accept that one too the same way
3. Generate a read token: https://hf.co/settings/tokens
4. Add to `.env`: `HF_TOKEN=hf_...`

Optional tuning in `.env` (defaults shown):

```
WHISPERX_MODEL=small          # tiny|base|small|medium|large-v2 — bigger = slower on CPU, more accurate
WHISPERX_COMPUTE_TYPE=int8    # int8 is the CPU-friendly choice
WHISPERX_LANGUAGE=pt
```

First launch downloads the model weights (~1-2GB, one-time, cached afterward) and takes noticeably longer to
start than the OpenAI backend, since models load eagerly at startup rather than on first recording.

Known limitation: speaker labels currently come out generic (`SPEAKER_00`, `SPEAKER_01`...) instead of the real
names (`Raffa`/`Interlocutor`) — mapping them via voice-reference embeddings is a follow-up, not yet implemented.

## Usage

1. Click **Gravar** to start recording
2. Click **Parar** when done
3. Click the file link to open the transcript

## License

MIT
