# BACKLOG — MemoryMeet

Itens pendentes e ideias futuras. O mais maduro fica no topo.

---

## 🟡 Migrar transcrição+diarização pra local (faster-whisper + pyannote.audio) — registrado 01/07/2026, implementação iniciada 03/07/2026

**Status (03/07/2026):** camada `transcribers/` implementada e validada ponta a ponta — `TRANSCRIBER=openai|whisperx` no `.env`, WhisperX rodando 100% local (venv Python 3.12 dedicado, ver README). Testado com gravação real: transcrição correta, diarização funcionando com rótulo genérico (`SPEAKER_00`). **Pendente:** mapear `SPEAKER_00`/`01` pro nome real (Raffa/Interlocutor) via embedding de voz das referências ancoradas — pyannote já devolve `speaker_embeddings` prontos (`DiarizationPipeline(..., return_embeddings=True)`), falta só o casamento por similaridade. Também pendente decidir se vale resolver o cold start de ~3min pra carregar os modelos (hoje eager no `__init__`).

**Justificativa do Raffa:** a API da OpenAI (`gpt-4o-transcribe-diarize`) está instável — **perdemos chunks inteiros em duas sessões diferentes** (hoje: chunk 3 da call ACT/BTG, timeout total após 2 retries; e a sessão da entrevista Nava/BMG também precisou de recuperação via reprocessamento do MP3). Além da instabilidade, um modelo local elimina custo por minuto.

**Contexto técnico (da investigação de 01/07/2026):** o gargalo de hoje foi causado por uma combinação de (a) o modelo da OpenAI dar timeout de 180s repetidamente em chunks de ~5-6 min, e (b) o `_chunk_loop` em `main.py` ser síncrono — o timer do próximo chunk só recomeça depois que o anterior termina de processar (com todos os retries), o que faz um chunk lento inflar o próximo (cascata: 5min → 14min → timeout total). Ver sessão `sessions/2026-07-01 - microfone code22, gargalo memorymeet e entrevista act-btg.md` no repo `projeto-carreira-2026` para o diagnóstico completo (inclui proposta de fix incremental: paralelizar `_processar_chunk` e/ou reduzir `CHUNK_SEGUNDOS`, discussão ainda pendente).

**Caminho de migração (levantado em conversa, não validado ainda):**
- **faster-whisper** (ou `openai-whisper`) para transcrição local — CPU ou GPU, sem custo por minuto.
- **pyannote.audio** para diarização — grátis, mas os modelos pretrained são "gated" no Hugging Face (aceitar licença + gerar token, sem custo).
- **WhisperX** empacota os dois com alinhamento palavra-a-palavra — é o caminho mais direto pra reproduzir o que o app faz hoje, 100% local.

**Trade-offs a validar antes de migrar:**
- **Sem GPU dedicada neste notebook** (só Intel Graphics integrado, sem `nvidia-smi` — checado em 01/07/2026). `faster-whisper` roda razoável em CPU (modelo pequeno/médio + int8); `pyannote.audio` em CPU é mais lento que com GPU. **Não é bloqueio real:** a arquitetura já processa em chunks com antecedência (não é tempo real hoje, mesmo com a API da OpenAI), então o que importa é só não acumular atraso indefinidamente — mesmo critério que já vale pro bug do `_chunk_loop` síncrono acima. Raffa está otimista que CPU dá conta nesse regime.
- A diarização por pyannote tende a ser um pouco menos estável em trocas rápidas de falante do que a abordagem atual (duas referências de voz ancoradas por canal, mic/sistema) — pode precisar adaptar a lógica de referência ancorada pro pyannote, não só trocar o modelo.
- Não elimina o bug do `_chunk_loop` síncrono por si só — são dois problemas distintos (estabilidade da API vs. arquitetura de chunking); resolver um não resolve o outro automaticamente.

---

## ✅ FEITO — Diarização (identificar quem fala) — 30/06/2026

Implementado e **provado funcionando** end-to-end. O TXT agora sai com os falantes separados e nomeados:

```
[Interlocutor] Estou querendo agora que o Flávio Bolsonaro responda...

[Raffa] O crime que lança aí do PT, e aí, cara, como é que pode...

[Interlocutor] ...
```

### Arquitetura final
- **Modelo:** `gpt-4o-transcribe-diarize` (`response_format="diarized_json"`, `chunking_strategy="auto"`), rodando no áudio **mixado**.
- **Duas referências ancoradas, uma por canal** (o app já captura mic e sistema separados):
  - `mic_frames` → `known_speaker_names=["Raffa"]`
  - `sys_frames` (loopback) → `["Interlocutor"]`
- Cada referência é montada no 1º chunk em que o canal tem fala suficiente (gate de energia, `REF_MIN_ENERGIA`), pegando a janela de **maior energia** (`best_speech_window`), e **reusada nos chunks seguintes** — estabiliza os rótulos (o diarize renumera speakers por chamada).
- `format_segments` monta o TXT mesclando turnos consecutivos do mesmo falante.

### Bugs vencidos no caminho (cada um valeu uma lição)
1. **`400: Part exceeded maximum size of 1024KB`** — a referência ia como WAV cru (8s estéreo/48kHz ≈ 1.5MB). **Fix:** mandar a referência como **MP3** (~128KB). O limite de 1024KB é só das `known_speaker_references`; o arquivo principal aceita até 25MB (testado com 4.83MB → chunk de 5 min é seguro).
2. **Zíper / "absolute cinema"** — TXT com as falas picadas e intercaladas. Causa-raiz: **o loopback WASAPI Bluetooth não entrega frames durante o silêncio** (o BT suspende o stream). O canal do sistema ficava ~32% mais curto que o mic (1827 vs 2693 frames), o `mix_frames` casa por índice e sobrepunha as falas. **NÃO era bug de diarização nem do mix em si** — era dessincronia de captura. **Fix: keep-alive** (`_keep_output_alive`) — toca silêncio inaudível contínuo na saída pro BT nunca suspender. Depois do fix: 1457 vs 1444 frames (0,3% de diferença) e saída limpa.

### Pegadinhas / regras fixas
- **Exige fone.** Com caixa de som o mic recaptura o lado remoto e contamina a referência.
- O bug do zíper **só aparece com silêncio na saída** (ex: pausar um vídeo). **Call ao vivo mantém o stream ativo** e não dispara — mas o keep-alive garante em qualquer cenário.
- `gpt-4o-transcribe-diarize` não aceita `timestamp_granularities` nem prompt.

### Ainda não exercitado / a vigiar
- **Multi-chunk real:** a reuse de referência entre chunks de 5 min só roda de verdade numa gravação longa (testes foram de 1 chunk). Risco baixo.
- **5% dos casos (3+ pessoas do mesmo lado):** todas viram um `[Interlocutor]` só. Aceitável; pra separá-las precisaria de diarize só no canal do sistema.

---

## 💭 Alternativa de design considerada — canais separados (mais simples, adiada)

Em vez de mixar + diarize + referências, dava pra **transcrever cada canal separado** (mic=Raffa, sistema=Interlocutor) e juntar por timestamp. Vantagens: atribuição **física** (não inferida), sem referências, sem o modelo de diarize, sem limite de 1024KB. Mas **também depende do fix de sincronia** (o canal do sistema comprimido desalinha o merge igual). Como o caminho atual (mix+diarize) já está provado e funcionando, ficou adiada — vale revisitar se quisermos simplificar.

---

## 💡 Futuro — Nomes reais dos interlocutores via Google Agenda

Integrar com o Google Calendar pra puxar os participantes do evento e mapear `[Interlocutor]` para o nome real do convidado. Evolução: biblioteca de voiceprints em `APP_DIR` — uma vez identificado "Bruno", reconhecê-lo em calls futuras.

---

## 📌 Itens antigos (da v2, sessão de nascimento 19/06/2026)

- **Instância única** — se o app já estiver aberto, trazer a janela pra frente em vez de abrir outra.
- **Distribuição via `.exe`** (PyInstaller) — rodar sem Python instalado.
