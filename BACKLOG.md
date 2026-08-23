# BACKLOG — MemoryMeet

Itens pendentes e ideias futuras. O mais maduro fica no topo.

---

## ✅ FEITO — MP3 do último chunk podia se perder se o app fechasse durante a transcrição — 23/07/2026

**Bug:** `_gravar_resultado` só escrevia o MP3 (e o TXT) quando o `ChunkResult` completo do chunk voltava do worker — ou seja, só depois que a transcrição inteira terminava. Numa gravação real, o app foi fechado com o último chunk (~6,5 min, já cortado e mixado — visto no log: "Chunk 12 — MP3 4.5 MB" bem antes do fechamento) ainda transcrevendo. Como a thread do worker morreu junto com o processo, `_gravar_resultado` nunca rodou pra esse chunk, e o áudio dele — que já existia pronto em memória havia tempo — nunca foi pro MP3 final. Confirmado batendo o timestamp do MP3 consolidado com o log: batia exatamente com o fim da transcrição do chunk *anterior*.

**Causa raiz:** acoplamento desnecessário entre a gravação do áudio e a gravação do texto — os dois só aconteciam juntos, gatilhados pelo fim da transcrição, embora o áudio esteja pronto muito antes (assim que o cortador corta e o worker mixa/codifica, antes mesmo de chamar o transcriber).

**Fix:** o MP3 agora é mixado e gravado em disco pelo **cortador** (`_cutter_loop` → `_concat_e_gravar_audio`), na hora em que o chunk é cortado — antes de ser enfileirado pra transcrição. Desde a otimização local de 21/08, `ChunkJob` voltou a carregar `mic_audio` e `sys_audio` separados para o worker; a mistura existe apenas para gerar o MP3 consolidado. `ChunkResult` não carrega MP3 e `_gravar_resultado` no orquestrador ficou só com o TXT, que continua reordenado por índice como antes. Efeito: se o app fechar/crashar no meio da transcrição de qualquer chunk (inclusive o "final", o buffer parcial ao apertar Parar — mesmo code path), o áudio dele já está seguro em disco; só o texto desse trecho específico fica faltando, que é uma falha bem menor (dá pra recuperar reprocessando o trecho do MP3, ver `skill-interview-feedback.md` no repo `projeto-carreira-2026`).

Sem lock novo: o cortador é thread única e sequencial, então é o único escritor do MP3 (o orquestrador, que também é único, ficou só com o TXT — arquivos diferentes, sem race).

**Verificação:** sem hardware real (mic/loopback), então testado com simulação isolada do pipeline (`ChunkJob`/`ChunkResult`/`processar_chunk` reais importados de `main.py`, frames e transcriber fake) cobrindo (a) chunk final "trava" antes do worker retornar — MP3 sobrevive intacto, TXT não ganha o texto dele; (b) transcrição falha com exceção tratada — mesma garantia. Não rodado end-to-end com gravação real.

---

## ✅ FEITO — Pipeline de chunks paralelo (cortador → workers puros → orquestrador) — 06/07/2026

Resolvido o bug do `_chunk_loop` síncrono (diagnóstico de 01/07, sentido na prática em 06/07: entrevista We Are Meta com "Finalizando" de ~18 min porque os chunks cresceram 5→9→14→18 min em cascata).

### Arquitetura
- **Cortador** (`_cutter_loop`): corta a cada `CHUNK_SEGUNDOS` fixos, concatena cada canal separadamente, mixa uma cópia e **já grava o MP3 em disco na hora** (`_concat_e_gravar_audio`, ver fix de 23/07/2026 acima). Depois enfileira `ChunkJob` imutável com `mic_audio` + `sys_audio` + snapshot das referências de voz. Nunca espera transcrição — chunk é sempre ~5 min, e o áudio nunca fica só em memória esperando o worker.
- **Worker** (`_worker_loop` → `processar_chunk`): função **pura** — só transcrição. No WhisperX, transcreve os dois canais isoladamente, rotula pela origem física e intercala por timestamp; no OpenAI, mixa os canais e usa a diarização da API. Zero side effects (sem arquivo, sem UI, sem estado compartilhado).
- **Orquestrador** (`_orchestrator_loop`): consumidor único do resultado da transcrição. Reordena por índice e só ele escreve o TXT (ordem garantida) e atualiza a UI.
- **Feedback sutil na UI:** "Gravando... · transcrevendo 2 de 3" durante a call; "Finalizando · trecho 4 de 4" no fim. Adeus spinner cego.

### Regra do WORKERS (decidido analisando o hardware em 06/07/2026)
`MEMORYMEET_WORKERS` no `.env`, **default 1 — não subir neste notebook**: o ctranslate2 já paraleliza por dentro (~3,5 dos 8 cores do Ultra 5 115U) e os workers compartilham uma única instância do modelo. A remoção de pyannote reduziu muito o peso do pipeline, mas N>1 continua sem validação de concorrência e sem necessidade prática neste hardware.

---

## ✅ FEITO — Transcrição local rápida com identificação física por canal — 03/07 a 23/08/2026

**Estado final:** camada `transcribers/` plugável e validada ponta a ponta — `TRANSCRIBER=openai|whisperx` no `.env`. O backend padrão roda 100% local em Python 3.12, sem custo por minuto, sem enviar áudio e sem depender de token/modelo gated do Hugging Face.

O primeiro caminho local reproduzia integralmente o pipeline de diarização: faster-whisper → alinhamento wav2vec2 → pyannote → embeddings de referência para converter `SPEAKER_XX` em nomes. Funcionava, mas as métricas adicionadas em 24/07 mostraram que estávamos pagando para inferir uma informação já conhecida pela captura. Numa gravação real de 21/08, um chunk cheio de 5 minutos levou em média **50,4s para transcrever, 67,1s para alinhar e 191,3s para diarizar** — 308,8s no total, ligeiramente mais que a duração do próprio chunk.

**Virada de 21/08:** o WhisperX local passou a receber `mic_audio` e `sys_audio` separados, transcrever cada canal isoladamente, atribuir os nomes configurados pela origem física (`mic = Raffa`, `sistema = Interlocutor`) e intercalar os segmentos pelo `start` retornado pelo Whisper. Alinhamento, pyannote, embeddings, threshold de similaridade e referências de voz foram eliminados do backend local. O OpenAI continua mixando os canais e usando `gpt-4o-transcribe-diarize`, preservando a alternativa de diarização real na nuvem.

**Validação real em 23/08/2026:** Raffa testou o fluxo ponta a ponta e confirmou melhora brutal de performance, **sem efeitos colaterais observados** em qualidade da transcrição, atribuição de falantes, áudio salvo ou finalização. Migração encerrada.

**Por que a migração nasceu:** a API da OpenAI perdeu chunks inteiros em duas sessões diferentes por timeout, tinha custo por minuto e, combinada ao antigo `_chunk_loop` síncrono, produziu a cascata 5→9→14→18 minutos. O pipeline paralelo resolveu a cascata em 06/07; a transcrição local removeu instabilidade e custo; a separação física dos canais resolveu a performance local em 21/08.

**OpenVINO arquivado como carta na manga, não como pendência:** o Intel Core Ultra 5 115U tem iGPU/NPU, mas ctranslate2 usa CPU ou CUDA. Como o caminho atual em CPU ficou rápido e estável depois da remoção de alinhamento/pyannote, não há motivo para trocar de motor agora. OpenVINO só volta à mesa se surgir uma necessidade nova de precisão ou performance que a stack atual não atenda.

---

## ✅ FEITO — Diarização no backend OpenAI — 30/06/2026

Implementado e **provado funcionando** end-to-end. Este desenho continua disponível quando `TRANSCRIBER=openai`; o backend local não usa mais diarização e identifica o falante pelo canal físico. O TXT sai com os falantes separados e nomeados:

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

## ✅ ADOTADA — Canais separados no backend local — decidida 21/08, validada 23/08/2026

Nasceu como alternativa adiada porque o caminho mix+diarize já funcionava. As métricas de produção mostraram que alinhamento e diarização eram o gargalo dominante, então a alternativa virou a arquitetura final do WhisperX: **transcrever cada canal separado** (mic=Raffa, sistema=Interlocutor) e juntar por timestamp. Atribuição física, sem referências, pyannote, embeddings ou limite de 1024KB. Continua dependendo do keep-alive que mantém os canais sincronizados; esse fundamento já estava validado desde o bug do "zíper". O teste real confirmou o ganho brutal sem efeitos colaterais observados.

---

## 💡 Futuro — Nomes reais dos interlocutores via Google Agenda

Integrar com o Google Calendar pra puxar os participantes do evento e mapear `[Interlocutor]` para o nome real do convidado. Evolução: biblioteca de voiceprints em `APP_DIR` — uma vez identificado "Bruno", reconhecê-lo em calls futuras.

---

## 💡 Futuro — Versão navegador (PWA em JS), zero instalação — emergida 09/07/2026

Ideia: uma versão do MemoryMeet que roda 100% no navegador, sem Python, sem setup. Bom pra "abre e grava" rápido, mantendo o app nativo como ferramenta séria (background, performance, entrevista longa).

### Cenário de uso pensado
Raffa abre a aba do MemoryMeet **antes** da agenda → o app pede o que capturar (`getDisplayMedia`) e ele escolhe **a aba do próprio Meet/Teams** com "compartilhar áudio da aba" (captura só a voz do outro lado, canal limpo, sem notificação/Spotify) → o mic dele é capturado em paralelo por `getUserMedia` em outro canal (mic não é exclusivo, não conflita com o Meet usando o mic) → faz a reunião → volta na aba e clica "finalizar".

### Peças e viabilidade
- **Mic (Raffa):** `getUserMedia({audio})` — trivial.
- **Outro lado (Interlocutor):** `getDisplayMedia({audio:true})` compartilhando a aba da reunião. **Chrome/Edge no Windows.**
- **Separação de speaker:** dois `MediaStream` = dois canais físicos → **rotular por canal e descartar o pyannote inteiro**. A peça sem porta boa em JS é exatamente a que não precisaríamos portar — mesma estratégia de canal físico agora validada no app nativo (mic=Raffa, loopback=Interlocutor).
- **Transcrição:** `transformers.js` (Xenova/whisper) via WASM ou **WebGPU**.
- **Saída MP3/TXT:** `lamejs`/MediaRecorder + File System Access API.

### O pivô do design — Web Worker (decisivo, 09/07/2026)
Design ingênuo (transcrever tudo no "finalizar") **trava** — parece que o app pendurou. Errado. Espelhar a arquitetura do app Python: **transcrição incremental num Web Worker** durante a gravação. Worker de background é estrangulado **bem menos** que a thread principal (o throttling de aba de fundo pega timers/rAF da main thread, não o Worker), então dá pra transcrever picado enquanto grava; no "finalizar" quase tudo já está pronto e o encerramento é rápido. **Sem o Worker, o produto não existe.**

### Os dois riscos reais (o resto é resolvível)
1. **Velocidade do WebGPU no Ultra 5:** na melhor hipótese ~1x tempo real (igual ao nativo). Se segurar 1x, o incremental acompanha e "finalizar" é instantâneo. Se ficar 2-3x mais lento, acumula atraso e a espera volta. **Empírico — só medindo.** Próximo passo concreto: página de teste que transcreve um áudio e cronometra WebGPU, **sem** captura de reunião, só pra ter o número antes de investir.
2. **Tarja de compartilhamento:** o Chrome mostra "está compartilhando esta aba" com botão "Parar" durante a call. Não tem como esconder em browser puro. Fricção genuína — é, sozinha, um bom motivo pra manter o nativo como ferramenta principal.

### Pegadinha de plataforma
Meet é sempre no navegador → share da aba funciona limpo. **Teams no app desktop** não dá pra compartilhar "a aba" → cairia em "tela inteira + áudio do sistema" (mais sujo, pega tudo). Teams **no navegador** funciona igual ao Meet.

---

## 📌 Itens antigos (da v2, sessão de nascimento 19/06/2026)

- **Instância única** — se o app já estiver aberto, trazer a janela pra frente em vez de abrir outra.
- **Distribuição via `.exe`** (PyInstaller) — rodar sem Python instalado.
