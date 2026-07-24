# BACKLOG — MemoryMeet

Itens pendentes e ideias futuras. O mais maduro fica no topo.

---

## ✅ FEITO — MP3 do último chunk podia se perder se o app fechasse durante a transcrição — 23/07/2026

**Bug:** `_gravar_resultado` só escrevia o MP3 (e o TXT) quando o `ChunkResult` completo do chunk voltava do worker — ou seja, só depois que a transcrição inteira terminava. Numa gravação real, o app foi fechado com o último chunk (~6,5 min, já cortado e mixado — visto no log: "Chunk 12 — MP3 4.5 MB" bem antes do fechamento) ainda transcrevendo. Como a thread do worker morreu junto com o processo, `_gravar_resultado` nunca rodou pra esse chunk, e o áudio dele — que já existia pronto em memória havia tempo — nunca foi pro MP3 final. Confirmado batendo o timestamp do MP3 consolidado com o log: batia exatamente com o fim da transcrição do chunk *anterior*.

**Causa raiz:** acoplamento desnecessário entre a gravação do áudio e a gravação do texto — os dois só aconteciam juntos, gatilhados pelo fim da transcrição, embora o áudio esteja pronto muito antes (assim que o cortador corta e o worker mixa/codifica, antes mesmo de chamar o transcriber).

**Fix:** o MP3 agora é mixado e gravado em disco pelo **cortador** (`_cutter_loop` → novo `_mixar_e_gravar_audio`), na hora em que o chunk é cortado — antes de ser enfileirado pra transcrição. `ChunkJob` passou a carregar o áudio já mixado (em vez de `mic`/`sys` crus) e `ChunkResult` perdeu o campo `mp3` — o worker (`processar_chunk`) só transcreve. `_gravar_resultado` no orquestrador ficou só com o TXT, que continua reordenado por índice como antes. Efeito: se o app fechar/crashar no meio da transcrição de qualquer chunk (inclusive o "final", o buffer parcial ao apertar Parar — mesmo code path), o áudio dele já está seguro em disco; só o texto desse trecho específico fica faltando, que é uma falha bem menor (dá pra recuperar reprocessando o trecho do MP3, ver `skill-interview-feedback.md` no repo `projeto-carreira-2026`).

Sem lock novo: o cortador é thread única e sequencial, então é o único escritor do MP3 (o orquestrador, que também é único, ficou só com o TXT — arquivos diferentes, sem race).

**Verificação:** sem hardware real (mic/loopback), então testado com simulação isolada do pipeline (`ChunkJob`/`ChunkResult`/`processar_chunk` reais importados de `main.py`, frames e transcriber fake) cobrindo (a) chunk final "trava" antes do worker retornar — MP3 sobrevive intacto, TXT não ganha o texto dele; (b) transcrição falha com exceção tratada — mesma garantia. Não rodado end-to-end com gravação real.

---

## ✅ FEITO — Pipeline de chunks paralelo (cortador → workers puros → orquestrador) — 06/07/2026

Resolvido o bug do `_chunk_loop` síncrono (diagnóstico de 01/07, sentido na prática em 06/07: entrevista We Are Meta com "Finalizando" de ~18 min porque os chunks cresceram 5→9→14→18 min em cascata).

### Arquitetura
- **Cortador** (`_cutter_loop`): corta a cada `CHUNK_SEGUNDOS` fixos, mixa e **já grava o MP3 em disco na hora** (`_mixar_e_gravar_audio`, ver fix de 23/07/2026 acima), depois enfileira `ChunkJob` imutável (áudio já mixado + snapshot das referências de voz). Nunca espera transcrição — chunk é sempre ~5 min, e o áudio nunca fica só em memória esperando o worker.
- **Worker** (`_worker_loop` → `processar_chunk`): função **pura** — só transcrição. Zero side effects (sem arquivo, sem UI, sem estado compartilhado).
- **Orquestrador** (`_orchestrator_loop`): consumidor único do resultado da transcrição. Reordena por índice e só ele escreve o TXT (ordem garantida) e atualiza a UI.
- **Feedback sutil na UI:** "Gravando... · transcrevendo 2 de 3" durante a call; "Finalizando · trecho 4 de 4" no fim. Adeus spinner cego.

### Regra do WORKERS (decidido analisando o hardware em 06/07/2026)
`MEMORYMEET_WORKERS` no `.env`, **default 1 — não subir neste notebook**: o ctranslate2 já paraleliza por dentro (~3,5 dos 8 cores do Ultra 5 115U), não sobra RAM pra segundo modelo (stack usa 4,7 GB de 15,5) e o pipeline pyannote não é confiável pra chamadas concorrentes no mesmo modelo. N>1 só faz sentido com GPU ou um modelo por worker.

---

## 🟡 Migrar transcrição+diarização pra local (faster-whisper + pyannote.audio) — registrado 01/07/2026, implementação iniciada 03/07/2026

**Status (16/07/2026):** camada `transcribers/` implementada e validada ponta a ponta — `TRANSCRIBER=openai|whisperx` no `.env`, WhisperX rodando 100% local (venv Python 3.12 dedicado, ver README). **Mapeamento `SPEAKER_XX` -> nome real implementado** em `transcribers/whisperx_transcriber.py`: `DiarizationPipeline(..., return_embeddings=True)` devolve embedding por `SPEAKER_XX` do chunk; o embedding de cada referência ancorada (`refs`, já vinha de `main.py` mas era ignorado) é extraído rodando a mesma pipeline no clip com `num_speakers=1` (cacheado por referência — evita recalcular a cada chunk de 5 min); casamento por similaridade de cosseno, com `WHISPERX_REF_SIM_THRESHOLD` (default `0.5`, env-configurável) como piso de confiança — abaixo disso mantém o rótulo genérico em vez de arriscar nome errado. Sem `refs` (lista vazia/None), comportamento idêntico ao anterior (rótulos genéricos), sem quebrar. Validado: import limpo, smoke test da lógica de matching (`_cosine_sim`/`_map_speakers`) isolado sem baixar modelos. **Não validado ainda** com gravação real (pipeline completo baixa modelos pyannote gated na primeira execução — não rodado nesta sessão); threshold de `0.5` é um chute razoável, não calibrado com dados reais — ajustar se sair nome errado ou rótulo genérico demais na prática. **Ainda pendente:** decidir se vale resolver o cold start de ~3min pra carregar os modelos (hoje eager no `__init__`).

**Carta na manga — hardware real da máquina (checado 03/07/2026):** o notebook é um **Intel Core Ultra 5 115U** (Meteor Lake) — tem GPU integrada (Intel Graphics Xe-LPG) e até um **NPU** dedicado pra IA. Não é "sem hardware de IA", é que a stack atual não sabe usar esse hardware: `ctranslate2` (motor do `faster-whisper`) só acelera em CPU ou GPU NVIDIA/CUDA; `whisperx` crava `device="cpu"`/`"cuda"` no código, sem caminho pra iGPU/NPU Intel.

Se a família de modelos atual (`tiny`→`large-v3`/`turbo`, todos via `ctranslate2`/CPU) não entregar velocidade ou precisão suficiente, o próximo lugar pra fazer discovery **não é subir de modelo dentro da mesma stack** — é trocar de stack: **OpenVINO** (toolkit da própria Intel, com modelos Whisper otimizados que rodam na iGPU e no NPU do Core Ultra). É uma troca de motor de inferência inteira (substitui `ctranslate2`/`faster-whisper`, não só uma variável de `.env`), então só vale investigar se o caminho atual em CPU se provar insuficiente na prática.

**Justificativa do Raffa:** a API da OpenAI (`gpt-4o-transcribe-diarize`) está instável — **perdemos chunks inteiros em duas sessões diferentes** (hoje: chunk 3 da call ACT/BTG, timeout total após 2 retries; e a sessão da entrevista Nava/BMG também precisou de recuperação via reprocessamento do MP3). Além da instabilidade, um modelo local elimina custo por minuto.

**Contexto técnico (da investigação de 01/07/2026):** o gargalo de hoje foi causado por uma combinação de (a) o modelo da OpenAI dar timeout de 180s repetidamente em chunks de ~5-6 min, e (b) o `_chunk_loop` em `main.py` ser síncrono — o timer do próximo chunk só recomeça depois que o anterior termina de processar (com todos os retries), o que faz um chunk lento inflar o próximo (cascata: 5min → 14min → timeout total). Ver sessão `sessions/2026-07-01 - microfone code22, gargalo memorymeet e entrevista act-btg.md` no repo `projeto-carreira-2026` para o diagnóstico completo (inclui proposta de fix incremental: paralelizar `_processar_chunk` e/ou reduzir `CHUNK_SEGUNDOS`, discussão ainda pendente).

**Caminho de migração (levantado em conversa, não validado ainda):**
- **faster-whisper** (ou `openai-whisper`) para transcrição local — CPU ou GPU, sem custo por minuto.
- **pyannote.audio** para diarização — grátis, mas os modelos pretrained são "gated" no Hugging Face (aceitar licença + gerar token, sem custo).
- **WhisperX** empacota os dois com alinhamento palavra-a-palavra — é o caminho mais direto pra reproduzir o que o app faz hoje, 100% local.

**Trade-offs a validar antes de migrar:**
- **Sem GPU dedicada neste notebook** (só Intel Graphics integrado, sem `nvidia-smi` — checado em 01/07/2026). `faster-whisper` roda razoável em CPU (modelo pequeno/médio + int8); `pyannote.audio` em CPU é mais lento que com GPU. **Não é bloqueio real:** a arquitetura já processa em chunks com antecedência (não é tempo real hoje, mesmo com a API da OpenAI), então o que importa é só não acumular atraso indefinidamente — mesmo critério que já vale pro bug do `_chunk_loop` síncrono acima. Raffa está otimista que CPU dá conta nesse regime.
- A diarização por pyannote tende a ser um pouco menos estável em trocas rápidas de falante do que a abordagem atual (duas referências de voz ancoradas por canal, mic/sistema) — pode precisar adaptar a lógica de referência ancorada pro pyannote, não só trocar o modelo.
- ~~Não elimina o bug do `_chunk_loop` síncrono por si só~~ — resolvido em 06/07/2026 com o pipeline cortador→worker→orquestrador (ver item FEITO acima).

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

## 💡 Futuro — Versão navegador (PWA em JS), zero instalação — emergida 09/07/2026

Ideia: uma versão do MemoryMeet que roda 100% no navegador, sem Python, sem setup. Bom pra "abre e grava" rápido, mantendo o app nativo como ferramenta séria (background, performance, entrevista longa).

### Cenário de uso pensado
Raffa abre a aba do MemoryMeet **antes** da agenda → o app pede o que capturar (`getDisplayMedia`) e ele escolhe **a aba do próprio Meet/Teams** com "compartilhar áudio da aba" (captura só a voz do outro lado, canal limpo, sem notificação/Spotify) → o mic dele é capturado em paralelo por `getUserMedia` em outro canal (mic não é exclusivo, não conflita com o Meet usando o mic) → faz a reunião → volta na aba e clica "finalizar".

### Peças e viabilidade
- **Mic (Raffa):** `getUserMedia({audio})` — trivial.
- **Outro lado (Interlocutor):** `getDisplayMedia({audio:true})` compartilhando a aba da reunião. **Chrome/Edge no Windows.**
- **Separação de speaker:** dois `MediaStream` = dois canais físicos → **rotular por canal e descartar o pyannote inteiro**. A peça sem porta boa em JS é exatamente a que não precisaríamos portar — mesma estratégia de canal ancorado que já dá o melhor resultado hoje (mic=Raffa, loopback=Interlocutor).
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
