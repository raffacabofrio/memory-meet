# BACKLOG — MemoryMeet

Itens pendentes e ideias futuras. O mais maduro fica no topo.

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
