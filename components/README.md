# Voice Pipeline v2 — Components

Custom Pipecat services for the Discord voice bot.

## Overview

These components bridge external systems (Discord, Clawdbot, Kyutai TTS) with Pipecat's frame-based pipeline architecture. Each component extends a Pipecat base class and implements the required abstract methods.

## Component Hierarchy

```
Pipecat Base Classes          Our Components
─────────────────────         ─────────────────────
BaseTransport            ───► DiscordTransport
├─ BaseInputTransport    ───► DiscordInputTransport
└─ BaseOutputTransport   ───► DiscordOutputTransport

AIService                ───► ClawdbotLLMService

TTSService               ───► KyutaiTTSService
```

## discord_transport.py

**Purpose:** Bridge py-cord's Discord voice I/O with Pipecat's async frame pipeline.

**Challenge:** Discord's voice send/receive runs in separate threads, but Pipecat is asyncio-based. We use `call_soon_threadsafe()` and asyncio queues to bridge safely.

### Classes

**`DiscordAudioSink`** — Receives raw audio from Discord voice recv thread
- `write(data, user)` called per 20ms audio packet
- Resamples 48kHz stereo → 16kHz mono
- Pushes `InputAudioRawFrame` to the input transport's queue

**`DiscordAudioSource`** — Provides audio to Discord voice send thread
- `read()` called every 20ms, must return exactly 3840 bytes
- Thread-safe buffer with lock
- Returns silence on buffer underrun

**`DiscordInputTransport`** — Pipecat input transport
- Extends `BaseInputTransport`
- Runs `_feeder_loop()` to drain the sync queue and push frames into the pipeline
- **Critical:** Calls `set_transport_ready()` to initialize the audio queue

**`DiscordOutputTransport`** — Pipecat output transport
- Extends `BaseOutputTransport`
- Implements `write_audio_frame()` to feed audio to Discord
- Resamples 24kHz mono → 48kHz stereo with volume scaling
- Lazy playback start (no green "speaking" indicator when idle)

**`DiscordTransport`** — Main transport coordinator
- Extends `BaseTransport`
- Manages Discord bot lifecycle
- Auto-joins when target user enters voice
- Auto-leaves when channel empties

### Audio Flow

```
Discord recv thread                    Pipecat async pipeline
────────────────────                   ─────────────────────
DiscordAudioSink.write()
    │
    ├─ resample_discord_to_pipeline()
    │
    └─ loop.call_soon_threadsafe(queue.put_nowait)
                                        │
                                        ▼
                                   _feeder_loop()
                                        │
                                        └─ push_audio_frame()
                                             │
                                             ▼
                                        VAD → STT → LLM → TTS
                                                          │
                                                          ▼
                                   write_audio_frame()
                                        │
    ┌───────────────────────────────────┘
    │
    ├─ resample_pipeline_to_discord()
    │
    └─ _audio_source.feed()
              │
              ▼
Discord send thread
────────────────────
DiscordAudioSource.read()
```

## clawdbot_llm_service.py

**Purpose:** Route LLM requests through Clawdbot's agent system for full context (SOUL.md, memory, tools).

**Why not use Pipecat's built-in LLMService?**
- LLMService assumes OpenAI-style context management
- We want Clawdbot's full agent experience, not raw API calls
- Subprocess call is simpler than reimplementing context injection

### Flow

1. Receive `TranscriptionFrame` from STT
2. Prepend voice hint (tells Claude this is a voice conversation)
3. Call `clawdbot agent --session-id voice --json`
4. Parse JSON response, extract text
5. Emit `LLMTextFrame` for TTS

### Error Handling

- Empty output → "I didn't catch that"
- Timeout → "I took too long thinking"
- JSON parse error → Treat as plain text
- Process cleanup on cancel

## kyutai_tts_service.py

**Purpose:** High-quality, fast text-to-speech using Kyutai's TTS 1.6B model.

**Why Kyutai over Orpheus?**
- Kyutai: 6x realtime at n_q=8, professional voice quality
- Orpheus: Great quality but required complex vLLM setup
- Kyutai's moshi library is simpler to integrate

### Configuration

```python
KYUTAI_N_Q = 8       # Quantization steps (lower = faster, slight quality loss)
KYUTAI_TEMP = 0.6    # Generation temperature
KYUTAI_CFG_COEF = 2.0  # Classifier-free guidance
```

### Lazy Loading

Model loading takes ~5s and uses ~4GB VRAM. We load on first TTS request rather than at startup, so the bot connects quickly even if TTS isn't needed immediately.

### Streaming

Audio is generated in one batch, then chunked into ~100ms frames for streaming playback. True streaming generation would require model changes.

---

## Development Notes

### Adding a new transport

1. Extend `BaseTransport`
2. Implement `input()` → `BaseInputTransport` subclass
3. Implement `output()` → `BaseOutputTransport` subclass
4. **Call `set_transport_ready(frame)` in both `start()` methods** — Pipecat creates critical queues there

### Adding a new LLM service

1. Extend `AIService` (simpler) or `LLMService` (if you need context management)
2. Override `process_frame()` to handle `TranscriptionFrame` or `TextFrame`
3. Emit `LLMFullResponseStartFrame`, `LLMTextFrame`, `LLMFullResponseEndFrame`

### Adding a new TTS service

1. Extend `TTSService`
2. Implement `run_tts(text) -> AsyncGenerator[Frame, None]`
3. Yield `TTSStartedFrame`, `TTSAudioRawFrame` chunks, `TTSStoppedFrame`
4. Set `sample_rate` in `__init__` before calling `super().__init__()`
