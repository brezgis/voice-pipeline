# Voice Pipeline v2

Real-time voice conversation with an LLM over Discord voice channels. Fully local audio processing (STT + TTS on GPU), with LLM inference routed through [OpenClaw](https://github.com/nicobailon/openclaw) for full agent context — system prompt, memory, tools, identity.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat), with custom services for Discord transport, OpenClaw gateway streaming, and Kyutai TTS.

## Architecture

```
Discord mic (48kHz stereo)
    │
    ▼
┌──────────────────┐
│ DiscordInput     │  Resample 48kHz stereo → 16kHz mono
│ Transport        │  Thread-safe async queue bridging
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Silero VAD       │  Voice activity detection (0.8s stop threshold)
│                  │  Detects speech start/stop, triggers STT on silence
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ faster-whisper   │  Whisper large-v3 (CUDA, float16)
│ STT              │  ~1s transcription latency
└────────┬─────────┘
         │ TranscriptionFrame
         ▼
┌──────────────────┐
│ Streaming LLM    │  SSE streaming via OpenClaw gateway
│ Service          │  /v1/chat/completions (OpenAI-compatible)
└────────┬─────────┘
         │ LLMTextFrame (token-by-token)
         ▼
┌──────────────────┐
│ Pipecat sentence │  Collects tokens into complete sentences
│ aggregator       │  TTS starts on FIRST sentence while LLM keeps generating
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Kyutai TTS       │  Kyutai TTS 1.6B (CUDA), 24kHz output
│ 1.6B             │  5.28x realtime at n_q=8
└────────┬─────────┘
         │ TTSAudioRawFrame
         ▼
┌──────────────────┐
│ DiscordOutput    │  Resample 24kHz mono → 48kHz stereo
│ Transport        │  Volume scaling, lazy playback start
└────────┬─────────┘
         │
         ▼
Discord speaker (48kHz stereo)
```

### The Streaming Breakthrough

The key innovation in v2 is **streaming LLM → TTS**. Instead of waiting for the entire LLM response before starting TTS:

1. The `StreamingGatewayLLMService` opens an SSE connection to the OpenClaw gateway
2. Tokens arrive one-by-one as `LLMTextFrame`s
3. Pipecat's built-in sentence aggregator collects tokens into complete sentences
4. TTS generates audio for the **first sentence** while the LLM is still producing text
5. The user hears the beginning of the response seconds before the LLM finishes

For a 12-second LLM response, the old pipeline (v1) waited 12 seconds in silence. The new pipeline starts audio at ~3.5 seconds.

### Fallback: Blocking CLI Mode

The `ClawdbotLLMService` provides a fallback that shells out to `openclaw agent` and waits for the complete response. Useful for debugging or when the gateway isn't available. Set `VOICE_LLM_STREAMING=false` to use it.

## Latency Benchmarks

Measured on RTX 5070 Ti with Claude Opus:

| Pipeline Stage | v1 (blocking) | v2 (streaming) |
|---|---|---|
| VAD stop detection | 1500ms | 800ms |
| STT (faster-whisper large-v3) | ~1000ms | ~1000ms |
| LLM time to first token | 2000–5000ms | 2670–3130ms |
| LLM total (blocking wait) | 2000–5000ms | N/A (streams) |
| TTS first sentence | waited for full LLM | ~500ms after TTFT |
| **End-of-speech → first audio** | **6–10s** | **~4.5–5s** |

**Key insight:** With streaming, TTS latency is decoupled from total LLM response time. A long, thoughtful response starts playing almost as quickly as a short one.

## Components

### `voice_bot_v2.py` — Main Orchestrator

Creates the Pipecat pipeline and runs the Discord bot. Configurable via environment variables.

### `components/streaming_gateway_llm_service.py` — Streaming LLM (default)

SSE streaming via the OpenClaw gateway's OpenAI-compatible `/v1/chat/completions` endpoint. Emits tokens as `LLMTextFrame`s for sentence-level TTS streaming. Maintains conversation history for multi-turn context.

Reads auth token from `OPENCLAW_TOKEN` env var or `~/.openclaw/openclaw.json`.

### `components/clawdbot_llm_service.py` — Blocking CLI Fallback

Routes through `openclaw agent` subprocess. Waits for the complete response before emitting text. Simpler but higher latency. Useful when the gateway isn't running.

### `components/kyutai_tts_service.py` — Kyutai TTS 1.6B

[Kyutai/moshi](https://github.com/kyutai-labs/moshi) TTS model. Extends Pipecat's `TTSService` with proper `run_tts()` implementation. Outputs 24kHz int16 PCM in ~100ms chunks.

- **Lazy loading:** Model loads on first TTS request (~5s, ~4GB VRAM)
- **Voice cloning:** Configurable reference voice from the `kyutai/tts-voices` HuggingFace repo
- **Performance tuning:** `n_q` parameter controls speed vs quality (8 = 5.28x realtime, 32 = 1.74x)

### `components/discord_transport.py` — Discord ↔ Pipecat Bridge

Custom Pipecat transport using [py-cord](https://github.com/Pycord-Development/pycord) for Discord voice. Handles:

- **Audio resampling:** 48kHz stereo (Discord) ↔ 16kHz mono (pipeline) ↔ 24kHz mono (TTS)
- **Thread-safe bridging:** Discord's voice recv/send threads ↔ Pipecat's asyncio event loop
- **Auto-join/leave:** Follows a target user into voice channels, leaves when channel empties
- **Volume scaling:** Default 30% to avoid blasting eardrums
- **Lazy playback:** No green "speaking" indicator when idle

## Setup

### Prerequisites

- NVIDIA GPU with 8GB+ VRAM (tested on RTX 5070 Ti with 16GB)
- CUDA toolkit
- Python 3.10+
- [OpenClaw](https://github.com/nicobailon/openclaw) gateway running locally
- A Discord bot application with voice permissions

### Installation

```bash
git clone https://github.com/brezgis/voice-pipeline.git
cd voice-pipeline/v2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### First-time model setup

```bash
python setup_models.py   # Downloads Kyutai TTS 1.6B (~4GB)
```

faster-whisper large-v3 downloads automatically on first use (~3GB).

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `VOICE_BOT_TOKEN` | **Yes** | — | Discord bot token |
| `DISCORD_GUILD_ID` | No | `0` (auto) | Discord server ID |
| `DISCORD_AUTO_JOIN_USER` | No | `0` (disabled) | User ID to follow into voice |
| `VAD_STOP_SECS` | No | `1.5` | Silence duration before processing |
| `VOICE_LLM_STREAMING` | No | `true` | Use streaming gateway (`true`) or blocking CLI (`false`) |
| `OPENCLAW_TOKEN` | No | from config | OpenClaw gateway auth token |

### Running

```bash
source venv/bin/activate
export VOICE_BOT_TOKEN="your_token_here"
python voice_bot_v2.py
```

Or use the launcher script from the parent directory:

```bash
./start-voice.sh
```

The bot auto-joins when the target user enters a voice channel and auto-leaves when the channel empties.

## Changing the Voice

Edit `KYUTAI_VOICE` in `components/kyutai_tts_service.py`:

```python
KYUTAI_VOICE = "expresso/ex03-ex01_awe_001_channel1_1323s.wav"
```

Available voices are in the [`kyutai/tts-voices`](https://huggingface.co/kyutai/tts-voices) HuggingFace repo. Run `sample_voices_kyutai.py` to generate comparison samples locally.

Longer reference clips produce better voice cloning quality.

## Hardware Requirements

| Resource | Requirement | Notes |
|---|---|---|
| GPU VRAM | 8GB minimum | Whisper (~4GB) + Kyutai TTS (~4GB) |
| RAM | 16GB+ | For model loading |
| Disk | ~10GB | Model weights (downloaded once) |
| GPU | NVIDIA with CUDA | CPU is technically possible but impractically slow |

**Tested on:** RTX 5070 Ti (16GB VRAM), Ryzen 9 7900X3D, 64GB RAM, Ubuntu 22.04

**Known GPU issue:** int8 compute type doesn't work on Blackwell GPUs. Use float16.

## VRAM Usage

| Component | VRAM | Notes |
|---|---|---|
| faster-whisper large-v3 | ~4 GB | CTranslate2, float16 |
| Kyutai TTS 1.6B | ~4 GB | n_q=8, CUDA |
| **Total** | **~8 GB** | Fits on 8GB+ GPUs |

## File Structure

```
voice-pipeline/v2/
├── voice_bot_v2.py              # Main entry point
├── components/
│   ├── streaming_gateway_llm_service.py  # SSE streaming LLM (default)
│   ├── clawdbot_llm_service.py           # Blocking CLI fallback
│   ├── kyutai_tts_service.py             # Kyutai TTS 1.6B
│   ├── discord_transport.py              # Discord ↔ Pipecat bridge
│   └── README.md                         # Component architecture docs
├── benchmarks/                  # Speed benchmarks for Kyutai TTS
├── tests/                       # Component and integration tests
├── requirements.txt             # Python dependencies
├── setup_models.py              # First-time model downloader
├── sample_voices_kyutai.py      # Voice comparison sample generator
└── .gitignore
```

## Tech Stack

- **Pipeline framework:** [Pipecat](https://github.com/pipecat-ai/pipecat) 0.0.101
- **STT:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3, CTranslate2)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **LLM:** Any model via [OpenClaw](https://github.com/nicobailon/openclaw) gateway (OpenAI-compatible API)
- **TTS:** [Kyutai TTS 1.6B](https://github.com/kyutai-labs/moshi) (moshi library)
- **Discord:** [py-cord](https://github.com/Pycord-Development/pycord) 2.x
- **GPU inference:** PyTorch + CUDA

## Future Work

- **Faster TTFT:** Sonnet or other fast models for voice (current Opus TTFT is 2.7–3.1s)
- **Streaming TTS:** True token-level TTS via `on_frame` callback (currently generates full sentence then chunks)
- **SmartTurn VAD:** ML-based turn detection instead of fixed silence threshold
- **Filler audio:** Play "hmm" / thinking sounds during LLM processing to reduce perceived latency
- **Multi-user:** Track and attribute multiple speakers in a voice channel

## Known Issues

1. **First TTS request is slow (~5s)** — Kyutai model loads on first request. Subsequent requests are fast.
2. **"Audio not received" warnings** — Normal when the voice channel is quiet. Pipecat warns about audio timeouts but continues working.
3. **int8 compute type crashes** — Blackwell GPU compatibility issue. Use float16.

## License

MIT
