# Voice Pipeline v2 — Pipecat Edition

Real-time voice conversation with Claude via Discord voice channels.

**Pipeline:** Discord mic → VAD → Whisper STT → Clawdbot LLM → Kyutai TTS → Discord speaker

## Quick Start

```bash
cd /home/anna/clawd/voice-pipeline/v2
source venv/bin/activate
VOICE_BOT_TOKEN="$(cat /home/anna/clawd/voice-pipeline/.voice-bot-token)" python3 voice_bot_v2.py
```

The bot auto-joins when Anna joins a voice channel, auto-leaves when the channel empties.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pipecat Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Discord Voice ──┐                                                      │
│  (48kHz stereo)  │                                                      │
│                  ▼                                                      │
│         ┌────────────────┐                                              │
│         │ DiscordInput   │  Resample 48kHz stereo → 16kHz mono         │
│         │ Transport      │  Thread-safe queue bridging                  │
│         └───────┬────────┘                                              │
│                 │                                                       │
│                 ▼                                                       │
│         ┌────────────────┐                                              │
│         │ VADProcessor   │  Silero VAD — detects speech start/stop     │
│         │                │  Triggers transcription on silence           │
│         └───────┬────────┘                                              │
│                 │                                                       │
│                 ▼                                                       │
│         ┌────────────────┐                                              │
│         │ WhisperSTT     │  faster-whisper large-v3 (float16, CUDA)    │
│         │ Service        │  ~0.3s transcription latency                 │
│         └───────┬────────┘                                              │
│                 │ TranscriptionFrame                                    │
│                 ▼                                                       │
│         ┌────────────────┐                                              │
│         │ ClawdbotLLM    │  Routes through `clawdbot agent`            │
│         │ Service        │  Full context: SOUL.md, memory, tools       │
│         └───────┬────────┘                                              │
│                 │ LLMTextFrame                                          │
│                 ▼                                                       │
│         ┌────────────────┐                                              │
│         │ KyutaiTTS      │  Kyutai TTS 1.6B (n_q=8)                     │
│         │ Service        │  ~6x realtime, 24kHz output                  │
│         └───────┬────────┘                                              │
│                 │ TTSAudioRawFrame                                      │
│                 ▼                                                       │
│         ┌────────────────┐                                              │
│         │ DiscordOutput  │  Resample 24kHz mono → 48kHz stereo         │
│         │ Transport      │  Volume scaling (30% default)                │
│         └───────┬────────┘                                              │
│                 │                                                       │
│                 ▼                                                       │
│         Discord Voice                                                   │
│         (48kHz stereo)                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### `voice_bot_v2.py`
Main entry point. Creates the Pipecat pipeline and runs the Discord bot.

**Configuration constants:**
- `GUILD_ID` — Discord server ID
- `AUTO_JOIN_USER_ID` — User to follow into voice channels (Anna's ID)
- `PIPELINE_SAMPLE_RATE_IN/OUT` — 16kHz input, 24kHz output

### `components/discord_transport.py`
Custom Pipecat transport for Discord voice. Three classes:

**`DiscordTransport`** — Main transport, implements `BaseTransport`:
- Creates Discord bot with py-cord
- Auto-joins when target user enters voice
- Auto-leaves when channel empties
- Provides `input()` and `output()` methods for the pipeline

**`DiscordInputTransport`** — Extends `BaseInputTransport`:
- Receives 48kHz stereo PCM from Discord's voice recv thread
- Resamples to 16kHz mono for the pipeline
- Thread-safe bridging via asyncio queue + `call_soon_threadsafe()`

**`DiscordOutputTransport`** — Extends `BaseOutputTransport`:
- Receives 24kHz mono PCM from TTS
- Resamples to 48kHz stereo for Discord
- Volume scaling (default 30%) to avoid blasting eardrums
- Lazy playback start (avoids green "speaking" outline when idle)

**Key fix:** Both transports explicitly call `set_transport_ready()` in their `start()` methods. Pipecat's base classes create critical queues in `set_transport_ready()`, but it may not be called automatically for custom transports. Without this, audio frames get silently dropped.

### `components/clawdbot_llm_service.py`
Routes LLM requests through the Clawdbot agent subprocess.

- Extends `AIService` (not `LLMService`) to avoid OpenAI-style context overhead
- Handles `TranscriptionFrame` directly from STT
- Prepends voice hint: tells Claude this is a voice conversation, no markdown
- Emits `LLMTextFrame` for TTS to consume
- 30-second timeout with graceful error messages

**Voice hint:**
```
[Voice conversation — you're talking live with Anna in a Discord voice channel.
Write naturally for speech: contractions, casual phrasing, no markdown/formatting/lists.
Talk like a real person would. Your response will be spoken aloud via TTS.]
```

### `components/kyutai_tts_service.py`
Kyutai TTS 1.6B text-to-speech service.

- Extends `TTSService` properly (implements `run_tts()` abstract method)
- Outputs 24kHz int16 PCM via `TTSAudioRawFrame`
- Lazy model loading (first TTS request takes ~5s, subsequent requests are fast)
- Streams audio in ~100ms chunks for responsive playback

**Configuration:**
- `n_q=8` — Fewer quantization steps = faster (5.28x realtime vs 1.74x at n_q=32)
- `voice` — Reference voice for cloning (default: `expresso/ex03-ex01_awe_001_channel1_1323s.wav`)
- `temp=0.6` — Generation temperature
- `cfg_coef=2.0` — Classifier-free guidance coefficient

## Changing the Voice

Edit `KYUTAI_VOICE` in `components/kyutai_tts_service.py`:

```python
KYUTAI_VOICE = "expresso/ex03-ex01_awe_001_channel1_1323s.wav"  # Current
```

Available voices are in the `kyutai/tts-voices` HuggingFace repo. Run `sample_voices_kyutai.py` to generate comparison samples.

**Good options:**
- `expresso/ex03-ex01_awe_001_channel1_1323s.wav` — wonder/curiosity (current, 1323s reference)
- `expresso/ex03-ex01_calm_001_channel1_1143s.wav` — warm, relaxed
- `expresso/ex03-ex02_narration_001_channel1_674s.wav` — storytelling
- `alba-mackenna/casual.wav` — conversational

Longer reference clips (the `_Xs.wav` suffix is duration) give better voice cloning quality.

## Changing the Volume

Edit `resample_pipeline_to_discord()` in `components/discord_transport.py`:

```python
def resample_pipeline_to_discord(pcm_data: bytes, source_rate: int, volume: float = 0.3) -> bytes:
```

Values: 0.0 (silent) to 1.0 (full volume). Default 0.3 is comfortable.

## Performance

Benchmarked on RTX 5070 Ti (16GB VRAM):

| Component | Latency | Notes |
|-----------|---------|-------|
| VAD | ~200ms | Wait for speech to stop |
| STT | ~300ms | faster-whisper large-v3 |
| LLM | ~4-6s | Claude Opus thinking time |
| TTS | ~3s for 18s audio | 6x realtime |
| **Total** | **~8-10s** | From end of speech to start of response |

**VRAM usage:** ~8GB total (Whisper + Kyutai TTS)

## File Structure

```
voice-pipeline/v2/
├── voice_bot_v2.py          # Main entry point
├── components/
│   ├── discord_transport.py # Discord ↔ Pipecat bridging
│   ├── clawdbot_llm_service.py # Clawdbot agent integration
│   └── kyutai_tts_service.py   # Kyutai TTS 1.6B
├── benchmarks/              # Speed benchmarks
├── voice_samples/           # Generated voice comparison samples
├── models/                  # Cached model weights
├── logs/                    # Runtime logs
├── venv/                    # Python virtual environment
└── requirements.txt         # Dependencies
```

## Dependencies

Key packages (see `requirements.txt` for full list):
- `pipecat-ai[silero,whisper]==0.0.101` — Pipeline framework
- `py-cord[voice]` — Discord API
- `moshi` — Kyutai TTS model
- `faster-whisper` — STT
- `torch` — GPU inference

## Known Issues

1. **First TTS request is slow (~5s)** — Model loading happens on first request. Subsequent requests are fast.

2. **Audio not received warnings** — Normal when voice channel is quiet. Pipecat warns about audio timeouts but continues working.

3. **int8 compute type doesn't work** — RTX 5070 Ti (Blackwell) has compatibility issues. Use float16.

## Troubleshooting

**Bot doesn't join voice channel:**
- Check `VOICE_BOT_TOKEN` is set
- Verify `AUTO_JOIN_USER_ID` matches Anna's Discord ID
- Check bot has voice permissions in the server

**Audio not playing:**
- Check logs for `set_transport_ready` — if missing, the output queue isn't initialized
- Verify volume isn't 0.0

**STT not transcribing:**
- Check VAD is detecting speech (look for `VADUserStarted/StoppedSpeakingFrame` in debug logs)
- Verify microphone is working in Discord

**High latency:**
- LLM thinking time is the main bottleneck (~4-6s with Opus)
- Consider using a faster model for voice (Sonnet, Haiku)

## Development

**Run with debug logging:**
```bash
LOGURU_LEVEL=DEBUG VOICE_BOT_TOKEN="..." python3 voice_bot_v2.py
```

**Generate voice samples:**
```bash
python3 sample_voices_kyutai.py
# Samples saved to voice_samples/
```

**Run benchmarks:**
```bash
python3 benchmarks/benchmark_kyutai.py
# Results in benchmarks/kyutai_speed_results.md
```

---

*Built Feb 4, 2026 by Claude & Anna. Four late nights, countless "it's not working" moments, one working voice bot.*
