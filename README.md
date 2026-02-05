# Voice Pipeline v2 - Pipecat Edition

A next-generation voice conversation pipeline for Discord using **Pipecat** orchestration and **Kyutai TTS 1.6B** for speech synthesis. This replaces the hand-rolled v1 pipeline with proper turn-taking, interruption handling, and streaming.

## Architecture

```
Discord Voice Channel (py-cord, 48kHz stereo PCM)
  ↕️
Custom Pipecat Discord Transport Adapter
  ↓
Pipecat Pipeline:
  → Silero VAD Processor (voice activity detection)
  → faster-whisper large-v3 STT Service (CUDA, float16)
  → LiveKit EOU Turn Detector (optional, CPU-based, 135M)
  → Clawdbot LLM Service (subprocess: clawdbot agent --session-id voice --json)
  → Kyutai TTS 1.6B Service (streaming, CUDA)
  ↓
Discord Audio Output
```

## Key Components

### 1. Discord Transport Adapter (`discord_transport.py`)
- **Novel Component**: Bridges py-cord's audio I/O with Pipecat's frame protocol
- Handles 48kHz stereo ↔ Pipecat's internal audio format conversion
- Manages Discord bot connection and voice channel lifecycle
- This is the most technically challenging part since Pipecat doesn't have native Discord support

### 2. Kyutai TTS Integration (`kyutai_tts_service.py`)
- **Model**: `kyutai/tts-1.6b-en_fr` (HuggingFace)
- **Codec**: Mimi for high-quality audio generation
- **Streaming**: True token-level streaming (starts playing before full text complete)
- **VRAM**: ~5GB allocation
- **Hardware**: CUDA, float16 (Blackwell GPU compatible)

### 3. Clawdbot LLM Service (`clawdbot_llm_service.py`)
- Routes through `clawdbot agent --session-id voice --json --timeout 30`
- Preserves full Claude session: SOUL.md, memory, tools, identity
- Subprocess management with proper timeout handling
- Same Claude instance as text chats

### 4. Pipeline Orchestration (`voice_bot_v2.py`)
- Main entry point that wires everything together
- Pipecat pipeline configuration and startup
- Error handling and recovery
- Discord bot lifecycle management

## Hardware Requirements

- **GPU**: RTX 5070 Ti (16GB VRAM, Blackwell architecture)
- **VRAM Budget**: faster-whisper (~4GB) + Kyutai TTS (~5GB) = ~9-10GB used
- **CPU**: Ryzen 9 7900X3D, 64GB RAM
- **OS**: Ubuntu 22.04, Python 3.10, CUDA available

## Installation

```bash
cd /home/anna/clawd/voice-pipeline/v2/
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- **Discord Bot Token**: Set `VOICE_BOT_TOKEN` environment variable
- **Guild ID**: 1465514323291144377
- **Auto-join User**: 1411361963308613867 (Anna)

## Usage

```bash
python voice_bot_v2.py
```

The bot will:
1. Connect to Discord
2. Join the voice channel when Anna joins
3. Listen for speech, transcribe, get Claude response, speak back
4. Handle interruptions gracefully via Pipecat's built-in turn management

## Implementation Status

### ✅ Completed
- [x] Virtual environment setup
- [x] Pipecat installation and exploration
- [x] Architecture design

### 🚧 In Progress
- [ ] Discord transport adapter implementation
- [ ] Kyutai TTS service integration
- [ ] Clawdbot LLM service wrapper

### ⏳ TODO  
- [ ] Pipeline wiring and orchestration
- [ ] Component testing scripts
- [ ] Integration testing
- [ ] Performance optimization

## Design Decisions

### Why Pipecat?
- **Turn Management**: Built-in interruption handling and turn detection
- **Streaming**: Proper streaming audio pipeline
- **Modularity**: Clean separation of concerns with processors/services
- **Audio Processing**: Advanced VAD and audio handling capabilities

### Why Kyutai TTS 1.6B?
- **Quality**: State-of-the-art neural codec (Mimi)
- **Streaming**: True token-level streaming for low latency
- **Size**: Efficient 1.6B parameters fit comfortably in VRAM budget
- **Multilingual**: English/French support

### Fallback Plans
If Pipecat Discord integration proves impossible:
- Use Pipecat as library for internal processing (VAD, turn detection, streaming)
- Keep py-cord for Discord I/O layer
- Bridge the two with custom adapters

If Kyutai TTS has issues:
- **Primary Fallback**: Dia2 1B TTS
- **Secondary Fallback**: Existing Orpheus TTS daemon (`/tmp/orpheus-tts.sock`)

## Technical Notes

### CRITICAL Hardware Constraints
- **NO int8 compute type** - Blackwell GPU incompatibility. Use float16/bfloat16 only.
- **vLLM Import Order** - If using vLLM, import BEFORE torch to avoid EngineCore deadlock

### Discord Audio Format
- **Input**: 48kHz stereo PCM from Discord Opus decoder
- **Processing**: Convert to Pipecat's preferred format (likely 16kHz mono)
- **Output**: Upsample back to 48kHz stereo for Discord Opus encoder

### Memory Management
- Streaming pipeline must balance memory usage vs latency
- ~10-11GB VRAM budget leaves 5-6GB for other processes
- Monitor VRAM usage during development

## Development Log

See git commits for detailed development progress. Key milestones:
- Initial Pipecat exploration and architecture design
- Discord transport adapter implementation
- TTS service integration
- Full pipeline testing and optimization

---

*This pipeline represents the evolution from v1's hand-rolled implementation to a production-ready, Pipecat-orchestrated solution that handles edge cases properly and provides a foundation for future enhancements.*