#!/usr/bin/env python3
"""
Voice Bot v2 — Pipecat Edition

End-to-end Discord voice bot:
  Discord mic → VAD → faster-whisper STT → Clawdbot LLM → Kyutai TTS → Discord speaker

Uses Pipecat 0.0.101's actual API:
- BaseTransport with input()/output() returning BaseInputTransport/BaseOutputTransport
- Pipeline([input_transport, vad, stt, llm, tts, output_transport])
- PipelineTask(pipeline, params=PipelineParams(...))
- PipelineRunner to manage lifecycle

Usage:
  export VOICE_BOT_TOKEN=your_discord_bot_token
  cd /path/to/voice-pipeline/v2
  source venv/bin/activate
  python voice_bot_v2.py

Configuration (environment variables):
  VOICE_BOT_TOKEN        - Discord bot token (required)
  DISCORD_GUILD_ID       - Server ID (default: 0, auto-detect)
  DISCORD_AUTO_JOIN_USER - User ID to follow into voice (default: unset)
  VAD_STOP_SECS          - Silence duration before processing (default: 0.8, or 0.2 with SmartTurn)
  VOICE_SMART_TURN       - Enable ML-based turn detection (default: true)
  VOICE_STREAMING_TTS    - Enable streaming TTS via on_frame callback (default: true)
"""

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADParams

# Ensure components/ is importable
sys.path.insert(0, str(Path(__file__).parent / "components"))

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.whisper.stt import WhisperSTTService

# Our custom components
from discord_transport import DiscordTransport
from kyutai_tts_service import KyutaiTTSService

# LLM service: streaming gateway (default) or blocking CLI fallback
USE_STREAMING_LLM = os.getenv("VOICE_LLM_STREAMING", "true").lower() in ("true", "1", "yes")
if USE_STREAMING_LLM:
    from streaming_gateway_llm_service import StreamingGatewayLLMService
else:
    from clawdbot_llm_service import ClawdbotLLMService

# =============================================================================
# Configuration — override via environment variables
# =============================================================================

# Discord server and user to follow — set via environment variables
GUILD_ID = int(os.getenv("DISCORD_GUILD_ID", "0"))
AUTO_JOIN_USER_ID = int(os.getenv("DISCORD_AUTO_JOIN_USER", "0"))

# Audio pipeline settings
PIPELINE_SAMPLE_RATE_IN = 16000
PIPELINE_SAMPLE_RATE_OUT = 24000  # Kyutai TTS outputs 24kHz

# SmartTurn: ML-based end-of-turn detection (pipecat 0.0.101+)
# When enabled, reduces VAD_STOP_SECS to 0.2 since SmartTurn handles
# intelligent end-of-turn prediction, saving ~600ms per turn.
USE_SMART_TURN = os.getenv("VOICE_SMART_TURN", "true").lower() in ("true", "1", "yes")

# VAD settings — how long to wait after speech stops before processing
# With SmartTurn: 0.2s (SmartTurn ML model predicts turn boundaries)
# Without SmartTurn: 0.8s (pure silence-based, conservative default)
VAD_CONFIDENCE = 0.7
VAD_START_SECS = 0.2  # How long speech must last to trigger
_DEFAULT_STOP_SECS = "0.2" if USE_SMART_TURN else "0.8"
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", _DEFAULT_STOP_SECS))
VAD_MIN_VOLUME = 0.6


async def main():
    """Main entry point."""
    bot_token = os.getenv("VOICE_BOT_TOKEN")
    if not bot_token:
        logger.error("VOICE_BOT_TOKEN environment variable not set")
        sys.exit(1)

    llm_mode = "Streaming Gateway" if USE_STREAMING_LLM else "Clawdbot CLI"
    logger.info("=" * 60)
    logger.info("Voice Bot v2 — Pipecat Edition")
    logger.info(f"Pipeline: Discord → VAD → Whisper STT → {llm_mode} → Kyutai TTS → Discord")
    logger.info("=" * 60)
    logger.info(f"Guild ID: {GUILD_ID}")
    logger.info(f"Auto-join user: {AUTO_JOIN_USER_ID}")
    logger.info(f"VAD stop seconds: {VAD_STOP_SECS}")
    logger.info(f"SmartTurn: {USE_SMART_TURN}")
    logger.info(f"Streaming TTS: {os.getenv('VOICE_STREAMING_TTS', 'true')}")
    logger.info(f"LLM mode: {llm_mode}")

    # --- Check prerequisites ---
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name()
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            logger.info(f"GPU: {gpu} ({vram_gb:.1f} GB)")
        else:
            logger.warning("CUDA not available — will be slow")
    except ImportError:
        logger.error("PyTorch not installed")
        sys.exit(1)

    # --- SmartTurn: ML-based end-of-turn detection ---
    turn_analyzer = None
    if USE_SMART_TURN:
        try:
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

            smart_turn_params = SmartTurnParams(
                stop_secs=3.0,       # Max silence before forced end-of-turn
                pre_speech_ms=500,   # Audio context before speech onset
            )
            turn_analyzer = LocalSmartTurnAnalyzerV3(params=smart_turn_params)
            logger.info("SmartTurn v3 (ONNX) loaded — ML-based turn detection active")
        except Exception as e:
            logger.warning(f"SmartTurn not available, falling back to silence-only VAD: {e}")
            turn_analyzer = None

    # --- Create transport ---
    # Pass turn_analyzer through TransportParams if SmartTurn is available.
    # Note: turn_analyzer on TransportParams is deprecated in pipecat 0.0.101+
    # in favor of LLMUserAggregator with TurnAnalyzerUserTurnStopStrategy.
    # TODO: Migrate to LLMUserAggregator when we refactor the pipeline architecture.
    transport_params = None
    if turn_analyzer:
        from pipecat.transports.base_transport import TransportParams
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE_IN,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE_IN,
            audio_out_channels=1,
            turn_analyzer=turn_analyzer,
        )

    transport = DiscordTransport(
        bot_token=bot_token,
        guild_id=GUILD_ID,
        auto_join_user_id=AUTO_JOIN_USER_ID,
        params=transport_params,
    )

    # --- Create pipeline processors ---

    # VAD: Silero-based voice activity detection
    # Longer stop_secs = wait longer after speech stops before processing
    vad_params = VADParams(
        confidence=VAD_CONFIDENCE,
        start_secs=VAD_START_SECS,
        stop_secs=VAD_STOP_SECS,
        min_volume=VAD_MIN_VOLUME,
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params))

    # STT: faster-whisper large-v3
    stt = WhisperSTTService(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        audio_passthrough=False,
    )

    # LLM: Streaming gateway (default) or blocking CLI fallback
    if USE_STREAMING_LLM:
        llm = StreamingGatewayLLMService(session_id="voice", timeout=30)
    else:
        llm = ClawdbotLLMService(session_id="voice", timeout=30)

    # TTS: Kyutai TTS 1.6B
    tts = KyutaiTTSService(n_q=8, device="cuda")

    # --- Build pipeline ---
    pipeline = Pipeline([
        transport.input(),   # DiscordInputTransport
        vad,                 # VADProcessor
        stt,                 # WhisperSTTService
        llm,                 # ClawdbotLLMService
        tts,                 # KyutaiTTSService
        transport.output(),  # DiscordOutputTransport
    ])

    # --- Create task ---
    params = PipelineParams(
        audio_in_sample_rate=PIPELINE_SAMPLE_RATE_IN,
        audio_out_sample_rate=PIPELINE_SAMPLE_RATE_OUT,
        allow_interruptions=True,
    )

    task = PipelineTask(pipeline, params=params)

    # --- Create runner ---
    runner = PipelineRunner(handle_sigint=True, handle_sigterm=True)

    # --- Start Discord bot concurrently with pipeline runner ---
    bot_task = asyncio.create_task(transport.start_bot())

    @task.event_handler("on_pipeline_started")
    async def on_started(task, frame):
        logger.info("Pipeline started and ready")

    @task.event_handler("on_pipeline_finished")
    async def on_finished(task, frame):
        logger.info(f"Pipeline finished: {frame}")

    try:
        logger.info("Starting pipeline runner (Discord bot starting in background)...")
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Pipeline cancelled")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Shutting down...")
        await transport.stop_bot()
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        logger.info("Voice bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
