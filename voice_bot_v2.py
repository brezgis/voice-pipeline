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
  cd /home/anna/clawd/voice-pipeline/v2
  source venv/bin/activate
  python voice_bot_v2.py
"""

import asyncio
import os
import sys
import signal
from pathlib import Path

from loguru import logger

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
from clawdbot_llm_service import ClawdbotLLMService
from kyutai_tts_service import KyutaiTTSService

# Configuration
GUILD_ID = 1465514323291144377
AUTO_JOIN_USER_ID = 1411361963308613867  # Anna
PIPELINE_SAMPLE_RATE_IN = 16000
PIPELINE_SAMPLE_RATE_OUT = 24000  # Kyutai TTS outputs 24kHz


async def main():
    """Main entry point."""
    bot_token = os.getenv("VOICE_BOT_TOKEN")
    if not bot_token:
        logger.error("VOICE_BOT_TOKEN environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Voice Bot v2 — Pipecat Edition")
    logger.info("Pipeline: Discord → VAD → Whisper STT → Clawdbot → Kyutai TTS → Discord")
    logger.info("=" * 60)

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

    # --- Create transport ---
    transport = DiscordTransport(
        bot_token=bot_token,
        guild_id=GUILD_ID,
        auto_join_user_id=AUTO_JOIN_USER_ID,
    )

    # --- Create pipeline processors ---

    # VAD: Silero-based voice activity detection
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    # STT: faster-whisper large-v3
    # audio_passthrough=False: no need to push raw audio past STT
    stt = WhisperSTTService(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        audio_passthrough=False,
    )

    # LLM: Route through Clawdbot agent
    llm = ClawdbotLLMService(session_id="voice", timeout=30)

    # TTS: Kyutai TTS 1.6B
    tts = KyutaiTTSService(n_q=8, device="cuda")

    # --- Build pipeline ---
    # Chain: discord_input → vad → stt → llm → tts → discord_output
    pipeline = Pipeline([
        transport.input(),   # DiscordInputTransport (BaseInputTransport)
        vad,                 # VADProcessor → VADUserStarted/StoppedSpeakingFrame
        stt,                 # WhisperSTTService → TranscriptionFrame
        llm,                 # ClawdbotLLMService → LLMTextFrame + LLMFullResponseEndFrame
        tts,                 # KyutaiTTSService → TTSAudioRawFrame
        transport.output(),  # DiscordOutputTransport (BaseOutputTransport)
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
    # The Discord bot needs to run alongside the pipeline.
    # We start the bot in a background task and run the pipeline runner.

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
