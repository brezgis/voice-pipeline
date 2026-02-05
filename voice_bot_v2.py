#!/usr/bin/env python3
"""
Voice Bot v2 - Pipecat Edition

Main entry point for the Pipecat-based voice pipeline. This replaces the hand-rolled v1 
with proper turn-taking, interruption handling, and streaming.

Architecture:
  Discord → Custom Transport → VAD → faster-whisper → Clawdbot → Kyutai TTS → Discord

Key improvements over v1:
- Pipecat orchestration for proper frame handling
- Built-in VAD and turn detection  
- Streaming TTS with lower latency
- Better interruption handling
- Cleaner separation of concerns

Usage:
  export VOICE_BOT_TOKEN=your_discord_bot_token
  python voice_bot_v2.py
"""

import asyncio
import os
import sys
import signal
from pathlib import Path

# Add components directory to path
sys.path.append(str(Path(__file__).parent / "components"))

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.processors.aggregators.sentence import SentenceAggregator

# Our custom components
from discord_transport import DiscordTransport
from clawdbot_llm_service import ClawdbotLLMService
from kyutai_tts_service import KyutaiTTSService, KyutaiTTSServiceSimple

# Configuration
GUILD_ID = 1465514323291144377
AUTO_JOIN_USER_ID = 1411361963308613867  # Anna


class VoicePipelineV2:
    """
    Main voice pipeline class that orchestrates all components
    """
    
    def __init__(self):
        self.runner = None
        self.task = None
        self.transport = None
        
        # Configuration
        self.bot_token = os.getenv('VOICE_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("VOICE_BOT_TOKEN environment variable not set")
            
        self.model_dir = Path(__file__).parent / "models"
        
    async def create_pipeline(self):
        """Create and configure the Pipecat pipeline"""
        print("Creating Pipecat pipeline...")
        
        # Transport layer
        self.transport = DiscordTransport(
            bot_token=self.bot_token,
            guild_id=GUILD_ID,
            auto_join_user_id=AUTO_JOIN_USER_ID
        )
        
        # Audio processing: VAD for voice activity detection
        vad = VADProcessor()
        
        # STT: faster-whisper for transcription
        stt = WhisperSTTService(
            model="large-v3",
            language="en",
            device="cuda",
            compute_type="float16"  # Blackwell compatible
        )
        
        # Text aggregation: collect sentences before sending to LLM
        sentence_aggregator = SentenceAggregator()
        
        # LLM: Route through Clawdbot agent
        llm = ClawdbotLLMService(
            session_id="voice",
            timeout=30
        )
        
        # TTS: Try Kyutai TTS, fallback to simple version
        try:
            # Check if models are available
            kyutai_model_path = self.model_dir / "kyutai_tts"
            if kyutai_model_path.exists():
                print("Using full Kyutai TTS service")
                tts = KyutaiTTSService(
                    model_path=str(kyutai_model_path),
                    device="cuda"
                )
            else:
                print("Using simple Kyutai TTS service (no local models)")
                tts = KyutaiTTSServiceSimple()
                
        except Exception as e:
            print(f"Error setting up Kyutai TTS: {e}")
            print("Falling back to simple TTS service")
            tts = KyutaiTTSServiceSimple()
        
        # Create pipeline with processor chain
        processors = [
            vad,                # Voice activity detection
            stt,                # Speech to text  
            sentence_aggregator, # Collect complete sentences
            llm,                # Language model processing
            tts                 # Text to speech
        ]
        
        pipeline = Pipeline(processors)
        
        # Create pipeline task with transport
        self.task = PipelineTask(
            pipeline,
            input_transport=self.transport,
            output_transport=self.transport
        )
        
        print("Pipeline created successfully!")
        return self.task
        
    async def start(self):
        """Start the voice pipeline"""
        print("Starting Voice Pipeline v2 (Pipecat Edition)")
        print("=" * 50)
        
        try:
            # Create pipeline
            task = await self.create_pipeline()
            
            # Create and start runner
            self.runner = PipelineRunner()
            
            print("Starting Discord transport...")
            await self.transport.start()
            
            print("Starting pipeline runner...")
            await self.runner.run(task)
            
        except KeyboardInterrupt:
            print("\\nReceived shutdown signal")
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the voice pipeline and cleanup"""
        print("\\nStopping Voice Pipeline v2...")
        
        if self.transport:
            await self.transport.stop()
            
        if self.runner:
            # Stop the runner (if it supports stopping)
            try:
                await self.runner.stop()
            except AttributeError:
                pass  # Runner might not have stop method
                
        print("Pipeline stopped.")

def setup_signal_handlers(pipeline):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        print(f"\\nReceived signal {sig}")
        # Create new event loop for cleanup if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # Schedule stop in the running loop
            asyncio.create_task(pipeline.stop())
        else:
            # Run stop directly
            loop.run_until_complete(pipeline.stop())
            
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_requirements():
    """Check that all requirements are met"""
    print("Checking requirements...")
    
    # Check Discord bot token
    if not os.getenv('VOICE_BOT_TOKEN'):
        print("✗ VOICE_BOT_TOKEN environment variable not set")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA available: {gpu_name} ({vram_gb:.1f}GB)")
        else:
            print("! CUDA not available - will use CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
        
    # Check faster-whisper
    try:
        import faster_whisper
        print("✓ faster-whisper available")
    except ImportError:
        print("✗ faster-whisper not installed")
        return False
        
    # Check moshi
    try:
        import moshi
        print("✓ moshi available")
    except ImportError:
        print("! moshi not available - TTS may be limited")
        
    # Check Pipecat
    try:
        import pipecat
        print(f"✓ pipecat {pipecat.__version__} available")
    except ImportError:
        print("✗ pipecat not installed")
        return False
    
    return True

def print_startup_info():
    """Print startup information and instructions"""
    print("Voice Pipeline v2 - Pipecat Edition")
    print("=" * 50)
    print("Architecture:")
    print("  Discord Voice → VAD → faster-whisper → Clawdbot → Kyutai TTS → Discord")
    print()
    print("Features:")
    print("  • Real-time voice activity detection")
    print("  • Streaming speech-to-text (large-v3)")
    print("  • Full Clawdbot integration (SOUL.md, memory, tools)")
    print("  • Streaming text-to-speech (Kyutai TTS 1.6B)")
    print("  • Proper interruption handling")
    print()
    print("Controls:")
    print("  • Speak normally - pipeline detects voice activity")
    print("  • Interrupt Claude while speaking - it will stop gracefully")
    print("  • Ctrl+C to shutdown")
    print()

async def main():
    """Main entry point"""
    print_startup_info()
    
    # Check requirements
    if not check_requirements():
        print("\\nRequirements not met. Please install missing dependencies:")
        print("pip install -r requirements.txt")
        print("\\nAlso ensure VOICE_BOT_TOKEN is set:")
        print("export VOICE_BOT_TOKEN=your_discord_bot_token")
        sys.exit(1)
    
    print("\\n" + "=" * 50)
    
    # Create and start pipeline
    pipeline = VoicePipelineV2()
    
    # Setup signal handlers
    setup_signal_handlers(pipeline)
    
    # Start the pipeline
    await pipeline.start()

if __name__ == "__main__":
    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)