#!/usr/bin/env python3
"""
Test script to explore Pipecat's basic capabilities and understand the framework.
This will help inform the design of our Discord transport and pipeline architecture.
"""

import asyncio
from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

# Test basic frame processing
class EchoProcessor(FrameProcessor):
    """Simple processor that echoes frames for testing"""
    
    async def process_frame(self, frame: Frame, direction):
        print(f"EchoProcessor received: {type(frame).__name__}")
        if isinstance(frame, AudioRawFrame):
            print(f"  Audio frame: {len(frame.audio)} samples, sample_rate={frame.sample_rate}")
        await self.push_frame(frame, direction)

async def test_basic_pipeline():
    """Test creating and running a basic pipeline"""
    print("=== Testing Basic Pipecat Pipeline ===")
    
    # Create a simple processor chain
    echo = EchoProcessor()
    
    # Create pipeline
    pipeline = Pipeline([echo])
    
    # Create and run pipeline task
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    
    print("Pipeline created successfully!")
    print(f"Pipeline processors: {[type(p).__name__ for p in pipeline._processors]}")

async def explore_services():
    """Explore available Pipecat services"""
    print("\n=== Exploring Pipecat Services ===")
    
    try:
        from pipecat.services.anthropic import AnthropicLLMService
        print("✓ AnthropicLLMService available")
    except ImportError as e:
        print(f"✗ AnthropicLLMService not available: {e}")
    
    try:
        from pipecat.services.whisper import WhisperSTTService
        print("✓ WhisperSTTService available")
    except ImportError as e:
        print(f"✗ WhisperSTTService not available: {e}")
    
    try:
        # Check for various TTS services
        from pipecat.services.elevenlabs import ElevenLabsTTSService
        print("✓ ElevenLabsTTSService available")
    except ImportError:
        print("✗ ElevenLabsTTSService not available")
        
    try:
        from pipecat.services.openai import OpenAITTSService
        print("✓ OpenAITTSService available")
    except ImportError:
        print("✗ OpenAITTSService not available")

async def explore_audio_processing():
    """Explore Pipecat's audio processing capabilities"""
    print("\n=== Exploring Audio Processing ===")
    
    try:
        from pipecat.processors.audio.audio import AudioBuffer
        print("✓ AudioBuffer available")
    except ImportError as e:
        print(f"✗ AudioBuffer not available: {e}")
    
    try:
        from pipecat.processors.audio.vad import VADProcessor
        print("✓ VADProcessor available")  
    except ImportError as e:
        print(f"✗ VADProcessor not available: {e}")
        
    # Try to understand audio frame format
    try:
        import numpy as np
        
        # Create a test audio frame
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.random.rand(samples).astype(np.float32)
        
        frame = AudioRawFrame(audio=audio_data.tobytes(), sample_rate=sample_rate, num_channels=1)
        print(f"✓ Created test AudioRawFrame: {len(frame.audio)} bytes, {frame.sample_rate}Hz")
        
    except Exception as e:
        print(f"✗ Error creating audio frame: {e}")

async def main():
    """Run all tests"""
    await test_basic_pipeline()
    await explore_services()
    await explore_audio_processing()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())