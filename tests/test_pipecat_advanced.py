#!/usr/bin/env python3
"""
Advanced Pipecat exploration to understand VAD, audio processing, and service integration.
"""

import asyncio
import os
import numpy as np
from pipecat.frames.frames import AudioRawFrame
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.audio.audio_buffer_processor import AudioBuffer

async def test_vad_processor():
    """Test VAD (Voice Activity Detection) processor"""
    print("=== Testing VAD Processor ===")
    
    try:
        # Try to create a VAD processor
        vad = VADProcessor()
        print("✓ VADProcessor created successfully")
        print(f"  Type: {type(vad).__name__}")
        
        # Create some test audio (silence then noise)
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # First half silence, second half noise
        audio_data = np.zeros(samples, dtype=np.float32)
        audio_data[samples//2:] = np.random.normal(0, 0.1, samples//2)
        
        frame = AudioRawFrame(
            audio=audio_data.tobytes(), 
            sample_rate=sample_rate, 
            num_channels=1
        )
        
        print(f"  Test frame: {len(frame.audio)} bytes, {frame.sample_rate}Hz")
        
    except Exception as e:
        print(f"✗ VAD test failed: {e}")

async def test_audio_buffer():
    """Test audio buffer processor"""
    print("\n=== Testing Audio Buffer ===")
    
    try:
        # Create audio buffer
        buffer = AudioBuffer()
        print("✓ AudioBuffer created successfully")
        print(f"  Type: {type(buffer).__name__}")
        
    except Exception as e:
        print(f"✗ AudioBuffer test failed: {e}")

async def test_anthropic_service():
    """Test Anthropic LLM service configuration"""
    print("\n=== Testing Anthropic Service ===")
    
    try:
        # Note: We can't actually test this without API keys, but we can see if it initializes
        service_class = AnthropicLLMService
        print(f"✓ AnthropicLLMService class available: {service_class}")
        print(f"  Module: {service_class.__module__}")
        
        # Check if we can see its parameters
        import inspect
        sig = inspect.signature(service_class.__init__)
        print(f"  Init parameters: {list(sig.parameters.keys())}")
        
    except Exception as e:
        print(f"✗ Anthropic service test failed: {e}")

async def test_whisper_service():
    """Test Whisper STT service"""
    print("\n=== Testing Whisper Service ===")
    
    try:
        service_class = WhisperSTTService
        print(f"✓ WhisperSTTService class available: {service_class}")
        print(f"  Module: {service_class.__module__}")
        
        # Check parameters
        import inspect
        sig = inspect.signature(service_class.__init__)
        print(f"  Init parameters: {list(sig.parameters.keys())}")
        
    except Exception as e:
        print(f"✗ Whisper service test failed: {e}")

async def explore_frame_types():
    """Explore different frame types available in Pipecat"""
    print("\n=== Exploring Frame Types ===")
    
    try:
        from pipecat.frames import frames
        
        frame_classes = []
        for attr_name in dir(frames):
            attr = getattr(frames, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Frame'):
                frame_classes.append(attr_name)
        
        print("Available frame types:")
        for frame_type in sorted(frame_classes):
            print(f"  - {frame_type}")
            
    except Exception as e:
        print(f"✗ Frame exploration failed: {e}")

async def main():
    """Run all advanced tests"""
    await test_vad_processor()
    await test_audio_buffer()
    await test_anthropic_service()
    await test_whisper_service()
    await explore_frame_types()
    print("\n=== Advanced Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())