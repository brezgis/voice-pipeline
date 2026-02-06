#!/usr/bin/env python3
"""
Component testing script for Voice Pipeline v2

Tests individual components before full pipeline integration:
1. Pipecat framework basics
2. Clawdbot LLM service 
3. Kyutai TTS service (if models available)
4. Discord transport (connection only, no audio)

This helps identify issues at the component level before attempting full integration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add components to path
sys.path.append(str(Path(__file__).parent.parent / "components"))

async def test_pipecat_basics():
    """Test basic Pipecat functionality"""
    print("=== Testing Pipecat Basics ===")
    
    try:
        from pipecat.frames.frames import Frame, TextFrame, AudioRawFrame
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.pipeline.pipeline import Pipeline
        
        # Create a simple echo processor
        class EchoProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                print(f"Echo: {type(frame).__name__}")
                await self.push_frame(frame, direction)
        
        # Test pipeline creation
        echo = EchoProcessor()
        pipeline = Pipeline([echo])
        
        print("✓ Basic Pipecat pipeline created")
        
        # Test frame creation
        text_frame = TextFrame(text="Hello, world!")
        print(f"✓ TextFrame created: {text_frame.text}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipecat test failed: {e}")
        return False

async def test_clawdbot_service():
    """Test Clawdbot LLM service"""
    print("\\n=== Testing Clawdbot Service ===")
    
    try:
        from clawdbot_llm_service import ClawdbotLLMService
        
        # Create service
        service = ClawdbotLLMService(session_id="voice-test")
        print("✓ ClawdbotLLMService created")
        
        # Test basic prompt (this will actually call clawdbot)
        test_prompt = "Hello, this is a test. Please respond briefly."
        print(f"Testing prompt: {test_prompt}")
        
        response_chunks = []
        try:
            async for chunk in service._send_prompt(test_prompt):
                response_chunks.append(chunk)
                if len(response_chunks) <= 3:  # Show first few chunks
                    print(f"  Chunk: {chunk[:50]}...")
                    
            full_response = ''.join(response_chunks)
            if full_response.strip():
                print(f"✓ Got response ({len(full_response)} chars)")
                return True
            else:
                print("✗ No response received")
                return False
                
        except Exception as e:
            print(f"✗ Clawdbot communication failed: {e}")
            return False
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_kyutai_tts():
    """Test Kyutai TTS service"""
    print("\\n=== Testing Kyutai TTS ===")
    
    try:
        from kyutai_tts_service import KyutaiTTSService
        
        # Create service (model loads lazily on first use)
        service = KyutaiTTSService()
        print("✓ KyutaiTTSService created")
        
        # Check if moshi is working
        import moshi
        print(f"✓ moshi library available")
        
        # Note: Full TTS test would require actual model files
        print("✓ TTS service initialized (models would be needed for audio generation)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ TTS service test failed: {e}")
        return False

async def test_discord_transport():
    """Test Discord transport (connection only)"""
    print("\\n=== Testing Discord Transport ===")
    
    token = os.getenv('VOICE_BOT_TOKEN')
    if not token:
        print("✗ VOICE_BOT_TOKEN not set, skipping Discord test")
        return False
        
    try:
        from discord_transport import DiscordTransport
        
        # Create transport (but don't start it)
        transport = DiscordTransport(
            bot_token=token,
            guild_id=0,  # Set your guild ID
            auto_join_user_id=0,  # Set your user ID
        )
        
        print("✓ DiscordTransport created")
        print("Note: Full Discord test requires actually connecting (skipped)")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Discord transport test failed: {e}")
        return False

async def test_audio_dependencies():
    """Test audio processing dependencies"""
    print("\\n=== Testing Audio Dependencies ===")
    
    success = True
    
    # Test faster-whisper
    try:
        from faster_whisper import WhisperModel
        print("✓ faster-whisper available")
        
        # Check CUDA
        try:
            model = WhisperModel("tiny", device="cuda", compute_type="float16")
            print("✓ faster-whisper CUDA support working")
        except Exception as e:
            print(f"! CUDA test failed, will use CPU: {e}")
            
    except ImportError:
        print("✗ faster-whisper not available")
        success = False
    
    # Test torch
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("! CUDA not available")
            
    except ImportError:
        print("✗ torch not available")
        success = False
        
    # Test audio libraries
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError:
        print("✗ numpy not available")
        success = False
        
    return success

def print_test_summary(results):
    """Print test summary"""
    print("\\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for full pipeline testing.")
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
        
    return passed == total

async def main():
    """Run all component tests"""
    print("Voice Pipeline v2 - Component Testing")
    print("=" * 50)
    
    # Run tests
    results = {
        "Pipecat Basics": await test_pipecat_basics(),
        "Audio Dependencies": await test_audio_dependencies(), 
        "Clawdbot Service": await test_clawdbot_service(),
        "Kyutai TTS": await test_kyutai_tts(),
        "Discord Transport": await test_discord_transport()
    }
    
    # Print summary
    success = print_test_summary(results)
    
    if success:
        print("\\nNext steps:")
        print("1. Set up model files: python setup_models.py")
        print("2. Run full pipeline: python voice_bot_v2.py")
    else:
        print("\\nNext steps:")
        print("1. Fix failed tests")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Set VOICE_BOT_TOKEN environment variable")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)