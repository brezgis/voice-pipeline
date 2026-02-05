#!/usr/bin/env python3
"""
Kyutai TTS 1.6B Service for Pipecat

This service integrates the Kyutai TTS 1.6B model for high-quality streaming text-to-speech.
The model supports true streaming generation, starting audio output before text is complete.

Key features:
- 1.6B parameter model with Mimi codec for high quality
- Streaming TTS: can start playing audio before full text is processed
- English/French support
- Voice conditioning through pre-computed embeddings
- CUDA acceleration with float16 precision

Technical details:
- Uses the moshi library for model inference
- Requires Kyutai TTS model files and voice embeddings
- Outputs 24kHz mono audio that gets resampled for Discord
- VRAM usage: ~5GB
"""

import asyncio
import os
import tempfile
from typing import Optional, Dict, Any, AsyncIterator
import numpy as np
import torch

# Pipecat imports
from pipecat.frames.frames import (
    Frame,
    TextFrame, 
    TTSStartedFrame,
    TTSStoppedFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    LLMTextFrame
)
from pipecat.services.tts_service import TTSService

# Moshi/Kyutai imports
try:
    from moshi import models
    from moshi.models.loaders import load_model_bundle 
    from moshi.models.generation import Generators
    import safetensors.torch
except ImportError:
    print("Warning: moshi package not available. Install with: pip install moshi")
    models = None


class KyutaiTTSService(TTSService):
    """
    Kyutai TTS service for Pipecat pipeline
    
    Provides streaming text-to-speech using the Kyutai TTS 1.6B model
    """
    
    def __init__(
        self,
        model_path: str,
        voice_model_path: str = None,
        voice_name: str = "default",
        device: str = "cuda",
        sample_rate: int = 24000,
        streaming: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if models is None:
            raise ImportError("moshi package is required for Kyutai TTS")
            
        self.model_path = model_path
        self.voice_model_path = voice_model_path
        self.voice_name = voice_name
        self.device = device
        self.sample_rate = sample_rate
        self.streaming = streaming
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.voice_embed = None
        self._model_loaded = False
        
        # Generation parameters
        self.generation_params = {
            'n_q': 32,  # Number of quantization levels
            'temp': 0.8,  # Temperature for sampling
            'cfg_coef': 1.0,  # Classifier-free guidance
            'padding_between': 0.0,  # Silence between segments
        }
        
    async def _load_model(self):
        """Load the Kyutai TTS model and voice embeddings"""
        if self._model_loaded:
            return
            
        print(f"Loading Kyutai TTS model from {self.model_path}")
        
        try:
            # Load model bundle (this includes the TTS model and tokenizers)
            bundle = load_model_bundle(self.model_path, device=self.device)
            self.model = bundle.model
            self.tokenizer = bundle.tokenizer
            
            # Load voice embeddings if provided
            if self.voice_model_path:
                print(f"Loading voice model: {self.voice_model_path}")
                voice_data = safetensors.torch.load_file(self.voice_model_path)
                self.voice_embed = voice_data.get('voice_embed')
                
                if self.voice_embed is not None:
                    self.voice_embed = self.voice_embed.to(self.device)
                    print(f"Voice embedding loaded: {self.voice_embed.shape}")
                else:
                    print("Warning: No voice embedding found in voice model file")
            
            # Set model to eval mode
            self.model.eval()
            
            # Enable CUDA optimizations if available
            if self.device == "cuda" and torch.cuda.is_available():
                # Use float16 for memory efficiency (Blackwell compatible)
                self.model = self.model.half()
                print(f"Model loaded on {self.device} with float16 precision")
            
            self._model_loaded = True
            print("Kyutai TTS model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Kyutai TTS model: {e}")
            raise
    
    async def _generate_speech_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Generate speech audio with streaming output
        
        This yields audio chunks as they're generated, enabling low-latency playback
        """
        await self._load_model()
        
        try:
            # Tokenize input text
            tokens = self.tokenizer.encode(text)
            tokens = torch.tensor([tokens], device=self.device)
            
            # Create generation context
            generators = Generators(
                model=self.model,
                device=self.device,
                **self.generation_params
            )
            
            # Generate audio tokens with streaming
            if self.streaming:
                # Stream generation: yield chunks as they're generated
                audio_chunks = []
                
                for chunk in generators.streaming_generate(
                    tokens=tokens,
                    voice_embed=self.voice_embed,
                    chunk_length_ms=500  # 500ms chunks for low latency
                ):
                    # Decode audio chunk
                    audio_chunk = chunk.cpu().numpy().astype(np.float32)
                    audio_chunks.append(audio_chunk)
                    
                    # Yield audio chunk immediately
                    yield audio_chunk.tobytes()
                    
            else:
                # Non-streaming: generate full audio then yield chunks
                audio_tokens = generators.generate(
                    tokens=tokens,
                    voice_embed=self.voice_embed
                )
                
                # Decode full audio
                audio = audio_tokens.cpu().numpy().astype(np.float32)
                
                # Chunk and yield
                chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    yield chunk.tobytes()
                    
        except Exception as e:
            print(f"Error generating speech with Kyutai TTS: {e}")
            # Return silence on error
            silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            yield silence.tobytes()
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames from the pipeline"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, (TextFrame, LLMTextFrame)):
            # Extract text
            if isinstance(frame, TextFrame):
                text = frame.text
            else:
                text = frame.text
                
            if not text or not text.strip():
                return
                
            # Send TTS started frame
            await self.push_frame(TTSStartedFrame(), direction)
            
            try:
                # Generate speech with streaming
                async for audio_chunk in self._generate_speech_streaming(text):
                    if audio_chunk:
                        # Create audio frame
                        audio_frame = OutputAudioRawFrame(
                            audio=audio_chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1  # Mono output
                        )
                        await self.push_frame(audio_frame, direction)
                        
            except Exception as e:
                print(f"TTS generation error: {e}")
                
            finally:
                # Send TTS stopped frame
                await self.push_frame(TTSStoppedFrame(), direction)
    
    async def start(self):
        """Start the TTS service"""
        await self._load_model()
        
    async def run_tts(self, text: str) -> AsyncIterator[Frame]:
        """
        Required abstract method for TTSService
        Generate TTS frames for the given text
        """
        if not text.strip():
            return
            
        # Send start frame
        yield TTSStartedFrame()
        
        try:
            # Generate speech with streaming
            async for audio_chunk in self._generate_speech_streaming(text):
                if audio_chunk:
                    # Create audio frame
                    audio_frame = OutputAudioRawFrame(
                        audio=audio_chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1  # Mono output
                    )
                    yield audio_frame
                    
        except Exception as e:
            print(f"TTS generation error: {e}")
            
        finally:
            # Send stop frame
            yield TTSStoppedFrame()
    
    async def stop(self):
        """Stop the TTS service and cleanup"""
        if self.model:
            # Move model to CPU to free GPU memory
            self.model = self.model.cpu()
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._model_loaded = False
        print("Kyutai TTS service stopped")


class KyutaiTTSServiceSimple(TTSService):
    """
    Simplified version using moshi CLI tools
    
    This version shells out to the moshi command-line tool instead of
    using the Python API directly. Useful as a fallback if the API
    integration has issues.
    """
    
    def __init__(
        self,
        model_repo: str = "kyutai/tts-1.6b-en_fr",
        voice_name: str = "default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_repo = model_repo
        self.voice_name = voice_name
        
    async def process_frame(self, frame: Frame, direction):
        """Process frames using moshi CLI"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, (TextFrame, LLMTextFrame)):
            text = frame.text if isinstance(frame, TextFrame) else frame.text
            
            if not text or not text.strip():
                return
                
            await self.push_frame(TTSStartedFrame(), direction)
            
            try:
                # Use temporary file for output
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    output_path = tmp_file.name
                    
                # Run moshi TTS command
                cmd = [
                    'python', '-m', 'moshi.run_inference',
                    '--hf-repo', self.model_repo,
                    '--text', text,
                    '--output', output_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0 and os.path.exists(output_path):
                    # Read generated audio file
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(output_path)
                    
                    # Convert to the expected format
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                        
                    # Send as audio frame
                    audio_frame = OutputAudioRawFrame(
                        audio=audio_data.tobytes(),
                        sample_rate=sample_rate,
                        num_channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1]
                    )
                    await self.push_frame(audio_frame, direction)
                    
                    # Cleanup temp file
                    os.unlink(output_path)
                    
                else:
                    print(f"Moshi TTS command failed: {stderr.decode() if stderr else 'Unknown error'}")
                    
            except Exception as e:
                print(f"Error with moshi CLI TTS: {e}")
                
            finally:
                await self.push_frame(TTSStoppedFrame(), direction)
    
    async def run_tts(self, text: str) -> AsyncIterator[Frame]:
        """
        Required abstract method for TTSService
        Generate TTS frames using moshi CLI
        """
        if not text.strip():
            return
            
        yield TTSStartedFrame()
        
        try:
            # Use temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
                
            # Run moshi TTS command
            cmd = [
                'python', '-m', 'moshi.run_inference',
                '--hf-repo', self.model_repo,
                '--text', text,
                '--output', output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and os.path.exists(output_path):
                # Read generated audio file
                import soundfile as sf
                audio_data, sample_rate = sf.read(output_path)
                
                # Convert to the expected format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    
                # Send as audio frame
                audio_frame = OutputAudioRawFrame(
                    audio=audio_data.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1]
                )
                yield audio_frame
                
                # Cleanup temp file
                os.unlink(output_path)
                
            else:
                print(f"Moshi TTS command failed: {stderr.decode() if stderr else 'Unknown error'}")
                
        except Exception as e:
            print(f"Error with moshi CLI TTS: {e}")
            
        finally:
            yield TTSStoppedFrame()


# Test function
async def test_kyutai_tts():
    """Test Kyutai TTS service"""
    print("=== Testing Kyutai TTS Service ===")
    
    # Test with simple version first (requires moshi to be installed)
    try:
        service = KyutaiTTSServiceSimple()
        print("KyutaiTTSServiceSimple created successfully")
        
        # Note: Full testing requires model files
        print("To test fully, download model files from:")
        print("https://huggingface.co/kyutai/tts-1.6b-en_fr")
        print("https://huggingface.co/kyutai/tts-voices")
        
    except Exception as e:
        print(f"Error creating service: {e}")

if __name__ == "__main__":
    asyncio.run(test_kyutai_tts())