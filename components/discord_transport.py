#!/usr/bin/env python3
"""
Discord Transport Adapter for Pipecat

This is the novel component that bridges py-cord's Discord voice I/O with Pipecat's frame protocol.
Since Pipecat doesn't have native Discord support, we create a custom transport that:

1. Handles Discord bot connection and voice channel lifecycle
2. Converts Discord's 48kHz stereo PCM to Pipecat's preferred audio format
3. Manages the bidirectional audio flow between Discord and the pipeline
4. Provides proper async integration with Pipecat's frame processing

Technical Notes:
- Discord uses Opus codec internally, py-cord provides PCM decoded frames
- We need to handle resampling: 48kHz stereo ↔ 16kHz mono for processing
- Must manage timing and buffering to avoid audio dropouts
"""

import asyncio
import os
import numpy as np
from typing import Optional, Callable, Awaitable
import discord
from discord.ext import commands

# Pipecat imports
from pipecat.frames.frames import (
    Frame, 
    AudioRawFrame, 
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame, 
    EndFrame,
    CancelFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.base_input import BaseInputTransport  
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.audio.audio_buffer_processor import (
    create_stream_resampler, 
    interleave_stereo_audio
)

# Audio configuration
DISCORD_SAMPLE_RATE = 48000  # Discord's audio sample rate
DISCORD_CHANNELS = 2         # Stereo
PROCESSING_SAMPLE_RATE = 16000  # Preferred for STT/processing
PROCESSING_CHANNELS = 1      # Mono

class DiscordAudioSink(discord.AudioSink):
    """
    Audio sink for receiving audio from Discord voice channel
    Converts incoming audio to Pipecat frames
    """
    
    def __init__(self, transport: 'DiscordTransport'):
        super().__init__()
        self.transport = transport
        self.resampler = create_stream_resampler(
            input_sample_rate=DISCORD_SAMPLE_RATE,
            output_sample_rate=PROCESSING_SAMPLE_RATE,
            num_channels=DISCORD_CHANNELS,
            output_channels=PROCESSING_CHANNELS
        )
        
    def on_audio(self, user, data):
        """Called when audio data is received from Discord"""
        if data and len(data) > 0:
            # Convert from Discord's format to processing format
            # data is bytes (PCM, 48kHz, stereo, signed 16-bit LE)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz mono
            processed_audio = self.resampler.resample(audio_np)
            
            # Create Pipecat audio frame
            frame = InputAudioRawFrame(
                audio=processed_audio.tobytes(),
                sample_rate=PROCESSING_SAMPLE_RATE,
                num_channels=PROCESSING_CHANNELS
            )
            
            # Send to pipeline (async)
            asyncio.create_task(self.transport._handle_input_frame(frame))

class DiscordAudioSource:
    """
    Audio source for sending audio to Discord voice channel
    Receives processed audio from Pipecat pipeline
    """
    
    def __init__(self, transport: 'DiscordTransport'):
        self.transport = transport
        self.voice_client: Optional[discord.VoiceClient] = None
        self.audio_queue = asyncio.Queue()
        self.resampler = create_stream_resampler(
            input_sample_rate=PROCESSING_SAMPLE_RATE,
            output_sample_rate=DISCORD_SAMPLE_RATE,
            num_channels=PROCESSING_CHANNELS,
            output_channels=DISCORD_CHANNELS
        )
        self.playing = False
        
    async def queue_audio(self, frame: OutputAudioRawFrame):
        """Queue audio frame for playback to Discord"""
        await self.audio_queue.put(frame)
        
        # Start playback if not already playing
        if not self.playing and self.voice_client:
            await self._start_playback()
    
    async def _start_playback(self):
        """Start audio playback to Discord"""
        if self.playing or not self.voice_client:
            return
            
        self.playing = True
        
        try:
            # Create audio source that reads from our queue
            audio_source = DiscordPCMAudioSource(self)
            self.voice_client.play(audio_source)
            
        except Exception as e:
            print(f"Error starting Discord audio playback: {e}")
            self.playing = False
            
    async def get_audio_chunk(self) -> Optional[bytes]:
        """Get next audio chunk for Discord playback"""
        try:
            # Wait for audio with timeout to avoid blocking Discord
            frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.02)
            
            # Convert from processing format to Discord format
            audio_np = np.frombuffer(frame.audio, dtype=np.float32)
            
            # Resample to Discord format (48kHz stereo)
            discord_audio = self.resampler.resample(audio_np)
            
            # Convert to int16 and return as bytes
            discord_audio_int = (discord_audio * 32767).astype(np.int16)
            return discord_audio_int.tobytes()
            
        except asyncio.TimeoutError:
            # No audio available, return silence
            silence_samples = int(DISCORD_SAMPLE_RATE * 0.02 * DISCORD_CHANNELS)  # 20ms of silence
            silence = np.zeros(silence_samples, dtype=np.int16)
            return silence.tobytes()

class DiscordPCMAudioSource(discord.AudioSource):
    """Discord audio source that reads from our processed audio queue"""
    
    def __init__(self, audio_source: DiscordAudioSource):
        super().__init__()
        self.audio_source = audio_source
        
    def read(self) -> bytes:
        """Read audio data for Discord (blocking call)"""
        # This is called by Discord in a thread, so we need to handle async carefully
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.audio_source.get_audio_chunk())
        finally:
            loop.close()
            
    def is_opus(self) -> bool:
        return False  # We're providing raw PCM

class DiscordTransport(BaseTransport):
    """
    Main Discord transport that integrates with Pipecat pipeline
    """
    
    def __init__(
        self, 
        bot_token: str,
        guild_id: int,
        auto_join_user_id: Optional[int] = None
    ):
        super().__init__()
        self.bot_token = bot_token
        self.guild_id = guild_id
        self.auto_join_user_id = auto_join_user_id
        
        # Discord bot setup
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.voice_client: Optional[discord.VoiceClient] = None
        
        # Audio components
        self.audio_sink: Optional[DiscordAudioSink] = None
        self.audio_source: Optional[DiscordAudioSource] = None
        
        # Setup Discord event handlers
        self._setup_discord_events()
        
    def _setup_discord_events(self):
        """Setup Discord bot event handlers"""
        
        @self.bot.event
        async def on_ready():
            print(f"Discord bot logged in as {self.bot.user}")
            
            # Auto-join voice channel if configured
            if self.auto_join_user_id:
                await self._try_auto_join()
                
        @self.bot.event
        async def on_voice_state_update(member, before, after):
            # Auto-join when target user joins a voice channel
            if (member.id == self.auto_join_user_id and 
                after.channel and 
                not self.voice_client):
                await self._join_voice_channel(after.channel)
                
    async def _try_auto_join(self):
        """Try to auto-join voice channel where target user is"""
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            print(f"Guild {self.guild_id} not found")
            return
            
        # Find the target user
        target_member = guild.get_member(self.auto_join_user_id)
        if not target_member:
            print(f"Auto-join user {self.auto_join_user_id} not found")
            return
            
        # Check if they're in a voice channel
        if target_member.voice and target_member.voice.channel:
            await self._join_voice_channel(target_member.voice.channel)
            
    async def _join_voice_channel(self, channel):
        """Join a voice channel and setup audio"""
        try:
            print(f"Joining voice channel: {channel.name}")
            self.voice_client = await channel.connect()
            
            # Setup audio sink for receiving audio
            self.audio_sink = DiscordAudioSink(self)
            await self.voice_client.create_sink(self.audio_sink)
            
            # Setup audio source for sending audio
            self.audio_source = DiscordAudioSource(self)
            self.audio_source.voice_client = self.voice_client
            
            print("Discord audio setup complete")
            
            # Send start frame to pipeline
            await self._handle_input_frame(StartFrame())
            
        except Exception as e:
            print(f"Error joining voice channel: {e}")
            
    async def _handle_input_frame(self, frame: Frame):
        """Handle frames coming from Discord audio"""
        if hasattr(self, '_input_processor') and self._input_processor:
            await self._input_processor.process_frame(frame, FrameDirection.DOWNSTREAM)
            
    async def _handle_output_frame(self, frame: Frame):
        """Handle frames going to Discord audio"""
        if isinstance(frame, OutputAudioRawFrame) and self.audio_source:
            await self.audio_source.queue_audio(frame)
            
    async def start(self):
        """Start the Discord transport"""
        print("Starting Discord transport...")
        
        # Start Discord bot
        await self.bot.start(self.bot_token)
        
    async def stop(self):
        """Stop the Discord transport"""
        print("Stopping Discord transport...")
        
        if self.voice_client:
            await self.voice_client.disconnect()
            
        await self.bot.close()
        
        # Send end frame
        if hasattr(self, '_input_processor') and self._input_processor:
            await self._handle_input_frame(EndFrame())


# Test function
async def test_discord_transport():
    """Test the Discord transport implementation"""
    token = os.getenv('VOICE_BOT_TOKEN')
    if not token:
        print("VOICE_BOT_TOKEN environment variable not set")
        return
        
    transport = DiscordTransport(
        bot_token=token,
        guild_id=1465514323291144377,
        auto_join_user_id=1411361963308613867
    )
    
    print("Discord transport created successfully")
    # Note: Full testing requires Discord connection

if __name__ == "__main__":
    asyncio.run(test_discord_transport())