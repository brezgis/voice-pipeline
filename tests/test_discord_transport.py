#!/usr/bin/env python3
"""
Tests for Discord Transport Adapter

Tests the Discord transport's ability to:
1. Initialize properly with correct configuration
2. Connect to Discord
3. Join voice channels  
4. Handle audio format conversions
5. Integrate with Pipecat pipeline

Run with: python test_discord_transport.py
Requires VOICE_BOT_TOKEN environment variable.
"""

import asyncio
import os
import sys
import numpy as np
from unittest.mock import Mock, AsyncMock

# Add the components directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components'))

from discord_transport import (
    DiscordTransport,
    DiscordAudioSink, 
    DiscordAudioSource,
    AudioBuffer,
    pcm_discord_to_processing,
    pcm_processing_to_discord,
    DISCORD_FRAME_SIZE,
    DISCORD_SAMPLE_RATE,
    DISCORD_CHANNELS,
    PROCESSING_SAMPLE_RATE,
    PROCESSING_CHANNELS
)

# Test configuration
TEST_GUILD_ID = 1465514323291144377
TEST_USER_ID = 1411361963308613867


class TestDiscordTransport:
    """Test suite for Discord transport."""
    
    @staticmethod
    def test_audio_conversion():
        """Test audio format conversion functions."""
        print("Testing audio format conversion...")
        
        # Generate test audio: 20ms of random Discord format audio
        discord_samples = np.random.randint(-32768, 32767, DISCORD_FRAME_SIZE // 2, dtype=np.int16)
        discord_audio = discord_samples.tobytes()
        
        # Test Discord -> Processing conversion
        processed_audio = pcm_discord_to_processing(discord_audio)
        assert len(processed_audio) > 0
        
        # Should be downsampled by 3x and converted to mono float32
        expected_samples = (DISCORD_FRAME_SIZE // 2 // 2) // 3  # stereo->mono, then 48k->16k
        actual_samples = len(processed_audio) // 4  # float32 = 4 bytes
        
        print(f"  Discord frame: {len(discord_audio)} bytes")
        print(f"  Processed frame: {len(processed_audio)} bytes ({actual_samples} samples)")
        
        # Test Processing -> Discord conversion (16kHz)
        back_to_discord = pcm_processing_to_discord(processed_audio, PROCESSING_SAMPLE_RATE)
        assert len(back_to_discord) > 0
        
        # Should be upsampled by 3x and converted to stereo int16
        print(f"  Back to Discord: {len(back_to_discord)} bytes")
        
        # Test 24kHz input (common TTS output rate)
        tts_samples = np.random.uniform(-1, 1, 480).astype(np.float32)  # 20ms at 24kHz
        tts_audio = tts_samples.tobytes()
        discord_from_tts = pcm_processing_to_discord(tts_audio, 24000)
        assert len(discord_from_tts) > 0
        print(f"  24kHz TTS -> Discord: {len(tts_audio)} -> {len(discord_from_tts)} bytes")
        
        print("  ✅ Audio conversion tests passed")

    @staticmethod
    def test_audio_buffer():
        """Test the VAD audio buffer logic."""
        print("Testing audio buffer VAD...")
        
        buffer = AudioBuffer()
        
        # Test silence (should not trigger)
        silence = np.zeros(960, dtype=np.int16).tobytes()  # 20ms silence
        for _ in range(10):
            assert not buffer.add_frame(silence)
        
        # Test speech (loud audio)
        speech = (np.random.randint(-10000, 10000, 960, dtype=np.int16)).tobytes()
        
        # Add speech frames
        for i in range(100):  # ~2 seconds of speech
            result = buffer.add_frame(speech)
            if result:
                print(f"  Speech detected after {i+1} frames")
                break
        
        # Should have accumulated audio
        audio_data = buffer.get_audio()
        assert len(audio_data) > 0
        print(f"  Accumulated {len(audio_data)} bytes of audio")
        
        # Reset should clear everything
        buffer.reset()
        assert len(buffer.get_audio()) == 0
        assert not buffer.is_speaking
        
        print("  ✅ Audio buffer tests passed")

    @staticmethod
    def test_transport_initialization():
        """Test transport initialization and configuration."""
        print("Testing transport initialization...")
        
        token = os.getenv('VOICE_BOT_TOKEN', 'test_token')
        
        transport = DiscordTransport(
            bot_token=token,
            guild_id=TEST_GUILD_ID,
            auto_join_user_id=TEST_USER_ID,
            name="test_transport"
        )
        
        # Check configuration
        assert transport.bot_token == token
        assert transport.guild_id == TEST_GUILD_ID
        assert transport.auto_join_user_id == TEST_USER_ID
        assert transport.name == "test_transport"
        
        # Check initial state
        assert transport.voice_client is None
        assert transport.audio_sink is None
        assert transport.audio_source is None
        assert len(transport.audio_buffers) == 0
        
        # Check Discord bot configuration
        assert transport.bot is not None
        assert transport.bot.intents.voice_states
        assert transport.bot.intents.guilds
        
        print("  ✅ Transport initialization tests passed")

    @staticmethod
    async def test_discord_connection():
        """Test actual Discord connection (requires token)."""
        token = os.getenv('VOICE_BOT_TOKEN')
        if not token:
            print("Skipping Discord connection test (no VOICE_BOT_TOKEN)")
            return
            
        print("Testing Discord connection...")
        
        transport = DiscordTransport(
            bot_token=token,
            guild_id=TEST_GUILD_ID,
            auto_join_user_id=TEST_USER_ID
        )
        
        try:
            # Start the transport (this will connect to Discord)
            # Use asyncio.wait_for with timeout to avoid hanging
            await asyncio.wait_for(transport.bot.login(token), timeout=10.0)
            print(f"  ✅ Successfully logged into Discord as {transport.bot.user}")
            
            # Check if we can access the guild
            guild = transport.bot.get_guild(TEST_GUILD_ID)
            if guild:
                print(f"  ✅ Found guild: {guild.name}")
                
                # List voice channels
                voice_channels = guild.voice_channels
                print(f"  Found {len(voice_channels)} voice channel(s)")
                for channel in voice_channels:
                    print(f"    - {channel.name} ({len(channel.members)} members)")
                    
            else:
                print(f"  ⚠️ Guild {TEST_GUILD_ID} not found (bot not invited?)")
                
        except asyncio.TimeoutError:
            print("  ⚠️ Discord connection timed out")
        except Exception as e:
            print(f"  ⚠️ Discord connection failed: {e}")
        finally:
            # Clean up
            if not transport.bot.is_closed():
                await transport.bot.close()

    @staticmethod
    def test_audio_sink():
        """Test the Discord audio sink."""
        print("Testing audio sink...")
        
        # Mock transport
        transport = Mock()
        transport.audio_buffers = {}
        transport.bot = Mock()
        transport.bot.loop = Mock()
        
        # Create sink
        sink = DiscordAudioSink(transport)
        
        # Test write method (simulates Discord calling it)
        test_audio = np.random.randint(-1000, 1000, 960, dtype=np.int16).tobytes()
        test_user_id = 123456
        
        # Mock the audio buffer
        mock_buffer = Mock()
        mock_buffer.add_frame.return_value = True  # Simulate completed speech
        mock_buffer.get_audio.return_value = test_audio
        transport.audio_buffers = {test_user_id: mock_buffer}
        
        # Test write
        sink.write(test_audio, test_user_id)
        
        # Should have called buffer methods
        mock_buffer.add_frame.assert_called_once_with(test_audio)
        mock_buffer.get_audio.assert_called_once()
        mock_buffer.reset.assert_called_once()
        
        # Should have scheduled async callback
        transport.bot.loop.call_soon_threadsafe.assert_called_once()
        
        print("  ✅ Audio sink tests passed")

    @staticmethod
    def test_audio_source():
        """Test the Discord audio source."""
        print("Testing audio source...")
        
        transport = Mock()
        source = DiscordAudioSource(transport)
        
        # Test initial state
        assert not source.is_opus()
        
        # Test reading when empty (should return silence)
        frame = source.read()
        assert len(frame) == DISCORD_FRAME_SIZE
        assert frame == b'\x00' * DISCORD_FRAME_SIZE
        
        # Feed some audio
        test_audio = np.random.randint(-1000, 1000, DISCORD_FRAME_SIZE // 2, dtype=np.int16).tobytes()
        source.feed_audio(test_audio)
        
        # Should now return the audio
        frame = source.read()
        assert len(frame) == DISCORD_FRAME_SIZE
        assert frame == test_audio
        
        # Mark as finished
        source.finish()
        
        # Should now return empty (end of stream)
        frame = source.read()
        assert frame == b''
        
        print("  ✅ Audio source tests passed")


async def run_all_tests():
    """Run all tests."""
    print("🧪 Running Discord Transport Tests\n")
    
    # Synchronous tests
    TestDiscordTransport.test_audio_conversion()
    print()
    
    TestDiscordTransport.test_audio_buffer()
    print()
    
    TestDiscordTransport.test_transport_initialization()
    print()
    
    TestDiscordTransport.test_audio_sink()
    print()
    
    TestDiscordTransport.test_audio_source()
    print()
    
    # Asynchronous test (requires token)
    await TestDiscordTransport.test_discord_connection()
    print()
    
    print("🎉 All tests completed!")


async def test_voice_channel_join():
    """Test joining a voice channel (if user is in one)."""
    token = os.getenv('VOICE_BOT_TOKEN')
    if not token:
        print("Skipping voice channel test (no VOICE_BOT_TOKEN)")
        return
        
    print("Testing voice channel join...")
    
    transport = DiscordTransport(
        bot_token=token,
        guild_id=TEST_GUILD_ID,
        auto_join_user_id=TEST_USER_ID
    )
    
    try:
        # Login to Discord
        await transport.bot.login(token)
        print(f"Logged in as {transport.bot.user}")
        
        # Try to find target user in a voice channel
        guild = transport.bot.get_guild(TEST_GUILD_ID)
        if not guild:
            print("Guild not found")
            return
            
        target_member = guild.get_member(TEST_USER_ID)
        if not target_member:
            print("Target user not found in guild")
            return
            
        if not target_member.voice or not target_member.voice.channel:
            print("Target user not in a voice channel")
            return
            
        print(f"Found target user in voice channel: {target_member.voice.channel.name}")
        
        # Try to join the channel
        await transport._join_voice_channel(target_member.voice.channel)
        
        if transport.voice_client and transport.voice_client.is_connected():
            print("✅ Successfully connected to voice channel")
            print(f"Audio sink active: {transport.audio_sink is not None}")
            print(f"Audio source ready: {transport.audio_source is not None}")
            
            # Wait a moment then disconnect
            await asyncio.sleep(2)
            await transport._leave_voice_channel()
            print("✅ Successfully disconnected from voice channel")
        else:
            print("❌ Failed to connect to voice channel")
            
    except Exception as e:
        print(f"Voice channel test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await transport.stop()


if __name__ == "__main__":
    # Check if we're being run directly
    if len(sys.argv) > 1 and sys.argv[1] == "voice":
        # Test voice channel functionality
        asyncio.run(test_voice_channel_join())
    else:
        # Run standard tests
        asyncio.run(run_all_tests())
        
        print("\n" + "="*50)
        print("To test voice channel functionality, run:")
        print("python test_discord_transport.py voice")
        print("\nMake sure Anna is in a voice channel first!")
        print("="*50)