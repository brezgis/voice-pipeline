#!/usr/bin/env python3
"""
Discord Transport for Pipecat v2

Extends BaseTransport with proper input()/output() methods returning
BaseInputTransport/BaseOutputTransport, bridging py-cord's Discord voice I/O
with Pipecat's frame pipeline.

Audio flow:
  Discord mic (48kHz stereo) → resample → InputAudioRawFrame (16kHz mono) → pipeline
  pipeline → OutputAudioRawFrame → resample → Discord speaker (48kHz stereo)

Threading model:
  Discord's voice recv/send runs in separate threads. We use asyncio queues
  and call_soon_threadsafe() to bridge safely to the pipeline's event loop.
"""

import asyncio
import threading
from typing import Optional

import discord
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

# =============================================================================
# Audio constants
# =============================================================================

DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_FRAME_BYTES = 3840  # 20ms at 48kHz stereo int16

PIPELINE_SAMPLE_RATE = 16000
PIPELINE_CHANNELS = 1

# Comfortable listening level — avoids clipping and earblasting.
# Range: 0.0 (silent) to 1.0 (full volume).
DEFAULT_OUTPUT_VOLUME = 0.3

# Log audio sink/feeder stats every N frames (~20ms each, so 500 ≈ 10s)
LOG_EVERY_N_FRAMES = 500


# =============================================================================
# Resampling functions
# =============================================================================

def resample_discord_to_pipeline(pcm_48k_stereo: bytes) -> bytes:
    """Convert Discord PCM (48kHz stereo int16) → pipeline (16kHz mono int16).

    Steps: stereo→mono by averaging, then downsample 3:1.
    """
    samples = np.frombuffer(pcm_48k_stereo, dtype=np.int16)
    if len(samples) == 0:
        return b""
    # Stereo to mono: average channels
    stereo = samples.reshape(-1, 2)
    mono = ((stereo[:, 0].astype(np.int32) + stereo[:, 1].astype(np.int32)) // 2).astype(np.int16)
    # Downsample 48kHz → 16kHz (3:1)
    downsampled = mono[::3]
    return downsampled.tobytes()


def resample_pipeline_to_discord(
    pcm_data: bytes,
    source_rate: int,
    volume: float = DEFAULT_OUTPUT_VOLUME,
) -> bytes:
    """Convert pipeline PCM (int16 mono) → Discord (48kHz stereo int16).

    Handles 16kHz (3x upsample) and 24kHz (2x upsample) source rates.
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    if len(samples) == 0:
        return b""

    # Upsample to 48kHz
    if source_rate == 16000:
        upsampled = np.repeat(samples, 3)
    elif source_rate == 24000:
        upsampled = np.repeat(samples, 2)
    elif source_rate == 48000:
        upsampled = samples
    else:
        ratio = 48000 / source_rate
        indices = np.arange(0, len(samples), 1.0 / ratio)[:int(len(samples) * ratio)]
        indices = np.clip(indices, 0, len(samples) - 1)
        upsampled = np.interp(
            indices, np.arange(len(samples)), samples.astype(np.float64)
        ).astype(np.int16)

    # Apply volume scaling
    scaled = (upsampled.astype(np.float32) * volume).clip(-32768, 32767).astype(np.int16)

    # Mono → stereo: interleave same data into both channels
    stereo = np.empty(len(scaled) * 2, dtype=np.int16)
    stereo[0::2] = scaled
    stereo[1::2] = scaled
    return stereo.tobytes()


# =============================================================================
# Discord audio sink/source (thread-safe bridges)
# =============================================================================

class DiscordAudioSink(discord.sinks.Sink):
    """Receives audio from Discord voice channel, forwards to input transport."""

    def __init__(self, input_transport: "DiscordInputTransport"):
        super().__init__()
        self._input_transport = input_transport
        self._write_count: int = 0

    def write(self, data: bytes, user: int) -> None:
        """Called from Discord's decode thread for each audio packet."""
        self._write_count += 1

        if user is None or not data:
            return

        pipeline_audio = resample_discord_to_pipeline(data)
        if not pipeline_audio:
            return

        frame = InputAudioRawFrame(
            audio=pipeline_audio,
            sample_rate=PIPELINE_SAMPLE_RATE,
            num_channels=PIPELINE_CHANNELS,
        )

        loop = self._input_transport._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(
                self._input_transport._audio_queue_sync.put_nowait, frame
            )

    def cleanup(self) -> None:
        pass


class DiscordAudioSource(discord.AudioSource):
    """Plays audio to Discord voice channel from output transport's buffer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._finished = False

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        """Called every 20ms from Discord's send thread.

        Must return exactly 3840 bytes of 48kHz stereo int16 PCM.
        """
        with self._lock:
            if len(self._buffer) >= DISCORD_FRAME_BYTES:
                frame = bytes(self._buffer[:DISCORD_FRAME_BYTES])
                del self._buffer[:DISCORD_FRAME_BYTES]
                return frame
            elif self._finished and len(self._buffer) > 0:
                frame = bytes(self._buffer) + b"\x00" * (DISCORD_FRAME_BYTES - len(self._buffer))
                self._buffer.clear()
                return frame
            elif self._finished:
                return b""
            else:
                return b"\x00" * DISCORD_FRAME_BYTES

    def feed(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._finished = False

    def cleanup(self) -> None:
        pass


# =============================================================================
# Pipecat transports
# =============================================================================

class DiscordInputTransport(BaseInputTransport):
    """Pipecat input transport that receives audio from Discord."""

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._audio_queue_sync: asyncio.Queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._feeder_task: Optional[asyncio.Task] = None
        self._feed_count: int = 0

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        # Pipecat creates _audio_in_queue in set_transport_ready(), which may
        # not be called automatically for custom transports. Without this,
        # all audio frames are silently dropped.
        await self.set_transport_ready(frame)
        self._loop = asyncio.get_running_loop()
        self._feeder_task = self.create_task(self._feeder_loop())

    async def stop(self, frame: EndFrame) -> None:
        if self._feeder_task:
            await self.cancel_task(self._feeder_task)
            self._feeder_task = None
        await super().stop(frame)

    async def _feeder_loop(self) -> None:
        """Drain audio from the sync queue and push into the pipeline."""
        logger.info("Audio feeder loop started")
        while True:
            try:
                frame = await asyncio.wait_for(
                    self._audio_queue_sync.get(), timeout=1.0
                )
                self._feed_count += 1
                await self.push_audio_frame(frame)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DiscordInputTransport feeder error: {e}")


class DiscordOutputTransport(BaseOutputTransport):
    """Pipecat output transport that plays audio to Discord."""

    def __init__(
        self,
        params: TransportParams,
        audio_source: DiscordAudioSource,
        transport: "DiscordTransport" = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._audio_source = audio_source
        self._transport = transport
        self._playback_started = False

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        # Pipecat creates media senders in set_transport_ready(), which may
        # not be called automatically for custom transports. Without this,
        # TTS audio frames are silently dropped.
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write audio frame to Discord. Called by the base class's media sender."""
        if not self._audio_source:
            return False

        if self._transport and self._transport._voice_client:
            vc = self._transport._voice_client

            if not self._playback_started:
                # Start playback on first audio frame (avoids green outline when idle)
                if not vc.is_playing():
                    self._audio_source.reset()
                    vc.play(self._audio_source)
                    self._playback_started = True
            elif not vc.is_playing():
                # Playback stopped (previous TTS finished) — restart for new audio.
                # Without this reset, the bot goes silent after the first response.
                self._audio_source.reset()
                vc.play(self._audio_source)

        discord_audio = resample_pipeline_to_discord(frame.audio, frame.sample_rate)
        self._audio_source.feed(discord_audio)
        return True


# =============================================================================
# Main transport
# =============================================================================

class DiscordTransport(BaseTransport):
    """Full Discord transport implementing BaseTransport.

    Creates a Discord bot, manages voice channel connections,
    and provides input()/output() frame processors for the pipeline.
    """

    def __init__(
        self,
        *,
        bot_token: str,
        guild_id: int = 0,
        auto_join_user_id: int = 0,
        params: Optional[TransportParams] = None,
    ):
        super().__init__(
            name="DiscordTransport",
            input_name="DiscordInput",
            output_name="DiscordOutput",
        )

        self._bot_token = bot_token
        self._guild_id = guild_id
        self._auto_join_user_id = auto_join_user_id

        self._params = params or TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_in_channels=PIPELINE_CHANNELS,
            audio_out_enabled=True,
            audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
            audio_out_channels=PIPELINE_CHANNELS,
        )

        # Discord bot with reconnect enabled
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        self._bot = discord.Bot(intents=intents)
        self._voice_client: Optional[discord.VoiceClient] = None

        # Audio I/O
        self._audio_source = DiscordAudioSource()
        self._audio_sink: Optional[DiscordAudioSink] = None

        # Pipecat transports
        self._input_transport = DiscordInputTransport(
            self._params, name="DiscordInput"
        )
        self._output_transport = DiscordOutputTransport(
            self._params, self._audio_source, transport=self, name="DiscordOutput"
        )

        self._setup_events()

    def input(self) -> FrameProcessor:
        return self._input_transport

    def output(self) -> FrameProcessor:
        return self._output_transport

    @property
    def bot(self) -> discord.Bot:
        return self._bot

    def _setup_events(self) -> None:
        @self._bot.event
        async def on_ready():
            logger.info(f"Discord bot connected as {self._bot.user}")
            if self._auto_join_user_id:
                await self._try_auto_join()

        @self._bot.event
        async def on_voice_state_update(member, before, after):
            # Auto-join when target user joins voice
            if (
                self._auto_join_user_id
                and member.id == self._auto_join_user_id
                and after.channel
                and not self._voice_client
            ):
                await self._join_channel(after.channel)

            # Auto-leave when no humans remain
            if before.channel and (after.channel is None or after.channel != before.channel):
                if self._voice_client and self._voice_client.channel == before.channel:
                    humans = [m for m in before.channel.members if not m.bot]
                    if not humans:
                        await self._leave_channel()

        @self._bot.event
        async def on_disconnect():
            logger.warning("Discord bot disconnected — py-cord will auto-reconnect")
            self._voice_client = None
            self._audio_sink = None

        @self._bot.event
        async def on_resumed():
            logger.info("Discord bot resumed — attempting to rejoin voice")
            if self._auto_join_user_id:
                await self._try_auto_join()

    async def _try_auto_join(self) -> None:
        """Find the target user and join their voice channel."""
        for guild in self._bot.guilds:
            member = guild.get_member(self._auto_join_user_id)
            if member and member.voice and member.voice.channel:
                await self._join_channel(member.voice.channel)
                return

    async def _join_channel(self, channel: discord.VoiceChannel) -> None:
        """Join a voice channel, start recording and playback."""
        try:
            if self._voice_client:
                await self._leave_channel()

            logger.info(f"Joining voice channel: {channel.name}")
            self._voice_client = await channel.connect()

            # Setup recording sink → feeds our input transport
            self._audio_sink = DiscordAudioSink(self._input_transport)

            async def _on_stop(*args, **kwargs):
                pass

            self._voice_client.start_recording(self._audio_sink, _on_stop, None)

            # Don't start playback yet — we'll start it when TTS has audio to play.
            # This avoids the green "speaking" outline when idle.
            self._audio_source.reset()

            logger.info("Discord voice connected and streaming")

        except Exception as e:
            logger.error(f"Error joining voice channel: {e}")

    async def _leave_channel(self) -> None:
        """Leave voice channel and clean up."""
        if self._voice_client:
            try:
                self._voice_client.stop_recording()
            except Exception:
                pass
            try:
                await self._voice_client.disconnect()
            except Exception:
                pass
            self._voice_client = None
            self._audio_sink = None
            logger.info("Left voice channel")

    async def start_bot(self) -> None:
        """Start the Discord bot."""
        await self._bot.start(self._bot_token)

    async def stop_bot(self) -> None:
        """Stop the Discord bot."""
        await self._leave_channel()
        if not self._bot.is_closed():
            await self._bot.close()
