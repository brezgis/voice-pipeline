#!/usr/bin/env python3
"""
Streaming Gateway LLM Service for Pipecat v2

Connects to the OpenClaw gateway's OpenAI-compatible /v1/chat/completions
streaming endpoint. Unlike ClawdbotLLMService (which shells out to `openclaw
agent` and blocks until the entire response completes), this service streams
tokens as they arrive, emitting each as an LLMTextFrame.

Pipecat's TTSService has built-in sentence aggregation (aggregate_sentences=True
by default with SimpleTextAggregator). It collects streaming LLMTextFrame tokens
into complete sentences and calls run_tts() per sentence. This means TTS starts
generating audio on the FIRST complete sentence while the LLM keeps generating.

The gateway routes through the real OpenClaw agent with full SOUL.md, memory,
tools, and identity — unlike a direct Anthropic API call which would lose all
OpenClaw context.

Estimated latency savings vs ClawdbotLLMService: 1500-3000ms.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService

# Default voice hint — tells the LLM this is a spoken conversation.
DEFAULT_VOICE_HINT = (
    "[Voice conversation — you're talking live in a Discord voice channel. "
    "Write naturally for speech: contractions, casual phrasing, no markdown/formatting/lists. "
    "Talk like a real person would. Your response will be spoken aloud via TTS.]\n\n"
)

# Transcript output directory (daily files: voice-YYYY-MM-DD.md)
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"

# Gateway endpoint
DEFAULT_GATEWAY_URL = "http://localhost:18789/v1/chat/completions"

# Config file path for fallback token
OPENCLAW_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"


def _load_token() -> str:
    """Load the OpenClaw gateway auth token.

    Priority:
    1. OPENCLAW_TOKEN environment variable
    2. ~/.openclaw/openclaw.json → gateway.auth.token
    """
    token = os.getenv("OPENCLAW_TOKEN")
    if token:
        return token

    try:
        config = json.loads(OPENCLAW_CONFIG_PATH.read_text())
        token = config.get("gateway", {}).get("auth", {}).get("token", "")
        if token:
            return token
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not read OpenClaw config: {e}")

    raise ValueError(
        "No OpenClaw token found. Set OPENCLAW_TOKEN env var or ensure "
        "~/.openclaw/openclaw.json contains gateway.auth.token"
    )


class StreamingGatewayLLMService(AIService):
    """LLM service that streams from the OpenClaw gateway's OpenAI-compatible endpoint.

    Instead of spawning an `openclaw agent` subprocess and waiting for the entire
    response, this POSTs to the gateway with `stream: true` and emits each token
    as an `LLMTextFrame`. Pipecat's TTS sentence aggregator collects tokens into
    sentences and starts TTS on the first complete sentence.

    Flow: TranscriptionFrame → gateway SSE stream → LLMTextFrame tokens → TTS
    """

    def __init__(
        self,
        *,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        token: Optional[str] = None,
        model: str = "openclaw",
        session_id: str = "voice",
        max_tokens: int = 300,
        voice_hint: str = DEFAULT_VOICE_HINT,
        max_history: int = 10,
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gateway_url = gateway_url
        self._token = token or _load_token()
        self._model = model
        self._session_id = session_id
        self._max_tokens = max_tokens
        self._voice_hint = voice_hint
        self._max_history = max_history
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._conversation_history: list[dict] = []

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session (connection-pooled)."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout + 10),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            logger.info(
                f"Gateway LLM session created → {self._gateway_url} "
                f"(model={self._model}, session={self._session_id})"
            )
        return self._session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        text: Optional[str] = None

        if isinstance(frame, TranscriptionFrame):
            text = frame.text
        elif isinstance(frame, TextFrame) and not isinstance(frame, LLMTextFrame):
            text = frame.text
        else:
            await self.push_frame(frame, direction)
            return

        if not text or not text.strip():
            return

        logger.info(f"StreamingGatewayLLM processing: {text[:80]}...")

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self._stream_response(text.strip())
        except Exception as e:
            logger.error(f"StreamingGatewayLLM error: {e}")
            await self.push_frame(
                LLMTextFrame("I'm sorry, I had trouble processing that.")
            )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _stream_response(self, user_text: str) -> None:
        """Stream response from the gateway SSE endpoint, emitting tokens as LLMTextFrames."""
        session = await self._ensure_session()

        # Build messages with voice hint prepended to the user message
        messages = list(self._conversation_history)
        messages.append({
            "role": "user",
            "content": self._voice_hint + user_text,
        })

        payload = {
            "model": self._model,
            "stream": True,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "user": self._session_id,
        }

        t_start = time.monotonic()
        full_response_parts: list[str] = []
        first_token_time: Optional[float] = None

        try:
            async with session.post(self._gateway_url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        f"Gateway returned {resp.status}: {body[:200]}"
                    )
                    await self.push_frame(
                        LLMTextFrame("Sorry, I had trouble connecting to the gateway.")
                    )
                    return

                # Parse SSE stream line by line
                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="replace").strip()

                    if not line:
                        continue

                    if line == "data: [DONE]":
                        break

                    if not line.startswith("data: "):
                        continue

                    json_str = line[6:]  # Strip "data: " prefix

                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Skipping non-JSON SSE line: {json_str[:100]}")
                        continue

                    # Extract content delta from OpenAI-format chunk
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")

                    if content:
                        if first_token_time is None:
                            first_token_time = time.monotonic()
                            ttft = first_token_time - t_start
                            logger.info(f"StreamingGatewayLLM TTFT: {ttft:.2f}s")

                        full_response_parts.append(content)
                        await self.push_frame(LLMTextFrame(content))

                    # Check for finish_reason
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        logger.debug(f"Stream finished: {finish_reason}")
                        break

        except aiohttp.ClientError as e:
            logger.error(f"Gateway connection error: {e}")
            if not full_response_parts:
                await self.push_frame(
                    LLMTextFrame("Sorry, I lost connection to the gateway.")
                )
            return
        except asyncio.TimeoutError:
            logger.error("Gateway request timed out")
            if not full_response_parts:
                await self.push_frame(
                    LLMTextFrame("Sorry, I took too long thinking about that.")
                )
            return

        elapsed = time.monotonic() - t_start
        response_text = "".join(full_response_parts)

        if response_text:
            logger.info(
                f"StreamingGatewayLLM response ({elapsed:.1f}s, "
                f"TTFT: {(first_token_time - t_start):.2f}s): "
                f"{response_text[:80]}..."
            )

            # Update conversation history (without voice hint for cleaner context)
            self._conversation_history.append(
                {"role": "user", "content": user_text}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": response_text}
            )

            # Trim history to max_history exchanges (each = 2 messages)
            max_messages = self._max_history * 2
            if len(self._conversation_history) > max_messages:
                self._conversation_history = self._conversation_history[-max_messages:]

            self._log_transcript(user_text, response_text, elapsed)
        else:
            logger.warning("Gateway returned empty response")
            await self.push_frame(
                LLMTextFrame("I'm sorry, I didn't catch that. Could you say that again?")
            )

    def _log_transcript(
        self, user_text: str, response_text: str, think_seconds: float
    ) -> None:
        """Append an exchange to the daily transcript file.

        Format matches v1 transcripts:
            ### HH:MM:SS
            **User (voice):** user text

            **Bot (voice):** response text

            _Response: 4.9s_
        """
        try:
            TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            filepath = TRANSCRIPT_DIR / f"voice-{now.strftime('%Y-%m-%d')}.md"

            # Create header on first entry of the day
            if not filepath.exists():
                header = (
                    f"# Voice Transcripts — {now.strftime('%Y-%m-%d')}\n\n"
                    "Auto-logged by voice_bot_v2 (StreamingGatewayLLMService).\n\n"
                )
                filepath.write_text(header)

            entry = (
                f"### {now.strftime('%H:%M:%S')}\n"
                f"**User (voice):** {user_text}\n\n"
                f"**Bot (voice):** {response_text}\n\n"
                f"_Response: {think_seconds:.1f}s_\n\n"
            )

            with open(filepath, "a") as f:
                f.write(entry)

            logger.debug(f"Transcript logged to {filepath.name}")
        except Exception as e:
            logger.warning(f"Failed to log transcript: {e}")

    async def cancel(self, frame: Frame) -> None:
        """Cancel any running request on pipeline cancel."""
        await super().cancel(frame)
        # aiohttp will cancel the in-flight request when the task is cancelled

    async def cleanup(self) -> None:
        """Clean up the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
