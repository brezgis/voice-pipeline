#!/usr/bin/env python3
"""
Clawdbot LLM Service for Pipecat v2

Routes LLM requests through `openclaw agent --session-id voice --json --timeout 30`
so the voice bot gets full OpenClaw context (SOUL.md, memory, tools, identity).

This extends Pipecat's AIService:
- Handles TranscriptionFrame (from STT) directly
- Emits LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame
- The TTSService downstream consumes the LLMTextFrame/LLMFullResponseEndFrame
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
# No length constraints: Claude decides response length based on context.
DEFAULT_VOICE_HINT = (
    "[Voice conversation — you're talking live in a Discord voice channel. "
    "Write naturally for speech: contractions, casual phrasing, no markdown/formatting/lists. "
    "Talk like a real person would. Your response will be spoken aloud via TTS.]\n\n"
)

# Transcript output directory (daily files: voice-YYYY-MM-DD.md)
TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"


class ClawdbotLLMService(AIService):
    """LLM service that routes through Clawdbot agent subprocess.

    Instead of extending LLMService (which requires OpenAI-style context
    handling, function calling, etc), we extend AIService directly and
    handle the frames we care about: TranscriptionFrame from STT.

    Flow: TranscriptionFrame → openclaw agent → LLMTextFrame chunks → TTS
    """

    def __init__(
        self,
        *,
        session_id: str = "voice",
        timeout: int = 30,
        voice_hint: str = DEFAULT_VOICE_HINT,
        model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._timeout = timeout
        self._voice_hint = voice_hint
        self._model = model
        self._current_process: Optional[asyncio.subprocess.Process] = None

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

        logger.info(f"ClawdbotLLM processing: {text[:80]}...")

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self._run_clawdbot(text.strip())
        except Exception as e:
            logger.error(f"ClawdbotLLM error: {e}")
            await self.push_frame(
                LLMTextFrame("I'm sorry, I had trouble processing that.")
            )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    def _log_transcript(
        self, user_text: str, response_text: str, think_seconds: float
    ) -> None:
        """Append an exchange to the daily transcript file.

        Format matches v1 transcripts:
            ### HH:MM:SS
            **User (voice):** user text

            **Bot (voice):** response text

            _Think: 4.9s_
        """
        try:
            TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            filepath = TRANSCRIPT_DIR / f"voice-{now.strftime('%Y-%m-%d')}.md"

            # Create header on first entry of the day
            if not filepath.exists():
                header = (
                    f"# Voice Transcripts — {now.strftime('%Y-%m-%d')}\n\n"
                    "Auto-logged by voice_bot_v2 (ClawdbotLLMService).\n\n"
                )
                filepath.write_text(header)

            entry = (
                f"### {now.strftime('%H:%M:%S')}\n"
                f"**User (voice):** {user_text}\n\n"
                f"**Bot (voice):** {response_text}\n\n"
                f"_Think: {think_seconds:.1f}s_\n\n"
            )

            with open(filepath, "a") as f:
                f.write(entry)

            logger.debug(f"Transcript logged to {filepath.name}")
        except Exception as e:
            logger.warning(f"Failed to log transcript: {e}")

    async def _run_clawdbot(self, prompt: str) -> None:
        """Run openclaw agent and parse response as LLMTextFrame.
        
        NOTE: Message is passed via -m flag which is visible in /proc/cmdline.
        openclaw agent does not support stdin message input. On this single-user
        system this is acceptable; on shared systems, consider proxying through
        the gateway API instead of the CLI.
        """
        msg_text = self._voice_hint + prompt
        cmd = [
            "openclaw", "agent",
            "--session-id", self._session_id,
            "--json",
            "--timeout", str(self._timeout),
            "-m", msg_text,
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        logger.debug(f"Running: openclaw agent --session-id {self._session_id}")

        t_start = time.monotonic()

        self._current_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                self._current_process.communicate(),
                timeout=self._timeout + 10,
            )

            think_seconds = time.monotonic() - t_start

            output = stdout.decode().strip()
            if stderr:
                logger.debug(f"OpenClaw stderr: {stderr.decode()[:200]}")

            if not output:
                logger.warning("OpenClaw returned empty output")
                await self.push_frame(
                    LLMTextFrame("I'm sorry, I didn't catch that. Could you say that again?")
                )
                return

            # Parse the JSON response
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                logger.debug(f"OpenClaw non-JSON output: {output[:200]}")
                await self.push_frame(LLMTextFrame(output))
                self._log_transcript(prompt, output, think_seconds)
                return

            status = data.get("status", "")
            if status == "error":
                logger.error(f"OpenClaw error: {data}")
                await self.push_frame(
                    LLMTextFrame("Sorry, I had trouble with that.")
                )
                return

            # Extract text from payloads
            payloads = data.get("result", {}).get("payloads", [])
            response_parts = []
            got_content = False
            for payload in payloads:
                text = payload.get("text", "")
                if text:
                    got_content = True
                    response_parts.append(text)
                    logger.info(f"ClawdbotLLM response: {text[:80]}...")
                    await self.push_frame(LLMTextFrame(text))

            if got_content:
                full_response = " ".join(response_parts)
                self._log_transcript(prompt, full_response, think_seconds)
            else:
                logger.warning(f"No text in openclaw response: {output[:200]}")
                await self.push_frame(
                    LLMTextFrame("I'm sorry, I didn't catch that. Could you say that again?")
                )

        except asyncio.TimeoutError:
            logger.error("OpenClaw agent timed out")
            await self.push_frame(
                LLMTextFrame("Sorry, I took too long thinking about that.")
            )
        except Exception as e:
            logger.error(f"OpenClaw communication error: {e}")
            raise
        finally:
            proc = self._current_process
            self._current_process = None
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    async def cancel(self, frame: Frame) -> None:
        """Cancel any running openclaw process on pipeline cancel."""
        await super().cancel(frame)
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None
