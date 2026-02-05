#!/usr/bin/env python3
"""
Clawdbot LLM Service for Pipecat v2

Routes LLM requests through `clawdbot agent --session-id voice --json --timeout 30`
so the voice bot gets full Clawdbot context (SOUL.md, memory, tools, identity).

This extends Pipecat's AIService:
- Handles TranscriptionFrame (from STT) directly
- Emits LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame
- The TTSService downstream consumes the LLMTextFrame/LLMFullResponseEndFrame
"""

import asyncio
import json
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


class ClawdbotLLMService(AIService):
    """LLM service that routes through Clawdbot agent subprocess.

    Instead of extending LLMService (which requires OpenAI-style context
    handling, function calling, etc), we extend AIService directly and
    handle the frames we care about: TranscriptionFrame from STT.

    Flow: TranscriptionFrame → clawdbot agent → LLMTextFrame chunks → TTS
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

    async def _run_clawdbot(self, prompt: str) -> None:
        """Run clawdbot agent and parse response as LLMTextFrame."""
        cmd = [
            "clawdbot", "agent",
            "--session-id", self._session_id,
            "--json",
            "--timeout", str(self._timeout),
            "-m", self._voice_hint + prompt,
        ]
        if self._model:
            cmd.extend(["--model", self._model])

        logger.debug(f"Running: clawdbot agent --session-id {self._session_id}")

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

            output = stdout.decode().strip()
            if stderr:
                logger.debug(f"Clawdbot stderr: {stderr.decode()[:200]}")

            if not output:
                logger.warning("Clawdbot returned empty output")
                await self.push_frame(
                    LLMTextFrame("I'm sorry, I didn't catch that. Could you say that again?")
                )
                return

            # Parse the JSON response
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                logger.debug(f"Clawdbot non-JSON output: {output[:200]}")
                await self.push_frame(LLMTextFrame(output))
                return

            status = data.get("status", "")
            if status == "error":
                logger.error(f"Clawdbot error: {data}")
                await self.push_frame(
                    LLMTextFrame("Sorry, I had trouble with that.")
                )
                return

            # Extract text from payloads
            payloads = data.get("result", {}).get("payloads", [])
            got_content = False
            for payload in payloads:
                text = payload.get("text", "")
                if text:
                    got_content = True
                    logger.info(f"ClawdbotLLM response: {text[:80]}...")
                    await self.push_frame(LLMTextFrame(text))

            if not got_content:
                logger.warning(f"No text in clawdbot response: {output[:200]}")
                await self.push_frame(
                    LLMTextFrame("I'm sorry, I didn't catch that. Could you say that again?")
                )

        except asyncio.TimeoutError:
            logger.error("Clawdbot agent timed out")
            await self.push_frame(
                LLMTextFrame("Sorry, I took too long thinking about that.")
            )
        except Exception as e:
            logger.error(f"Clawdbot communication error: {e}")
            raise
        finally:
            if self._current_process:
                try:
                    self._current_process.terminate()
                    await asyncio.wait_for(self._current_process.wait(), timeout=5)
                except Exception:
                    try:
                        self._current_process.kill()
                    except Exception:
                        pass
            self._current_process = None

    async def cancel(self, frame: Frame) -> None:
        """Cancel any running clawdbot process on pipeline cancel."""
        await super().cancel(frame)
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
            self._current_process = None
