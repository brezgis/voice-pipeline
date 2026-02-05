#!/usr/bin/env python3
"""
Kyutai TTS 1.6B Service for Pipecat v2

Extends Pipecat's TTSService properly:
- Implements run_tts() abstract method
- Yields TTSAudioRawFrame with 24kHz int16 PCM
- Base class handles text aggregation, LLMFullResponseEndFrame flushing, etc.
- BaseOutputTransport handles resampling if needed

Uses the moshi library to load Kyutai TTS 1.6B from HuggingFace cache.
Benchmarked at 5.28x realtime with n_q=8 on RTX 5070 Ti.
"""

import asyncio
import gc
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

# =============================================================================
# Kyutai TTS constants
# =============================================================================

KYUTAI_SAMPLE_RATE = 24000
KYUTAI_MODEL_REPO = "kyutai/tts-1.6b-en_fr"

# Voice reference clip for cloning. Longer clips = better quality.
# See voice_samples/ and sample_voices_kyutai.py for alternatives.
KYUTAI_VOICE = "expresso/ex03-ex01_awe_001_channel1_1323s.wav"

# Performance tuning
KYUTAI_N_Q = 8       # Codec quantization steps — lower = faster (8 → 5.28x RT, 32 → 1.74x)
KYUTAI_TEMP = 0.6    # Generation temperature
KYUTAI_CFG_COEF = 2.0  # Classifier-free guidance strength

# Audio chunking for streaming playback (~100ms chunks)
TTS_CHUNK_MS = 100


class KyutaiTTSService(TTSService):
    """Kyutai TTS 1.6B service for Pipecat.

    Loads from HuggingFace cache. Outputs 24kHz mono int16 PCM.
    """

    def __init__(
        self,
        *,
        model_repo: str = KYUTAI_MODEL_REPO,
        voice: str = KYUTAI_VOICE,
        n_q: int = KYUTAI_N_Q,
        temp: float = KYUTAI_TEMP,
        cfg_coef: float = KYUTAI_CFG_COEF,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(sample_rate=KYUTAI_SAMPLE_RATE, **kwargs)

        self._model_repo = model_repo
        self._voice = voice
        self._n_q = n_q
        self._temp = temp
        self._cfg_coef = cfg_coef
        self._device_str = device

        self._tts_model: Optional[object] = None  # moshi.models.tts.TTSModel
        self._voice_path: Optional[object] = None  # pathlib.Path
        self._condition_attributes: Optional[object] = None
        self._loaded: bool = False

    async def _ensure_loaded(self) -> None:
        """Load the model on first use (~5s, ~4GB VRAM)."""
        if self._loaded:
            return

        logger.info(f"Loading Kyutai TTS from {self._model_repo} (n_q={self._n_q})...")

        def _load():
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import TTSModel

            device = torch.device(self._device_str)
            checkpoint_info = CheckpointInfo.from_hf_repo(self._model_repo)
            tts_model = TTSModel.from_checkpoint_info(
                checkpoint_info, n_q=self._n_q, temp=self._temp, device=device
            )
            voice_path = tts_model.get_voice_path(self._voice)
            condition_attributes = tts_model.make_condition_attributes(
                [voice_path], cfg_coef=self._cfg_coef
            )
            return tts_model, voice_path, condition_attributes

        try:
            self._tts_model, self._voice_path, self._condition_attributes = (
                await asyncio.get_event_loop().run_in_executor(None, _load)
            )
            self._loaded = True
            vram = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Kyutai TTS loaded. VRAM: {vram:.2f} GB")
        except Exception:
            # Clean up partial state on failure
            self._tts_model = None
            self._voice_path = None
            self._condition_attributes = None
            gc.collect()
            torch.cuda.empty_cache()
            raise

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text, yielding TTSAudioRawFrame chunks.

        This is the abstract method required by TTSService. The base class
        calls this via process_generator() after text aggregation.
        """
        if not text.strip():
            return

        await self._ensure_loaded()

        logger.debug(f"TTS generating: {text[:60]}...")

        yield TTSStartedFrame()

        try:
            audio_int16 = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, text
            )

            if audio_int16 is not None and len(audio_int16) > 0:
                chunk_samples = KYUTAI_SAMPLE_RATE * TTS_CHUNK_MS // 1000
                chunk_bytes = chunk_samples * 2  # int16 = 2 bytes per sample

                audio_bytes = audio_int16.tobytes()
                for i in range(0, len(audio_bytes), chunk_bytes):
                    chunk = audio_bytes[i : i + chunk_bytes]
                    if chunk:
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=KYUTAI_SAMPLE_RATE,
                            num_channels=1,
                        )
            else:
                logger.warning("TTS generated empty audio")

        except Exception as e:
            logger.error(f"TTS generation error: {e}")
        finally:
            yield TTSStoppedFrame()

    def _generate_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous TTS generation. Runs in executor thread.

        Returns int16 numpy array of audio samples at 24kHz.
        """
        try:
            entries = self._tts_model.prepare_script([text], padding_between=1)

            pcms: list[np.ndarray] = []

            def _on_frame(frame):
                if (frame != -1).all():
                    pcm = self._tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))

            with self._tts_model.mimi.streaming(1):
                self._tts_model.generate(
                    [entries], [self._condition_attributes], on_frame=_on_frame
                )

            if pcms:
                audio_float = np.concatenate(pcms, axis=-1)
                audio_int16 = (audio_float * 32767).astype(np.int16)
                return audio_int16
            return None

        except Exception as e:
            logger.error(f"Kyutai TTS sync generation error: {e}")
            return None

    async def cancel(self, frame: Frame) -> None:
        """Handle cancel frame."""
        await super().cancel(frame)

    def can_generate_metrics(self) -> bool:
        return True
