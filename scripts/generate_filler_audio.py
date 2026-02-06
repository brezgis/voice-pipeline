#!/usr/bin/env python3
"""
Generate filler/thinking audio clips using Kyutai TTS.

Produces short "thinking cue" clips (e.g. "Hmm.", "Okay.") using the same
Kyutai TTS model and voice as the main voice bot. Saves them as raw PCM
int16 files at 24kHz mono in filler_audio/.

Run once at setup, or automatically on voice bot startup if clips don't exist.

Usage:
    cd voice-pipeline/v2
    source venv/bin/activate
    python scripts/generate_filler_audio.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# =============================================================================
# Config — must match kyutai_tts_service.py
# =============================================================================

MODEL_REPO = "kyutai/tts-1.6b-en_fr"
VOICE = "expresso/ex03-ex01_awe_001_channel1_1323s.wav"
N_Q = 8
TEMP = 0.6
CFG_COEF = 2.0
SAMPLE_RATE = 24000

FILLER_PHRASES = [
    "Hmm.",
    "Let me think.",
    "Okay.",
    "One sec.",
]

# Output directory (relative to v2/)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "filler_audio"


def generate_filler_clips():
    """Load Kyutai TTS and generate filler audio clips."""
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading Kyutai TTS from {MODEL_REPO} (n_q={N_Q}, device={device})...")

    checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=N_Q, temp=TEMP, device=device
    )
    voice_path = tts_model.get_voice_path(VOICE)
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=CFG_COEF
    )

    logger.info(f"Model loaded. Generating {len(FILLER_PHRASES)} filler clips...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for phrase in FILLER_PHRASES:
        # Sanitize filename
        safe_name = phrase.lower().replace(" ", "_").replace(".", "").replace(",", "")
        out_path = OUTPUT_DIR / f"{safe_name}.pcm"

        logger.info(f"  Generating: '{phrase}' → {out_path.name}")

        entries = tts_model.prepare_script([phrase], padding_between=1)
        pcms: list[np.ndarray] = []

        def _on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))

        with tts_model.mimi.streaming(1):
            tts_model.generate([entries], [condition_attributes], on_frame=_on_frame)

        if pcms:
            audio_float = np.concatenate(pcms, axis=-1)
            audio_int16 = (audio_float * 32767).astype(np.int16)
            audio_int16.tofile(out_path)
            duration_ms = len(audio_int16) / SAMPLE_RATE * 1000
            logger.info(f"    Saved: {out_path.name} ({len(audio_int16)} samples, {duration_ms:.0f}ms)")
        else:
            logger.warning(f"    No audio generated for '{phrase}'")

    # Write a manifest so the processor knows what's available
    manifest_path = OUTPUT_DIR / "manifest.txt"
    with open(manifest_path, "w") as f:
        for phrase in FILLER_PHRASES:
            safe_name = phrase.lower().replace(" ", "_").replace(".", "").replace(",", "")
            f.write(f"{safe_name}.pcm\t{phrase}\n")

    logger.info(f"Done! {len(FILLER_PHRASES)} clips saved to {OUTPUT_DIR}/")
    logger.info(f"Manifest written to {manifest_path}")

    # Cleanup
    del tts_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_filler_clips()
