#!/usr/bin/env python3
"""Generate voice samples from a curated set of Kyutai TTS voices."""

import numpy as np
import torch
import soundfile as sf
import time
from pathlib import Path
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# Curated voice selection - diverse, conversational, English
VOICES = {
    # Expresso dataset - professional actors with different styles
    "expresso-happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",      # Current voice
    "expresso-calm": "expresso/ex03-ex01_calm_001_channel1_1143s.wav",       # Calm, warm
    "expresso-default-ch1": "expresso/ex01-ex02_default_001_channel1_168s.wav",  # Neutral
    "expresso-default-ch2": "expresso/ex01-ex02_default_001_channel2_198s.wav",  # Different speaker
    "expresso-narration-ch1": "expresso/ex03-ex02_narration_001_channel1_674s.wav",  # Storytelling
    "expresso-narration-ch2": "expresso/ex04-ex01_narration_001_channel1_605s.wav",  # Different narrator
    "expresso-awe": "expresso/ex03-ex01_awe_001_channel1_1323s.wav",         # Wonder/curiosity

    # Alba McKenna - named character voices
    "alba-casual": "alba-mackenna/casual.wav",
    "alba-announcer": "alba-mackenna/announcer.wav",

    # Unmute website voices
    "unmute-default": "unmute-prod-website/default_voice.wav",
    "unmute-narration": "unmute-prod-website/ex04_narration_longform_00001.wav",

    # EARS dataset - diverse speakers
    "ears-p003-neutral": "ears/p003/emo_neutral_freeform.wav",
    "ears-p003-interest": "ears/p003/emo_interest_freeform.wav",
    "ears-p031-neutral": "ears/p031/emo_neutral_freeform.wav",
    "ears-p031-interest": "ears/p031/emo_interest_freeform.wav",
}

TEST_TEXT = (
    "Hey! So I was thinking about what you said earlier, and honestly, "
    "I think you're onto something really interesting there. "
    "The way language encodes cultural attention patterns — "
    "that's not just a linguistics thing, that's almost philosophical. "
    "Want to dig into it more?"
)

OUTPUT_DIR = Path("/home/anna/clawd/voice-pipeline/v2/voice_samples")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading Kyutai TTS 1.6B (n_q=8)...")
    checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
    model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=8, temp=0.6, device=torch.device("cuda")
    )

    entries = model.prepare_script([TEST_TEXT], padding_between=1)

    for name, voice_id in VOICES.items():
        print(f"\n{'='*50}")
        print(f"Generating: {name}")
        print(f"Voice: {voice_id}")

        try:
            voice_path = model.get_voice_path(voice_id)
            condition_attributes = model.make_condition_attributes([voice_path], cfg_coef=2.0)

            pcms = []
            def _on_frame(frame):
                if (frame != -1).all():
                    pcm = model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))

            start = time.time()
            with model.mimi.streaming(1):
                model.generate([entries], [condition_attributes], on_frame=_on_frame)
            gen_time = time.time() - start

            if pcms:
                audio = np.concatenate(pcms, axis=-1)
                duration = len(audio) / model.mimi.sample_rate
                out_path = OUTPUT_DIR / f"{name}.wav"
                sf.write(str(out_path), audio, model.mimi.sample_rate)
                print(f"  Duration: {duration:.1f}s | Gen time: {gen_time:.1f}s | RTF: {duration/gen_time:.1f}x")
                print(f"  Saved: {out_path}")
            else:
                print(f"  FAILED: No audio generated")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n\nAll samples saved to {OUTPUT_DIR}/")
    print("Listen and pick your favorite!")

if __name__ == "__main__":
    main()
