#!/usr/bin/env python3
"""Quick test of Kyutai TTS 1.6B — generate a sample and check VRAM usage."""

import time
import numpy as np
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM before load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_VOICE_REPO, TTSModel

# Load model from local files
t0 = time.monotonic()
checkpoint_info = CheckpointInfo.from_hf_repo("kyutai/tts-1.6b-en_fr")
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=32, temp=0.6, device=torch.device("cuda")
)
load_time = time.monotonic() - t0

vram_after = torch.cuda.memory_allocated() / 1e9
vram_reserved = torch.cuda.memory_reserved() / 1e9
print(f"\nModel loaded in {load_time:.1f}s")
print(f"VRAM allocated: {vram_after:.2f} GB")
print(f"VRAM reserved:  {vram_reserved:.2f} GB")

# Generate a test utterance
text = "Hey there! This is a test of the Kyutai TTS model running on our local GPU. How does it sound?"
voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"

print(f"\nGenerating: \"{text}\"")
print(f"Voice: {voice}")

entries = tts_model.prepare_script([text], padding_between=1)
voice_path = tts_model.get_voice_path(voice)
condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)

pcms = []
t0 = time.monotonic()

def _on_frame(frame):
    if (frame != -1).all():
        pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
        pcms.append(np.clip(pcm[0, 0], -1, 1))

all_entries = [entries]
all_condition_attributes = [condition_attributes]
with tts_model.mimi.streaming(len(all_entries)):
    result = tts_model.generate(
        all_entries, all_condition_attributes, on_frame=_on_frame
    )

gen_time = time.monotonic() - t0
audio = np.concatenate(pcms, axis=-1)
duration = len(audio) / tts_model.mimi.sample_rate

print(f"\nGenerated {duration:.1f}s of audio in {gen_time:.1f}s ({duration/gen_time:.1f}x realtime)")
print(f"Sample rate: {tts_model.mimi.sample_rate} Hz")
print(f"VRAM after generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Save to wav
import wave
wav_path = "/tmp/kyutai_test.wav"
audio_int16 = (audio * 32767).astype(np.int16)
with wave.open(wav_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(tts_model.mimi.sample_rate)
    wf.writeframes(audio_int16.tobytes())

print(f"Saved to {wav_path}")
