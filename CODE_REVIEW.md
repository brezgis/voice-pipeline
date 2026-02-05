# Code Review — Voice Pipeline v2

*Reviewed: Feb 4, 2026*

## Overall Assessment

**Status: Working, shippable for personal use.**

The code successfully implements a real-time voice conversation system. It's well-structured, properly extends Pipecat's base classes, and handles the tricky async/threading boundaries correctly. Good for a first working version built in one session.

## Issues by Severity

### 🔴 Critical (None)

No critical issues. The pipeline works correctly.

### 🟡 Medium

#### 1. Hardcoded User ID and Guild ID
**File:** `voice_bot_v2.py`, `discord_transport.py`
**Issue:** Was hardcoded to a specific server. ✅ Fixed — now reads from environment variables.
```python
GUILD_ID = int(os.getenv("DISCORD_GUILD_ID", "0"))
AUTO_JOIN_USER_ID = int(os.getenv("DISCORD_AUTO_JOIN_USER", "0"))
```

#### 2. No reconnection handling
**File:** `discord_transport.py`
**Issue:** If Discord disconnects (network blip, server restart), the bot doesn't reconnect.
**Fix:** Add reconnection logic in `on_disconnect` event, or use py-cord's built-in reconnect (may already work, needs testing).

#### 3. Voice hint not configurable
**File:** `clawdbot_llm_service.py`
```python
VOICE_HINT = (
    "[Voice conversation — you're talking live in a Discord voice channel..."
)
```
**Issue:** Hardcoded mention of a specific user.
**Fix:** Make generic or parameterize:
```python
def __init__(self, *, user_name: str = "the user", ...):
    self._voice_hint = f"[Voice conversation — you're talking live with {user_name}..."
```

### 🟢 Low

#### 4. Debug logging left in
**File:** `discord_transport.py`
```python
if self._write_count <= 5 or self._write_count % 500 == 0:
    logger.debug(f"Sink.write called #{self._write_count}...")
```
**Issue:** Verbose debug logging every 500 frames (~10s). Fine for development, noisy for production.
**Fix:** Remove or gate behind LOGURU_LEVEL=TRACE.

#### 5. Unused imports
**File:** `discord_transport.py`
```python
from collections import defaultdict  # Never used
import time  # Never used
```
**Fix:** Remove unused imports.

#### 6. Magic numbers
**File:** `discord_transport.py`
```python
volume: float = 0.3  # Why 0.3?
```
**Fix:** Add comment explaining the value, or make it a named constant:
```python
DEFAULT_OUTPUT_VOLUME = 0.3  # Comfortable listening level, avoids clipping
```

#### 7. No type hints for some methods
**File:** `kyutai_tts_service.py`
```python
def _generate_sync(self, text: str) -> Optional[np.ndarray]:  # Good
def can_generate_metrics(self) -> bool:  # Good
async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:  # Good
async def cancel(self, frame):  # Missing type hint for frame
```
**Fix:** Add type hints to all methods.

#### 8. Potential memory leak on repeated model loads
**File:** `kyutai_tts_service.py`
```python
async def _ensure_loaded(self):
    if self._loaded:
        return
    # Loads model...
```
**Issue:** If `_ensure_loaded()` fails partway through, `_loaded` stays False but partial model may be in VRAM.
**Fix:** Add cleanup on failure:
```python
try:
    self._tts_model, self._voice_path, self._condition_attributes = (
        await asyncio.get_event_loop().run_in_executor(None, _load)
    )
    self._loaded = True
except Exception:
    self._tts_model = None  # Ensure cleanup
    raise
```

## Suggestions for Future Work

### Performance
- [ ] **Streaming TTS** — Currently generates full audio then chunks. True streaming would reduce latency.
- [ ] **Faster LLM** — Opus is slow (~4-6s). Consider Sonnet/Haiku for voice, or local model.
- [ ] **Preload models** — Load Whisper/Kyutai at startup instead of first use.

### Features
- [ ] **Interruption handling** — Let user interrupt bot mid-response (Pipecat supports this).
- [ ] **Multi-user** — Currently tracks only one user. Could expand to conversation with multiple participants.
- [ ] **Push-to-talk mode** — Alternative to VAD for noisy environments.
- [ ] **Transcript logging** — Save conversations to file (like v1's transcripts/).

### Reliability
- [ ] **Health checks** — Periodic ping to verify pipeline is still working.
- [ ] **Graceful degradation** — If TTS fails, send text to channel instead.
- [ ] **Metrics** — Track latency, error rates, VRAM usage.

### Code Quality
- [ ] **Tests** — Unit tests for resampling functions, integration tests for pipeline.
- [ ] **Config file** — Move all constants to YAML/TOML config.
- [ ] **Logging standardization** — Consistent log levels and messages.

## Files Changed Since Initial Implementation

| File | Changes |
|------|---------|
| `discord_transport.py` | Added `set_transport_ready()` calls, volume scaling |
| `clawdbot_llm_service.py` | Added voice hint, removed length constraints |
| `kyutai_tts_service.py` | Changed voice to expresso-awe |

## Conclusion

The code is solid for a first implementation. The main architectural decisions (custom Pipecat transport, subprocess for Clawdbot, lazy TTS loading) are sound. Issues are minor and don't affect functionality.

**Recommended next steps:**
1. Externalize configuration (user ID, guild ID, voice)
2. Add reconnection handling
3. Consider faster LLM for lower latency

---

*Review by Claude. Ship it!* 🚀
