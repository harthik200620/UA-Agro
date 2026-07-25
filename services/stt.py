"""Speech-to-text via Sarvam AI.

Primary: Saaras v3 with code-mix mode (Telugu + English "Tenglish"), tuned for short
phone-style clips. If the configured model/endpoint is rejected, it retries once with
Saarika (plain te-IN transcription) so a key change in Sarvam's API doesn't break the demo.
The browser sends a 16 kHz mono WAV; we POST it as multipart/form-data.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct

import httpx

from . import _http
from . import sarvam_keys

try:
    import websockets                     # ships with uvicorn[standard]; pinned in requirements
except Exception:                         # absent → streaming STT unavailable, batch path used
    websockets = None

# Strip BOM / zero-width chars that dashboard bulk-pastes inject (str.strip() misses them).
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


SARVAM_STT_MODEL = _clean("SARVAM_STT_MODEL", "saaras:v3")
STT_URL = "https://api.sarvam.ai/speech-to-text"

# Ordered attempts: first the configured model (code-mix), then a robust fallback.
_ATTEMPTS = [
    {"model": SARVAM_STT_MODEL, "extra": {"mode": "codemix"}},
    {"model": "saarika:v2.5", "extra": {}},
]


def stt_available() -> bool:
    return sarvam_keys.available()


async def transcribe_wav(wav_bytes: bytes, language_code: str = "en-IN") -> str:
    """`language_code` is the caller's chosen language (en-IN / hi-IN / te-IN). Saaras code-mix
    still understands English words mixed in; this just biases recognition to the right script.

    Walks the KEY pool outside the MODEL attempts: a usage-limited key fails identically on every
    model, so re-trying models on a spent key would just burn the turn.
    """
    keys = sarvam_keys.order()
    if not keys:
        raise RuntimeError("SARVAM_API_KEY not set")

    last_err = None
    client = _http.client()
    for key in keys:
        rotate = False
        for attempt in _ATTEMPTS:
            files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
            data = {"model": attempt["model"], "language_code": language_code, **attempt["extra"]}
            try:
                resp = await client.post(STT_URL, headers={"api-subscription-key": key},
                                         files=files, data=data, timeout=30)
                if resp.status_code >= 400:
                    last_err = f"Sarvam STT {resp.status_code} ({attempt['model']}): {resp.text[:300]}"
                    if sarvam_keys.should_rotate(resp.status_code):
                        sarvam_keys.mark_bad(key, resp.status_code)
                        rotate = True          # this KEY is spent — the next model won't help
                        break
                    continue
                sarvam_keys.mark_ok(key)
                j = resp.json()
                return (j.get("transcript") or "").strip()
            except Exception as e:  # network / parse — try the next attempt
                last_err = f"{type(e).__name__}: {e}"
                continue
        if not rotate:
            break                              # failed for a reason another key won't fix
    raise RuntimeError(last_err or "Sarvam STT failed")


# ── STREAMING STT ──────────────────────────────────────────────────────────────────────────
# The browser already ships PCM frames to us WHILE the caller speaks, but the batch endpoint
# above can't start until they stop — so a whole POST of the whole utterance sat between "caller
# finished" and "agent starts thinking". Forwarding those frames to Sarvam's socket as they
# arrive means the transcript is ready essentially the moment the caller stops.
# MEASURED on identical real speech: 1-95ms after speech-end vs 652-1506ms for the batch call.
# STREAM_STT=0 restores the batch path.
STREAM_STT = _clean("STREAM_STT", "1").lower() not in ("0", "false", "no", "off")
_WS_URL = ("wss://api.sarvam.ai/speech-to-text/ws"
           "?model={model}&mode=codemix&language-code={lang}&sample_rate=16000"
           "&high_vad_sensitivity=true&flush_signal=true")


def _wav(pcm: bytes, sr: int = 16000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)


def join_segments(parts: list[str]) -> str:
    """Sarvam segments ONE utterance on its own VAD, so a single sentence commonly arrives as
    two or three `data` messages — and consecutive segments sometimes re-recognise a word across
    the boundary ("… पर" then "पर सफ़ेद …"). Taking only the first segment silently truncated the
    caller's sentence (measured: the actual question was dropped); joining naively duplicated
    words. Join everything, trimming a word-level overlap at each seam."""
    out: list[str] = []
    for seg in parts:
        w = (seg or "").split()
        if out and w:
            for k in range(min(4, len(out), len(w)), 0, -1):
                if [x.strip("।,?.!") for x in out[-k:]] == [x.strip("।,?.!") for x in w[:k]]:
                    w = w[k:]
                    break
        out.extend(w)
    return " ".join(out).strip()


class SarvamStream:
    """Streaming STT for one caller turn. start() → feed(pcm) per frame → finish() -> transcript.

    `ok` goes False on any failure and finish() returns "", which tells the caller to fall back
    to the batch endpoint with the frames it buffered anyway — a failed socket costs correctness
    nothing, only the latency saving.
    """

    def __init__(self, language_code: str = "hi-IN"):
        self.lang = language_code or "hi-IN"
        self.ok = False
        self._ws = None
        self._key = ""
        self._opening: asyncio.Task | None = None
        self._reader: asyncio.Task | None = None
        self._parts: list[str] = []
        self._err = ""

    def usable(self) -> bool:
        return bool(STREAM_STT and websockets and sarvam_keys.available())

    def start(self) -> None:
        """Open the socket without blocking — the handshake overlaps the caller still speaking."""
        if self.usable() and self._opening is None:
            self._opening = asyncio.ensure_future(self._open())

    async def _open(self) -> None:
        url = _WS_URL.format(model=SARVAM_STT_MODEL, lang=self.lang)
        self._key = sarvam_keys.current()
        hdrs = {"Api-Subscription-Key": self._key}
        try:
            try:
                conn = websockets.connect(url, additional_headers=hdrs, max_size=None)
            except TypeError:                       # websockets < 14 spelled it extra_headers
                conn = websockets.connect(url, extra_headers=hdrs, max_size=None)
            self._ws = await asyncio.wait_for(conn, timeout=6)
        except Exception as exc:
            # A rejected upgrade carries the HTTP status; rotate off a spent key so the batch
            # fallback below (and the next turn) start on a different one.
            code = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if sarvam_keys.should_rotate(code):
                sarvam_keys.mark_bad(self._key, code)
            self._ws, self._err = None, f"{type(exc).__name__} {code or ''}".strip()
            return
        self._reader = asyncio.ensure_future(self._read())
        self.ok = True

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    m = json.loads(raw)
                except Exception:
                    continue
                if m.get("type") == "data":
                    tr = ((m.get("data") or {}).get("transcript") or "").strip()
                    if tr:
                        self._parts.append(tr)
                elif m.get("type") == "error":
                    self._err = str(m.get("data"))[:160]
        except Exception:
            pass

    async def _ready(self) -> None:
        task = self._opening
        if task is not None:
            try:
                await task
            except Exception:
                pass
            if self._opening is task:
                self._opening = None

    async def feed(self, pcm16: bytes) -> None:
        """One raw PCM16 @16k frame, exactly as the browser sent it."""
        await self._ready()
        if not (self.ok and self._ws and pcm16):
            return
        try:
            # Raw pcm_s16le via input_audio_codec was tried and the server closes the socket on
            # it; WAV-wrapping each chunk is the shape that actually works.
            await self._ws.send(json.dumps({"audio": {
                "data": base64.b64encode(_wav(pcm16)).decode("ascii"),
                "sample_rate": "16000", "encoding": "audio/wav"}}))
        except Exception:
            self.ok = False

    async def finish(self, timeout: float = 2.5) -> str:
        """Flush, collect every remaining segment, and return the stitched transcript ("" on
        failure, which means: fall back to the batch call)."""
        await self._ready()
        if not (self.ok and self._ws):
            await self._shut()
            return ""
        try:
            await self._ws.send(json.dumps({"type": "flush"}))
        except Exception:
            self.ok = False
        # Sarvam gives no "this was the last segment" marker, so completeness is decided by a
        # quiet gap. Measured on real speech, segments of one utterance land ~1ms and ~94ms after
        # flush, so 150ms of silence is comfortably past the last one while costing far less than
        # the 220ms-per-segment sleep this originally used (that alone was ~440ms of the 560ms
        # this call was taking). Poll finely rather than sleeping in big steps.
        loop = asyncio.get_event_loop()
        hard_deadline = loop.time() + timeout
        quiet_s, seen, last_change = 0.15, len(self._parts), loop.time()
        while loop.time() < hard_deadline:
            await asyncio.sleep(0.02)
            if len(self._parts) != seen:
                seen, last_change = len(self._parts), loop.time()
                continue
            if self._parts and (loop.time() - last_change) >= quiet_s:
                break
        await self._shut()
        if self._parts:
            sarvam_keys.mark_ok(self._key)
        return join_segments(self._parts)

    async def cancel(self) -> None:
        self.ok = False
        await self._shut()

    async def _shut(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
