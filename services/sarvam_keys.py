"""Sarvam API keys with rotation, shared by STT and TTS.

One free Sarvam key runs out part-way through a demo day; a second behind it keeps the call
alive instead of dropping the caller to browser STT and a dead voice.

STT and TTS deliberately share ONE rotation. They bill against the same Sarvam account, so a key
that has hit its limit transcribing has hit it for speech too — discovering that separately would
waste a round-trip on every single turn.

Keys come from SARVAM_API_KEY, _2, _3, _4, or a comma-separated SARVAM_API_KEYS.
"""
from __future__ import annotations

import os
import time

# Strip BOM / zero-width chars that dashboard bulk-pastes inject (str.strip() misses them).
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


def _load() -> list[str]:
    raw: list[str] = []
    combo = _clean("SARVAM_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    for name in ("SARVAM_API_KEY", "SARVAM_API_KEY_2", "SARVAM_API_KEY_3", "SARVAM_API_KEY_4"):
        raw.append(_clean(name))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


KEYS = _load()
_idx = 0
_cooldown: dict[str, float] = {}


def available() -> bool:
    return bool(KEYS)


def current() -> str:
    """The key to use when only one can be tried (the streaming socket)."""
    return order()[0] if KEYS else ""


def order() -> list[str]:
    """Keys to try for this call: the live one first, then the rest, anything cooling LAST.

    Cooling keys are still returned rather than dropped — if every key is rate-limited, trying an
    exhausted one and failing is strictly better than not calling at all and losing the turn.
    """
    if not KEYS:
        return []
    n = len(KEYS)
    rot = [KEYS[(_idx + i) % n] for i in range(n)]
    now = time.time()
    hot = [k for k in rot if _cooldown.get(k, 0) <= now]
    cold = [k for k in rot if _cooldown.get(k, 0) > now]
    return hot + cold


def should_rotate(status: int) -> bool:
    """429 = rate/usage limit, 401/403 = key rejected or subscription spent, 5xx = their side."""
    return status in (401, 403, 429) or status >= 500


def mark_bad(key: str, status: int = 0) -> None:
    """Take a key out of rotation and advance, so the NEXT call starts on a different one."""
    global _idx
    if not key or not KEYS:
        return
    # A spent subscription (401/403) won't recover in a minute the way a rate limit does.
    _cooldown[key] = time.time() + (900 if status in (401, 403) else 60)
    if KEYS[_idx % len(KEYS)] == key:
        _idx = (_idx + 1) % len(KEYS)


def mark_ok(key: str) -> None:
    _cooldown.pop(key, None)


def status() -> dict:
    """For /config — never exposes the key material itself."""
    now = time.time()
    return {
        "keys": len(KEYS),
        "live": sum(1 for k in KEYS if _cooldown.get(k, 0) <= now),
    }
