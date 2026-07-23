"""Text-to-speech behind a single pluggable synthesize() function.

Default voice = ElevenLabs (model from ELEVENLABS_MODEL_ID, normally "eleven_v3" — the
only ElevenLabs model that speaks Telugu). At startup we probe the key with GET /v1/models;
if the model isn't available we transparently fall back to Sarvam Bulbul (reliably natural
Telugu). If neither is usable, synthesize() returns (None, None) and the agent shows text only.

synthesize(text) -> (audio_bytes | None, mime | None)
  ElevenLabs -> mp3  (audio/mpeg)
  Sarvam     -> wav  (audio/wav)   both play natively in the browser.
"""
from __future__ import annotations

import os
import re
import base64
import asyncio

import httpx

from . import _http

# Strip BOM / zero-width chars (U+FEFF, U+200B-U+200D) that dashboard bulk-pastes inject
# and that str.strip() does NOT remove. Built via chr() so the source stays pure ASCII.
_JUNK = (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D))


def _clean(name: str, default: str = "") -> str:
    """Read an env var, removing BOM/zero-width chars plus quotes/whitespace."""
    v = os.getenv(name, default) or ""
    for ch in _JUNK:
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


TTS_PROVIDER = _clean("TTS_PROVIDER", "elevenlabs").lower()


def _load_eleven_keys() -> list[str]:
    """ELEVENLABS_API_KEY + _2/_3 (or comma-separated ELEVENLABS_API_KEYS) for rotation —
    free accounts get 10k credits/month, which heavy demo days exhaust. NOTE: the voice
    (ELEVENLABS_VOICE_ID) must be added to EACH account's voice library."""
    raw = []
    combo = _clean("ELEVENLABS_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    for name in ("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY_2", "ELEVENLABS_API_KEY_3"):
        raw.append(_clean(name))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_ELEVEN_KEYS = _load_eleven_keys()
_eleven_key_idx = 0
ELEVEN_KEY = _ELEVEN_KEYS[0] if _ELEVEN_KEYS else ""
# One voice per language, straight from env — ELEVENLABS_VOICE_ID is the English/primary
# voice (ELEVENLABS_VOICE_ID_EN overrides it if ever set); Hindi and Telugu use their own
# ids and fall back to the primary when unset. No hardcoded voice ids in code.
_ENV_PRIMARY_VOICE = _clean("ELEVENLABS_VOICE_ID")
ELEVEN_VOICE = _clean("ELEVENLABS_VOICE_ID_EN") or _ENV_PRIMARY_VOICE                     # English
ELEVEN_VOICE_HI = _clean("ELEVENLABS_VOICE_ID_HI") or _ENV_PRIMARY_VOICE                  # Hindi
ELEVEN_VOICE_TE = _clean("ELEVENLABS_VOICE_ID_TE") or _ENV_PRIMARY_VOICE                  # Telugu
ELEVEN_MODEL = _clean("ELEVENLABS_MODEL_ID", "eleven_v3")            # Telugu (v3-only language)
# EN/HI ride multilingual_v2: steadier pronunciation (Indian place names!) and faster than v3.
ELEVEN_MODEL_ENHI = _clean("ELEVENLABS_MODEL_ID_ENHI", "eleven_multilingual_v2")


def _model_for(lang: str) -> str:
    return ELEVEN_MODEL if (lang or "").lower() == "telugu" else ELEVEN_MODEL_ENHI


def _voice_for(lang: str) -> str:
    """eleven_v3 is multilingual, but a voice designed for a language sounds best in it —
    each language can have its own voice; unset ones fall back to the primary."""
    l = (lang or "").lower()
    if l == "telugu":
        return ELEVEN_VOICE_TE
    if l == "hindi":
        return ELEVEN_VOICE_HI
    return ELEVEN_VOICE


# 128 kbps mp3 keeps consonants crisp (clarity on phone-grade audio) for a tiny transfer cost
# over 64 kbps. Override with ELEVENLABS_OUTPUT_FORMAT if a plan needs a different format.
_OUTPUT_FORMAT = _clean("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")


def _voice_settings_for(lang: str) -> dict:
    """ElevenLabs' natural, clean defaults — the settings LEAST likely to sound weird/warbly.
      • stability 0.5   → the "natural" default: even but not flat.
      • similarity 0.75 → high enough to keep the voice's timbre, but NOT so high that it
                          reproduces artifacts from the source audio (a common cause of a
                          weird, robotic quality — especially on English).
      • style 0.0       → no exaggerated emphasis; neutral and easy to listen to.
      • use_speaker_boost → lifts intelligibility.
    NOTE: these knobs only shape DELIVERY. The accent/quality of English comes from the VOICE
    you pick (ELEVENLABS_VOICE_ID) — a clean native-English voice matters far more than any
    setting. Every knob is env-overridable (ELEVENLABS_STABILITY / _SIMILARITY / _STYLE)."""
    return {
        "stability": _float_env("ELEVENLABS_STABILITY", "0.5"),
        "similarity_boost": _float_env("ELEVENLABS_SIMILARITY", "0.75"),
        "style": _float_env("ELEVENLABS_STYLE", "0.0"),
        "use_speaker_boost": True,
    }


def _float_env(name: str, default: str) -> float:
    try:
        return float(_clean(name, default))
    except ValueError:
        return float(default)


# eleven_v3 treats [bracketed] text as performance directions (e.g. [laughs], [whispers]) and
# does NOT read it aloud — strip any such tags so the voice speaks EXACTLY the reply, nothing
# more or less. The short cap avoids touching genuine bracketed content.
_AUDIO_TAG = re.compile(r"\[[^\]\n]{0,30}\]")


def _strip_audio_tags(text: str) -> str:
    return _AUDIO_TAG.sub("", text).strip()


# Respell names ONLY for the SPOKEN audio (the transcript keeps the real spelling). The EN
# voice (U9JK "keerthi hindi english perfect") is a Voice-Design voice built for Indian names,
# so we respell as little as possible — on the sibling PetSecure build, the same kind of forced
# respelling made "Priya" come out as "Praya" (over-correction, not under). Hindi/Telugu replies
# are in native script (रिया / రియా) and never touched. Add a pair ONLY if a name is genuinely
# mis-said by ear on THIS voice.
_PRONOUNCE = {}
_PRONOUNCE_RE = [(re.compile(r"\b" + re.escape(k) + r"\b", re.I), v) for k, v in _PRONOUNCE.items()]


def _fix_pronunciation(text: str) -> str:
    for rx, repl in _PRONOUNCE_RE:
        text = rx.sub(repl, text)
    return text

SARVAM_KEY = _clean("SARVAM_API_KEY")
SARVAM_TTS_MODEL = _clean("SARVAM_TTS_MODEL", "bulbul:v2")
SARVAM_TTS_SPEAKER = _clean("SARVAM_TTS_SPEAKER", "anushka")

# eleven_v3 is the ONLY ElevenLabs model that speaks Telugu, and it is noticeably SLOW.
# Sarvam Bulbul speaks Telugu natively — faster, and it reads Indic numbers/dates cleanly.
# So Telugu defaults to Sarvam whenever a Sarvam key exists (falls back to eleven_v3 if not).
# Force eleven_v3 with TELUGU_TTS=elevenlabs.
TELUGU_TTS = _clean("TELUGU_TTS", "sarvam").lower()

# Probe results (set by probe_elevenlabs at startup)
_eleven_ok: bool | None = None
_eleven_reason: str = "not probed"
# Voices this key can't speak (402 paid_plan_required on library voices, 404 voice_not_found).
# Tracked PER VOICE — one dead voice must never silence the languages whose voices work.
_dead_voices: set[str] = set()
_probe_fails = 0  # consecutive probe exceptions (see probe_elevenlabs)


def eleven_ok() -> bool:
    return bool(_eleven_ok)


def eleven_reason() -> str:
    return _eleven_reason


def active_provider() -> str:
    """What will actually speak, given keys + probe result."""
    if TTS_PROVIDER == "none":
        return "none"
    if TTS_PROVIDER == "elevenlabs" and _eleven_ok:
        return "elevenlabs"
    if SARVAM_KEY:
        return "sarvam"
    return "none"


async def probe_elevenlabs() -> None:
    """Check whether the configured ElevenLabs model is available on this key."""
    global _eleven_ok, _eleven_reason, _probe_fails
    if TTS_PROVIDER != "elevenlabs":
        _eleven_ok = False
        _eleven_reason = f"TTS_PROVIDER={TTS_PROVIDER}"
        return
    if not ELEVEN_KEY:
        _eleven_ok = False
        _eleven_reason = "no ELEVENLABS_API_KEY"
        return
    if not ELEVEN_VOICE:
        _eleven_ok = False
        _eleven_reason = "no ELEVENLABS_VOICE_ID"
        return
    try:
        resp = await _http.client().get(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": ELEVEN_KEY}, timeout=15,
        )
        resp.raise_for_status()
        models = resp.json()
        ids = {m.get("model_id") for m in models} if isinstance(models, list) else set()
        if ELEVEN_MODEL in ids:
            _eleven_ok = True
            _eleven_reason = "ok"
        else:
            _eleven_ok = False
            _eleven_reason = (
                f"{ELEVEN_MODEL} not on this key — falling back to Sarvam. "
                f"available: {sorted(i for i in ids if i)}"
            )
    except Exception as e:
        # Transient (network flake): leave _eleven_ok as None so the next turn re-probes —
        # a single hiccup must not lock this instance onto Sarvam for its whole life (that's
        # how calls ended up switching voices between turns). After 3 straight failures,
        # accept it's really down so we stop paying the probe latency every turn.
        _probe_fails += 1
        _eleven_ok = False if _probe_fails >= 3 else None
        _eleven_reason = f"probe failed ({type(e).__name__}: {e}) — Sarvam this turn"
        return
    _probe_fails = 0


async def _elevenlabs(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    """ElevenLabs TTS with key rotation. On quota/auth failure it advances to the next key;
    when every key is exhausted it flips _eleven_ok off so later turns skip the wasted call
    and go straight to Sarvam (no extra latency, and /config reports the truth)."""
    global _eleven_key_idx, _eleven_ok, _eleven_reason
    voice = _voice_for(lang)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    params = {"output_format": _OUTPUT_FORMAT}  # crisp, clear consonants
    body = {
        "text": text,
        "model_id": _model_for(lang),
        # Clear, warm, professional delivery — see _voice_settings_for().
        "voice_settings": _voice_settings_for(lang),
    }
    last_detail, quota_fail = "", False
    for attempt in range(max(3, len(_ELEVEN_KEYS))):
        key = _ELEVEN_KEYS[_eleven_key_idx] if _ELEVEN_KEYS else ELEVEN_KEY
        resp = await _http.client().post(
            url, headers={"xi-api-key": key, "Content-Type": "application/json"},
            params=params, json=body,
        )
        if resp.status_code < 400:
            return resp.content, "audio/mpeg"
        last_detail = resp.text[:200]
        # 401 / quota_exceeded = the KEY is dead (flip off). A bare 429 is usually just the
        # free tier's 2-concurrent-request limit — transient, never disable the voice for it.
        quota_fail = resp.status_code == 401 or "quota_exceeded" in last_detail
        if quota_fail:
            if len(_ELEVEN_KEYS) > 1:
                _eleven_key_idx = (_eleven_key_idx + 1) % len(_ELEVEN_KEYS)
                continue
            break  # single dead key — no point retrying it
        if resp.status_code == 429 or resp.status_code >= 500:
            # Transient: the concurrency slot frees when the in-flight synth finishes.
            # WAIT and retry with the SAME voice — falling back to Sarvam here is what made
            # one line of a call speak in a different voice.
            await asyncio.sleep(0.8 * (attempt + 1))
            continue
        break
    if quota_fail:
        _eleven_ok = False
        _eleven_reason = "ElevenLabs credits exhausted (all keys) — speaking with Sarvam fallback"
    elif resp.status_code in (402, 404):
        # 402 paid_plan_required (library voice on a free account) or 404 voice_not_found —
        # both are specific to THIS voice on this key, NOT account-wide: an owned Voice-Design
        # voice can return 200 while library voices 402. Kill just this voice so later turns
        # skip the wasted call; the other languages keep their working voices.
        _dead_voices.add(voice)
        _eleven_reason = f"voice {voice} unusable ({resp.status_code}) — Sarvam for that language"
    raise RuntimeError(f"ElevenLabs failed ({resp.status_code}): {last_detail}")


_SARVAM_LANG = {"english": "en-IN", "hindi": "hi-IN", "telugu": "te-IN"}


async def _sarvam(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    if not SARVAM_KEY:
        return None, None
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}
    body = {
        "inputs": [text[:480]],  # Bulbul caps input length
        "target_language_code": _SARVAM_LANG.get((lang or "").lower(), "en-IN"),
        "model": SARVAM_TTS_MODEL,
        "speaker": SARVAM_TTS_SPEAKER,
    }
    resp = await _http.client().post(url, headers=headers, json=body)
    resp.raise_for_status()
    j = resp.json()
    audios = j.get("audios") or []
    if not audios:
        return None, None
    return base64.b64decode(audios[0]), "audio/wav"


async def synthesize(text: str, lang: str = "english") -> tuple[bytes | None, str | None]:
    text = _fix_pronunciation(_strip_audio_tags((text or "").strip()))
    if not text or TTS_PROVIDER == "none":
        return None, None

    lng = (lang or "").lower()

    # Lazy probe (serverless cold start may not have run the startup hook).
    if _eleven_ok is None:
        await probe_elevenlabs()

    # Telugu: prefer Sarvam Bulbul — faster than the slow eleven_v3 and native to the language
    # (also reads numbers/dates more cleanly). Only when a Sarvam key exists; else fall through
    # to ElevenLabs v3 as before.
    if lng == "telugu" and TELUGU_TTS == "sarvam" and SARVAM_KEY:
        try:
            audio, mime = await _sarvam(text, lang)
            if audio:
                return audio, mime
        except Exception:
            pass  # fall through to ElevenLabs v3

    # Preferred: ElevenLabs (if probe said it's usable and this language's voice isn't dead)
    if TTS_PROVIDER == "elevenlabs" and _eleven_ok and _voice_for(lng) not in _dead_voices:
        try:
            audio, mime = await _elevenlabs(text, lang)
            if audio:
                return audio, mime
        except Exception:
            pass  # fall through to Sarvam

    # Fallback / explicit Sarvam
    try:
        return await _sarvam(text, lang)
    except Exception:
        return None, None
