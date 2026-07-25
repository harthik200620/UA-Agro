"""Verba × UA Agro (Kisan Sewa Kendra) — AI voice calling agents (a Sahayak AI product).

FastAPI app:
  GET  /              -> the agent page (scenario picker: sales line / Kisan Gosthi outreach)
  GET  /crm           -> Verba CRM: live write-back view of every conversation outcome
  GET  /config        -> which STT/TTS providers are live (the page configures itself)
  POST /api/opening   -> the line the agent speaks FIRST for a scenario (text + cached audio)
  POST /api/crm       -> recent CRM rows
  POST /api/turn      -> one stateless turn for HTTP/serverless clients
  WS   /ws            -> the turn loop when WebSockets are available (local uvicorn)

Run:  python -m uvicorn main:app --reload --port 8000   ->  http://localhost:8000
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# override=True: this machine has stale machine-level env vars from other projects that can
# silently SHADOW .env (dotenv's default is never-override). On Vercel there's no .env file
# (.vercelignore), so this is a no-op there and platform env vars rule as before.
load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import db
from services import stt, tts, llm, directory
from services.prompts import scenario_of, norm_lang, opener_for

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    await tts.probe_elevenlabs()
    yield


app = FastAPI(title="Verba AI — voice agents for UA Agro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/crm")
async def crm_page():
    return FileResponse(str(STATIC_DIR / "crm.html"))


@app.get("/config")
async def config():
    # The page hits /config on load — use it to warm the ElevenLabs probe on a serverless
    # cold start so the FIRST spoken turn doesn't pay for the probe.
    if tts._eleven_ok is None:
        await tts.probe_elevenlabs()
    return {
        "brand": "Verba",
        "llm_ok": llm.llm_available(),
        "stt": "sarvam" if stt.stt_available() else "webspeech",
        "tts": tts.active_provider(),
        "voice_ok": tts.eleven_ok(),
        "voice_detail": tts.eleven_reason(),
        "model": llm.GEMINI_MODEL,
        "llm_keys": llm.key_count(),
        # Which voice each language actually resolves to on THIS deploy — lets you verify a
        # voice change went live at a glance (GET /config), instead of guessing from audio.
        "voices": {
            "english": tts._voice_for("english"),
            "hindi": tts._voice_for("hindi"),
            "telugu": tts._voice_for("telugu"),
        },
        "telugu_tts": tts.TELUGU_TTS,   # "sarvam" (fast/native) or "elevenlabs"
    }


@app.post("/api/login")
async def api_login(password: str = Form(default="")):
    """The access gate was removed — always open (kept so older cached pages still unlock)."""
    return {"ok": True}


@app.post("/api/crm")
async def api_crm(password: str = Form(default="")):
    return {"records": db.recent_crm()}


# Chosen language → Sarvam STT language_code.
_LANG_CODE = {"english": "en-IN", "hindi": "hi-IN", "telugu": "te-IN"}

# Spoken when the LLM fails mid-call — a graceful "say that again?" instead of a raw error.
_RETRY_LINE = {
    "english": "Sorry, the line broke for a second — could you say that again?",
    "hindi": "माफ़ कीजिए जी, आवाज़ कट गई थी — एक बार फिर बता दीजिए?",
    "telugu": "క్షమించండి అండి, లైన్ కట్ అయ్యింది — మరోసారి చెప్పండి?",
}

# False until this serverless instance serves its first /api/turn — surfaces cold-start cost
# in the per-turn timing telemetry.
_WARMED = False


def _pcm16_to_wav(pcm: bytes, sr: int = 16000) -> bytes:
    """Wrap raw mono 16-bit PCM (the streamed-mic frames) in a minimal WAV header."""
    import struct
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
           + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
           + b"data" + struct.pack("<I", len(pcm)))
    return hdr + pcm


@app.post("/api/fillers")
async def api_fillers(password: str = Form(default=""), scenario: str = Form(default=""),
                      lang: str = Form(default="")):
    """The 'one moment…' filler lines were removed at the user's request — the agent thinks
    silently. Endpoint kept (empty) so older cached pages don't error."""
    return {"fillers": []}


_opening_cache: dict[str, dict] = {}


async def _opening_audio(scenario: str, lng: str) -> tuple[str | None, str | None]:
    """Synthesized audio of the scenario's first line, cached per (provider, voice, scenario,
    lang) — the outbound intro plays instantly, no per-call TTS wait."""
    if tts._eleven_ok is None:
        await tts.probe_elevenlabs()
    key = f"{tts.active_provider()}::{tts._voice_for(lng)}::{tts._model_for(lng)}::{scenario}::{lng}"
    if key not in _opening_cache:
        audio_b64, mime = None, None
        try:
            a, m = await tts.synthesize(opener_for(scenario, lng), lng)
            if a:
                audio_b64, mime = base64.b64encode(a).decode("ascii"), m
        except Exception:
            pass
        _opening_cache[key] = {"audio_b64": audio_b64, "audio_mime": mime}
    cached = _opening_cache[key]
    return cached["audio_b64"], cached["audio_mime"]


@app.post("/api/opening")
async def api_opening(password: str = Form(default=""), scenario: str = Form(default="sales"),
                      lang: str = Form(default="")):
    """The scenario's first line in the chosen language. For the chat scenario it's the
    greeting shown before the customer types (text-only); for the outbound voice scenarios
    the page calls this just to WARM the audio cache — the line itself is delivered as the
    canned reply to the customer's pickup."""
    sc = scenario_of(scenario)
    lng = norm_lang(lang, scenario)
    text = opener_for(scenario, lng)
    if sc["chat"]:
        return {"text": text, "audio_b64": None, "audio_mime": None}
    audio_b64, mime = await _opening_audio(scenario, lng)
    return {"text": text, "audio_b64": audio_b64, "audio_mime": mime}


# Short spoken acknowledgements played the instant the caller stops talking, while the model is
# still thinking. The MEASURED reason they exist: Gemini's time-to-first-token on this tier is
# ~1.25s of FIXED overhead — a one-word reply costs the same as a full sentence, and removing
# thinkingLevel:minimal triples it — so that wait cannot be optimised away, only covered, the way
# a person says "mm-hm" while they think. Synthesized once per process; the client holds them in
# memory so playing one costs no network at all.
_ACK_LINES = {
    "hindi": ["जी…", "अच्छा…", "जी, समझ गई…", "हाँ जी…"],
    "english": ["Mm-hmm…", "Right…", "I see…", "Sure…"],
}
_ack_cache: dict[str, list] = {}


async def _ack_clips(lng: str) -> list[dict]:
    key = f"{tts.active_provider()}::{tts._voice_for(lng)}::{tts._model_for(lng)}::{lng}"
    if key in _ack_cache:
        return _ack_cache[key]
    out = []
    for line in _ACK_LINES.get(lng, _ACK_LINES["english"]):
        try:
            a, m = await tts.synthesize(line, lng)      # serial: the free tier allows 2 at once
        except Exception:                                # and the opener may still be rendering
            a, m = None, None
        if a:
            out.append({"audio_b64": base64.b64encode(a).decode("ascii"), "mime": m})
    _ack_cache[key] = out
    return out


@app.post("/api/acks")
async def api_acks(password: str = Form(default=""), scenario: str = Form(default="sales"),
                   lang: str = Form(default="")):
    lng = norm_lang(lang, scenario)
    if scenario_of(scenario)["chat"]:
        return {"acks": []}
    return {"acks": await _ack_clips(lng)}


@app.post("/api/say")
async def api_say(text: str = Form(default=""), password: str = Form(default=""),
                  scenario: str = Form(default=""), lang: str = Form(default="")):
    """TTS-only — speak a fixed line without invoking the LLM (used for no-reply nudges)."""
    lng = norm_lang(lang, scenario)
    audio_b64, mime = None, None
    try:
        a, m = await tts.synthesize(text, lng)
        if a:
            audio_b64, mime = base64.b64encode(a).decode("ascii"), m
    except Exception:
        pass
    return {"audio_b64": audio_b64, "audio_mime": mime}


# What the CRM row looks like for each tool — one place that turns tool args into the
# name / phone / summary / status the CRM page shows.
_ENQUIRY_OUTCOME_PRETTY = {
    "routine_enquiry": "Enquiry",
    "bulk_enquiry": "Bulk enquiry",
    "complaint": "Complaint",
    "wrong_number": "Wrong number",
    "off_topic": "Off-topic / no response",
}

_RSVP_OUTCOME_PRETTY = {
    "attending": "Attending ✓",
    "declined": "Declined",
    "callback_requested": "Callback requested",
    "no_commitment": "No commitment",
}

_ISSUE_TYPE_PRETTY = {
    "pest": "Pest issue", "disease": "Crop disease", "product_not_working": "Product issue",
    "delivery": "Delivery issue", "service_complaint": "Service complaint", "other": "Other issue",
}


def _crm_row(scenario: str, tool: str, args: dict) -> dict:
    a = {k: str(v).strip() for k, v in (args or {}).items() if v is not None}
    details = json.dumps(args or {}, ensure_ascii=False)
    if tool == "log_fap_enquiry":
        status = _ENQUIRY_OUTCOME_PRETTY.get(a.get("outcome", ""), a.get("outcome", "logged"))
        bits = [a.get("product_interest"), a.get("crop"),
                ("near " + a["village"]) if a.get("village") else None,
                a.get("notes")]
        return {"scenario": scenario, "kind": "enquiry", "name": a.get("name", ""),
                "phone": a.get("phone", ""), "summary": " · ".join(b for b in bits if b),
                "details": details, "status": status}
    if tool == "log_farmer_issue":
        kind_label = _ISSUE_TYPE_PRETTY.get(a.get("issue_type", ""), "Issue")
        status = "Resolved on call" if a.get("outcome") == "resolved_on_call" else "Needs follow-up"
        bits = [kind_label, a.get("crop"), a.get("description"), a.get("notes")]
        return {"scenario": scenario, "kind": "issue", "name": a.get("name", ""),
                "phone": a.get("phone", ""), "summary": " · ".join(b for b in bits if b),
                "details": details, "status": status}
    if tool == "log_rsvp":
        status = _RSVP_OUTCOME_PRETTY.get(a.get("outcome", ""), a.get("outcome", "logged"))
        bits = [a.get("guest_note"), a.get("notes")]
        return {"scenario": scenario, "kind": "rsvp", "name": a.get("name", ""),
                "phone": a.get("phone", ""), "summary": " · ".join(b for b in bits if b),
                "details": details, "status": status}
    return {"scenario": scenario, "kind": tool, "name": a.get("name", ""),
            "phone": a.get("phone", ""), "summary": details[:200], "details": details,
            "status": "new"}


def _handlers_for(scenario: str, captured: dict, on_row=None) -> dict:
    """Tool handlers. log_fap_enquiry/log_rsvp/log_farmer_issue write a CRM row (and push it,
    for the WS path); find_nearest_fap is a read-only lookup — its result feeds the model's
    next reply and never touches the CRM."""
    async def _save(tool: str, args: dict) -> dict:
        row = db.insert_crm(_crm_row(scenario, tool, args))
        captured["crm"] = row
        if on_row:
            await on_row(row)
        return row

    async def _lookup_fap(args: dict) -> dict:
        return directory.find_nearest_fap(args.get("place", ""))

    return {
        "log_fap_enquiry": lambda args: _save("log_fap_enquiry", args),
        "log_farmer_issue": lambda args: _save("log_farmer_issue", args),
        "log_rsvp": lambda args: _save("log_rsvp", args),
        "find_nearest_fap": lambda args: _lookup_fap(args),
    }


@app.post("/api/turn")
async def api_turn(
    text: str = Form(default=""),
    history: str = Form(default="[]"),
    password: str = Form(default=""),
    scenario: str = Form(default="sales"),
    lang: str = Form(default=""),
    audio: UploadFile = File(default=None),
):
    """Stateless turn for HTTP/serverless clients (Vercel has no WebSocket).
    The client carries the conversation history."""
    sc = scenario_of(scenario)
    lng = norm_lang(lang, scenario)
    try:
        contents = json.loads(history) if history else []
    except Exception:
        contents = []

    global _WARMED
    cold = not _WARMED
    _WARMED = True
    t_start = time.perf_counter()
    stt_ms = llm_ms = tts_ms = 0

    user_text = (text or "").strip()
    transcript = user_text
    if audio is not None:
        wav = await audio.read()
        t0 = time.perf_counter()
        try:
            transcript = await stt.transcribe_wav(wav, _LANG_CODE.get(lng, "en-IN"))
        except Exception as e:
            return {"error": f"stt: {e}", "history": contents}
        stt_ms = round((time.perf_counter() - t0) * 1000)
        user_text = transcript
    if not user_text:
        return {"error": "no input", "history": contents}

    # First line of every call is fixed and pre-synthesized — delivered instantly, no LLM
    # round-trip. Outbound: the intro after the customer's "Hello?" (or after 3s of silence —
    # the client sends a silent marker). Inbound (clinic): the greeting the moment the agent
    # picks up. The prompt says this line was already spoken, so the model continues from it.
    if not contents:
        intro = opener_for(scenario, lng)
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        contents.append({"role": "model", "parts": [{"text": intro}]})
        audio_b64, mime = await _opening_audio(scenario, lng)
        return {"transcript": transcript, "reply": intro, "crm": None, "history": contents,
                "audio_b64": audio_b64, "audio_mime": mime, "rest_text": None}

    captured = {"crm": None}
    t0 = time.perf_counter()
    try:
        reply = await llm.gemini_turn(contents, user_text,
                                      _handlers_for(scenario, captured),
                                      scenario=scenario, lang=lng)
    except Exception as e:
        # NEVER surface a raw error to the caller — log it server-side and speak a graceful
        # "say that again?" line instead; the conversation continues on the next turn.
        print(f"[llm-fail] {type(e).__name__}: {e}")
        reply = _RETRY_LINE.get(lng, _RETRY_LINE["english"])
        # gemini_turn already appended the user's turn; add only the graceful model line.
        contents.append({"role": "model", "parts": [{"text": reply}]})
    llm_ms = round((time.perf_counter() - t0) * 1000)

    # Chat scenario: text is the product — instant replies, no TTS spend.
    if sc["chat"]:
        return {"transcript": transcript, "reply": reply, "crm": captured["crm"],
                "history": contents, "audio_b64": None, "audio_mime": None, "rest_text": None}

    # Voice: synthesize ONLY the first sentence here and hand the rest back as text — the
    # client plays the first chunk immediately and fetches the remainder via /api/say while
    # it plays. Cuts the rest-of-reply TTS time out of the perceived response.
    chunks = _split_for_tts(reply)
    audio_b64, mime, rest_text = None, None, None
    if chunks:
        t0 = time.perf_counter()
        try:
            a, m = await tts.synthesize(chunks[0], lng)
            if a:
                audio_b64, mime = base64.b64encode(a).decode("ascii"), m
                if len(chunks) > 1:
                    rest_text = chunks[1]
        except Exception:
            pass
        tts_ms = round((time.perf_counter() - t0) * 1000)

    timing = {
        "stt_ms": stt_ms, "llm_ms": llm_ms, "tts_ms": tts_ms,
        "total_ms": round((time.perf_counter() - t_start) * 1000),
        "llm_attempts": llm.last_attempt_count, "served_by": llm.last_served_by,
        "cold": cold,
    }
    # One compact line per turn into the runtime logs — real production stage costs, always on.
    print(f"[timing] {timing['total_ms']}ms total | stt {stt_ms} | llm {llm_ms} "
          f"(x{timing['llm_attempts']} {timing['served_by']}) | tts {tts_ms} | cold={cold}")

    return {
        "transcript": transcript,
        "reply": reply,
        "crm": captured["crm"],
        "history": contents,
        "audio_b64": audio_b64,
        "audio_mime": mime,
        "rest_text": rest_text,
        "timing": timing,
    }


async def _send(ws: WebSocket, obj: dict):
    async with _lock_for(ws):
        await ws.send_text(json.dumps(obj, ensure_ascii=False))


def _lock_for(ws: WebSocket) -> asyncio.Lock:
    lock = getattr(ws, "_uaagro_send_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(ws, "_uaagro_send_lock", lock)
    return lock


async def _send_bytes(ws: WebSocket, data: bytes):
    """Every frame leaves under ONE per-connection lock. On the streaming path the audio
    forwarder task and the turn coroutine both write to this socket, and letting a text frame
    interleave into a binary one would corrupt the stream."""
    async with _lock_for(ws):
        await ws.send_bytes(data)


_SENT_END = re.compile(r"[.!?…।]\s")


def _split_for_tts(text: str) -> list[str]:
    """One continuous utterance for anything reply-sized — splitting sounds BROKEN on
    eleven_v3 (each chunk gets fresh prosody + dead air while chunk 2 renders, and v3
    pronounces short fragments worse). Replies are one sentence now, so only something
    unusually long (>160 chars) still gets the play-while-rendering split."""
    t = (text or "").strip()
    if len(t) < 160:
        return [t] if t else []
    m = _SENT_END.search(t)
    if not m:
        return [t]
    first, rest = t[: m.end()].strip(), t[m.end():].strip()
    return [first, rest] if rest else [t]


async def _process_text(ws: WebSocket, state: dict, text: str, silent: bool = False):
    """One full customer turn. `silent` hides the user echo (internal no-reply nudges)."""
    text = (text or "").strip()
    if not text:
        await _send(ws, {"type": "status", "state": "idle"})
        return
    sid = state["session_id"]
    # End-of-speech timestamp when this turn came from the mic — the only honest zero point for
    # "how long until the caller hears something", since it includes STT.
    t_turn0 = state.pop("t_turn0", None)
    scenario = state.get("scenario", "sales")
    sc = scenario_of(scenario)
    lng = norm_lang(state.get("lang", ""), scenario)
    if not silent:
        await _send(ws, {"type": "transcript", "role": "user", "text": text})
    db.log_turn(sid, "user", text)

    # First line of every call is fixed: outbound intro / inbound greeting, served instantly
    # (silent markers from the client trigger it too — e.g. no "hello" after 3s).
    if not state["contents"]:
        intro = opener_for(scenario, lng)
        state["contents"].append({"role": "user", "parts": [{"text": text}]})
        state["contents"].append({"role": "model", "parts": [{"text": intro}]})
        await _send(ws, {"type": "assistant_text", "role": "assistant", "text": intro})
        db.log_turn(sid, "assistant", intro)
        audio_b64, mime = await _opening_audio(scenario, lng)
        if audio_b64:
            audio = base64.b64decode(audio_b64)
            await _send(ws, {"type": "tts_audio_meta", "mime": mime, "bytes": len(audio)})
            await _send_bytes(ws, audio)
        await _send(ws, {"type": "status", "state": "idle"})
        return

    await _send(ws, {"type": "status", "state": "thinking"})

    captured = {"crm": None}
    t_turn = time.perf_counter()
    t_zero = t_turn0 or t_turn       # end-of-speech if known, else the start of thinking
    ttfa_ms = None
    first_clause_ms = None           # when the LLM had enough words to start synthesising

    async def on_row(row: dict):
        await _send(ws, {"type": "crm_created", "crm": row})
        db.log_turn(sid, "tool", "crm " + json.dumps(row, ensure_ascii=False))

    # ── STREAMED TURN ──────────────────────────────────────────────────────────────────────
    # Gemini's clauses go into the ElevenLabs socket as they are written; its audio goes out to
    # the browser as it comes back. The three stages OVERLAP instead of running end to end, so
    # the first sound lands while the model is still generating the rest of the sentence. If the
    # socket can't be opened (flag off, no websockets lib, dead key) `stream` stays None and the
    # blocking synth below runs exactly as it always did.
    stream = None
    if not sc["chat"]:
        cand = tts.ElevenStream(lng)
        if cand.usable():
            cand.start()             # handshake overlaps the LLM's time-to-first-token
            stream = cand

    fwd = None
    if stream is not None:
        async def _forward():
            nonlocal ttfa_ms
            first = True
            try:
                async for buf in stream.chunks():
                    if first:
                        first = False
                        ttfa_ms = round((time.perf_counter() - t_zero) * 1000)
                        await _send(ws, {"type": "tts_audio_meta", "mime": stream.mime,
                                         "sample_rate": stream.sample_rate, "streaming": True})
                        await _send(ws, {"type": "status", "state": "speaking"})
                    await _send_bytes(ws, buf)
            except Exception:
                pass          # caller hung up mid-reply — the turn is over, nothing to report
        fwd = asyncio.ensure_future(_forward())

    async def _on_clause(clause: str):
        nonlocal first_clause_ms
        if first_clause_ms is None:
            first_clause_ms = round((time.perf_counter() - t_zero) * 1000)
        await stream.feed(clause)

    try:
        assistant_text = await llm.gemini_turn(
            state["contents"], text,
            _handlers_for(scenario, captured, on_row), scenario=scenario, lang=lng,
            on_clause=(_on_clause if stream is not None else None),
        )
    except Exception as e:
        # Graceful recovery — log the real error, speak a "say that again?" line.
        print(f"[llm-fail] {type(e).__name__}: {e}")
        assistant_text = _RETRY_LINE.get(lng, _RETRY_LINE["english"])
        state["contents"].append({"role": "model", "parts": [{"text": assistant_text}]})
    llm_ms = round((time.perf_counter() - t_turn) * 1000)

    await _send(ws, {"type": "assistant_text", "role": "assistant", "text": assistant_text})
    db.log_turn(sid, "assistant", assistant_text)

    if sc["chat"]:                      # chat scenario: no TTS — instant text
        await _send(ws, {"type": "status", "state": "idle"})
        return

    t0 = time.perf_counter()
    rest = assistant_text
    if stream is not None:
        # What has the caller ALREADY heard? Exactly what the socket accepted. The final text is
        # normally a strict extension of it; it can also be CONTAINED in it (a tool re-ask, where
        # turn 1 was spoken and turn 2's line is the one returned), or diverge outright (a canned
        # fallback replacing a too-short model line). Only the genuinely unspoken remainder is
        # ever synthesised, so the caller never hears the same line twice.
        said, final = llm.norm_spoken(stream.spoken), llm.norm_spoken(assistant_text)
        if not said:
            rest = final
        elif final in said:
            rest = ""
        elif final.startswith(said):
            rest = final[len(said):].strip()
        else:
            rest = final
        if rest and stream.ok:
            await stream.feed(rest)
            rest = ""
        await stream.finish()
        if fwd is not None:
            try:
                await fwd
            except Exception:
                pass
        if not stream.got_audio:     # socket produced nothing at all — blocking synth still owes it
            rest = final

    if rest:
        await _send(ws, {"type": "status", "state": "speaking"})
        # Sentence-by-sentence, SEQUENTIAL on purpose: chunk 2 synthesizes while chunk 1 is
        # already playing in the browser, and staying at 1 concurrent request keeps the
        # ElevenLabs free tier (2-concurrent limit) from 429ing when fillers or /api/say overlap.
        for chunk in _split_for_tts(rest):
            try:
                audio, mime = await tts.synthesize(chunk, lng)
            except Exception:
                audio, mime = None, None
            if audio:
                if ttfa_ms is None:
                    ttfa_ms = round((time.perf_counter() - t_zero) * 1000)
                await _send(ws, {"type": "tts_audio_meta", "mime": mime, "bytes": len(audio)})
                await _send_bytes(ws, audio)
    tts_ms = round((time.perf_counter() - t0) * 1000)
    # first-audio is THE number that matters to a caller; the rest is bookkeeping.
    print(f"[timing/ws] first-audio {ttfa_ms if ttfa_ms is not None else '-'}ms | "
          f"first-clause {first_clause_ms if first_clause_ms is not None else '-'}ms | "
          f"llm {llm_ms} (x{llm.last_attempt_count} {llm.last_served_by}) | tts {tts_ms} | "
          f"stream={'on' if stream is not None else 'off'}", flush=True)
    await _send(ws, {"type": "status", "state": "idle"})


async def _process_audio(ws: WebSocket, state: dict, wav: bytes):
    # Zero point for time-to-first-audio: the caller has stopped speaking and everything from
    # here — STT included — is wait. _process_text pops it.
    state["t_turn0"] = time.perf_counter()
    await _send(ws, {"type": "status", "state": "transcribing"})
    lng = norm_lang(state.get("lang", ""), state.get("scenario", "sales"))
    t0 = time.perf_counter()
    try:
        text = await stt.transcribe_wav(wav, _LANG_CODE.get(lng, "en-IN"))
    except Exception as e:
        await _send(ws, {"type": "error", "where": "stt", "message": str(e), "recoverable": True})
        await _send(ws, {"type": "status", "state": "idle"})
        return
    print(f"[timing/ws] stt {round((time.perf_counter() - t0) * 1000)}ms | {len(wav)}B wav")
    if not text:
        await _send(ws, {"type": "status", "state": "idle", "detail": "no speech detected"})
        return
    await _process_text(ws, state, text)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state = {"session_id": uuid.uuid4().hex, "contents": [], "scenario": "sales", "lang": ""}
    db.ensure_conversation(state["session_id"])
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                # Streamed-mic mode: between mic_start/mic_end the binary frames are raw 16k
                # PCM16 chunks shipped WHILE the caller speaks — buffer them. Otherwise it's
                # the legacy single complete-WAV upload.
                if state.get("mic_frames") is not None:
                    state["mic_frames"].append(msg["bytes"])   # kept as the batch fallback
                    sstream = state.get("stt_stream")
                    if sstream is not None:
                        await sstream.feed(msg["bytes"])       # …and transcribed as it arrives
                else:
                    await _process_audio(ws, state, msg["bytes"])
                continue

            raw = msg.get("text")
            if raw is None:
                continue
            data = json.loads(raw)
            mtype = data.get("type")

            if mtype == "mic_start":
                state["mic_frames"] = []
                # Open Sarvam's socket NOW and transcribe frames as they arrive, so the transcript
                # is ready the moment the caller stops rather than after a whole-utterance POST.
                lng0 = norm_lang(state.get("lang", ""), state.get("scenario", "sales"))
                sstream = stt.SarvamStream(_LANG_CODE.get(lng0, "en-IN"))
                sstream.start()
                state["stt_stream"] = sstream if sstream.usable() else None
            elif mtype == "mic_end":
                frames = state.get("mic_frames") or []
                state["mic_frames"] = None
                sstream = state.pop("stt_stream", None)
                text = ""
                if frames and sstream is not None:
                    state["t_turn0"] = time.perf_counter()      # the caller stopped speaking HERE
                    t_stt = time.perf_counter()
                    try:
                        text = await sstream.finish()
                    except Exception:
                        text = ""
                    print(f"[timing/ws] stt-stream {round((time.perf_counter() - t_stt) * 1000)}ms"
                          f" | {'hit' if text else 'MISS -> batch'}", flush=True)
                elif sstream is not None:
                    await sstream.cancel()
                if text:
                    await _process_text(ws, state, text)
                elif frames:
                    # Socket failed or returned nothing — the frames were buffered all along, so
                    # this costs correctness nothing, only the latency saving.
                    await _process_audio(ws, state, _pcm16_to_wav(b"".join(frames)))
            elif mtype == "mic_abort":
                state["mic_frames"] = None
                sstream = state.pop("stt_stream", None)
                if sstream is not None:
                    await sstream.cancel()
            elif mtype == "hello":
                if data.get("scenario"):
                    state["scenario"] = data["scenario"]
                if data.get("lang"):
                    state["lang"] = data["lang"]
                await _send(ws, {"type": "status", "state": "idle", "detail": "connected"})
            elif mtype == "turn_text":
                if data.get("scenario"):
                    state["scenario"] = data["scenario"]
                if data.get("lang"):
                    state["lang"] = data["lang"]
                await _process_text(
                    ws, state, data.get("text", ""), silent=bool(data.get("silent"))
                )
            elif mtype == "control":
                action = data.get("action")
                if action == "restart":
                    state["session_id"] = uuid.uuid4().hex
                    state["contents"] = []
                    db.ensure_conversation(state["session_id"])
                    await _send(ws, {"type": "status", "state": "idle", "detail": "restarted"})
                elif action == "stop":
                    await _send(ws, {"type": "status", "state": "idle", "detail": "stopped"})
    except WebSocketDisconnect:
        pass
    except Exception:
        # keep the server alive; the client will reconnect on next Start
        try:
            await ws.close()
        except Exception:
            pass
