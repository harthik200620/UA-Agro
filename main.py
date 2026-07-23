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

import base64
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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
    """Tool handlers. log_fap_enquiry/log_rsvp write a CRM row (and push it, for the WS path);
    find_nearest_fap is a read-only lookup — its result feeds the model's next reply and never
    touches the CRM."""
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

    user_text = (text or "").strip()
    transcript = user_text
    if audio is not None:
        wav = await audio.read()
        try:
            transcript = await stt.transcribe_wav(wav, _LANG_CODE.get(lng, "en-IN"))
        except Exception as e:
            return {"error": f"stt: {e}", "history": contents}
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
        try:
            a, m = await tts.synthesize(chunks[0], lng)
            if a:
                audio_b64, mime = base64.b64encode(a).decode("ascii"), m
                if len(chunks) > 1:
                    rest_text = chunks[1]
        except Exception:
            pass

    return {
        "transcript": transcript,
        "reply": reply,
        "crm": captured["crm"],
        "history": contents,
        "audio_b64": audio_b64,
        "audio_mime": mime,
        "rest_text": rest_text,
    }


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj, ensure_ascii=False))


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
            await ws.send_bytes(audio)
        await _send(ws, {"type": "status", "state": "idle"})
        return

    await _send(ws, {"type": "status", "state": "thinking"})

    captured = {"crm": None}

    async def on_row(row: dict):
        await _send(ws, {"type": "crm_created", "crm": row})
        db.log_turn(sid, "tool", "crm " + json.dumps(row, ensure_ascii=False))

    try:
        assistant_text = await llm.gemini_turn(
            state["contents"], text,
            _handlers_for(scenario, captured, on_row), scenario=scenario, lang=lng,
        )
    except Exception as e:
        # Graceful recovery — log the real error, speak a "say that again?" line.
        print(f"[llm-fail] {type(e).__name__}: {e}")
        assistant_text = _RETRY_LINE.get(lng, _RETRY_LINE["english"])
        state["contents"].append({"role": "model", "parts": [{"text": assistant_text}]})

    await _send(ws, {"type": "assistant_text", "role": "assistant", "text": assistant_text})
    db.log_turn(sid, "assistant", assistant_text)

    if sc["chat"]:                      # chat scenario: no TTS — instant text
        await _send(ws, {"type": "status", "state": "idle"})
        return

    await _send(ws, {"type": "status", "state": "speaking"})
    # Sentence-by-sentence, SEQUENTIAL on purpose: chunk 2 synthesizes while chunk 1 is already
    # playing in the browser, and staying at 1 concurrent request keeps the ElevenLabs free
    # tier (2-concurrent limit) from 429ing when fillers or /api/say overlap.
    for chunk in _split_for_tts(assistant_text):
        try:
            audio, mime = await tts.synthesize(chunk, lng)
        except Exception:
            audio, mime = None, None
        if audio:
            await _send(ws, {"type": "tts_audio_meta", "mime": mime, "bytes": len(audio)})
            await ws.send_bytes(audio)
    await _send(ws, {"type": "status", "state": "idle"})


async def _process_audio(ws: WebSocket, state: dict, wav: bytes):
    await _send(ws, {"type": "status", "state": "transcribing"})
    lng = norm_lang(state.get("lang", ""), state.get("scenario", "sales"))
    try:
        text = await stt.transcribe_wav(wav, _LANG_CODE.get(lng, "en-IN"))
    except Exception as e:
        await _send(ws, {"type": "error", "where": "stt", "message": str(e), "recoverable": True})
        await _send(ws, {"type": "status", "state": "idle"})
        return
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
                await _process_audio(ws, state, msg["bytes"])
                continue

            raw = msg.get("text")
            if raw is None:
                continue
            data = json.loads(raw)
            mtype = data.get("type")

            if mtype == "hello":
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
