"""The agent brain: Google Gemini with function-calling, scenario-aware.

gemini_turn() appends the user's utterance to the running conversation, calls Gemini with the
active scenario's system prompt + tools (find_nearest_fap / log_fap_enquiry / log_rsvp), runs
the matching handler, and returns the agent's reply. `contents` is mutated in place.

Most tools here are WRITES (log_fap_enquiry, log_rsvp) — the model already knows what to say
when it calls one, so on success the fast path below speaks the model's own same-turn text and
skips a second Gemini round-trip. find_nearest_fap is a READ: the model can't know the answer
until the lookup runs, so it gets a genuine second turn instead (see _QUERY_TOOLS).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import httpx

from . import _http
from .prompts import build_system_prompt, tools_for, norm_lang, GOSTHI_CASE

def _clean(name: str, default: str = "") -> str:
    """Read an env var, removing BOM/zero-width chars plus quotes/whitespace."""
    v = os.getenv(name, default) or ""
    for ch in (chr(0xFEFF), chr(0x200B), chr(0x200C), chr(0x200D)):
        v = v.replace(ch, "")
    return v.strip().strip('"').strip("'").strip()


def _load_keys() -> list[str]:
    """Gather Gemini API keys for rotation: a comma-separated GEMINI_API_KEYS, plus the
    numbered GEMINI_API_KEY / GEMINI_API_KEY_2 … GEMINI_API_KEY_100 vars (add more keys by
    just adding env vars — no code change, as long as you stay under _100). Deduped, empties
    dropped."""
    raw = []
    combo = _clean("GEMINI_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    raw.append(_clean("GEMINI_API_KEY"))
    for n in range(2, 101):
        raw.append(_clean(f"GEMINI_API_KEY_{n}"))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_KEYS = _load_keys()
_key_idx = 0   # round-robins one step per request (spreads load) + advances on quota/invalid

# Model choice. We want the BEST Flash model and NEVER the -lite tier (lite is the weak,
# bot-like one). So:
#   • empty / garbled env      → gemini-flash-latest
#   • a -lite id in the env    → OVERRIDDEN to gemini-flash-latest (so a lite value pinned in the
#                                hosting env can't force the weak model — no env edit needed)
#   • an explicit non-lite id  → honoured
# "-latest" is a rolling alias to the newest stable Flash, valid across all mixed-age keys
# (pinned ids get retired for newer accounts). Force a specific model with a non-lite GEMINI_MODEL.
_BEST_MODEL = "gemini-flash-latest"
_raw_model = _clean("GEMINI_MODEL")
if re.fullmatch(r"gemini-[A-Za-z0-9.\-]+", _raw_model) and "lite" not in _raw_model.lower():
    GEMINI_MODEL = _raw_model
else:
    GEMINI_MODEL = _BEST_MODEL
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _int_env(name: str, default: int) -> int:
    try:
        return int(_clean(name, str(default)) or default)
    except ValueError:
        return default


# HYBRID MODEL POOL: some keys (typically newer-account keys) 404 on GEMINI_MODEL ("no longer
# available to new users") while older-account keys serve it fine. Rather than wasting a
# request on every 404 before rotating past them, each key gets its OWN model up front: the
# first GEMINI_PRIMARY_KEY_COUNT keys (.env order) use GEMINI_MODEL; every key after that uses
# GEMINI_MODEL_FALLBACK. Unset -> every key uses GEMINI_MODEL (today's single-model behavior —
# AgriSetu's keys 1-27 serve GEMINI_MODEL (2.5-flash), keys 28-40 serve GEMINI_MODEL_FALLBACK
# (3-flash-preview) — same active hybrid pool as the sibling TradeCredit/PetSecure builds.
_raw_fallback = _clean("GEMINI_MODEL_FALLBACK")
GEMINI_MODEL_FALLBACK = (
    _raw_fallback if re.fullmatch(r"gemini-[A-Za-z0-9.\-]+", _raw_fallback) else GEMINI_MODEL
)
_PRIMARY_KEY_COUNT = _int_env("GEMINI_PRIMARY_KEY_COUNT", 10**9)


def _model_for_key_idx(idx: int) -> str:
    return GEMINI_MODEL if idx < _PRIMARY_KEY_COUNT else GEMINI_MODEL_FALLBACK


# Let the model THINK briefly before it answers — this is what turns a reflexive, bot-like
# reply into a considered, wise one (understand the intent, then respond). Costs a little
# latency. Tune with GEMINI_THINKING_BUDGET (0 = off, back to instant snap replies).
_THINKING_BUDGET = max(0, _int_env("GEMINI_THINKING_BUDGET", 512))


def _is_gemini3(model: str) -> bool:
    return model.startswith("gemini-3")


def _thinking_config_for(model: str) -> dict | None:
    """Gemini 3 models think BY DEFAULT and their thoughts share the output-token pool: with a
    small cap the thoughts (~200 tokens) ate the whole budget and replies came out TRUNCATED.
    So 3-series gets thinkingLevel "minimal" (fastest valid level — budget knobs are 2.5-era
    API); 2.5-series keeps the tunable budget; -lite gets no thinking (latency)."""
    if _is_gemini3(model):
        return {"thinkingLevel": "minimal"}
    if "2.5" in model or model.endswith("-latest"):
        eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
        return {"thinkingBudget": eff}
    return None


def _max_tokens_for(model: str) -> int:
    """When thinking is on, the visible answer shares the token pool with the thinking, so give
    it generous headroom (the prompt still keeps the spoken reply to one short sentence)."""
    if _is_gemini3(model):
        return 1024
    eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
    return max(1024, eff + 512) if eff > 0 else 220

# Fields each tool needs before it may fire; the server enforces this even if the model rushes.
_REQUIRED_BY_TOOL = {
    "find_nearest_fap": ("place",),
    "log_fap_enquiry": ("outcome",),
    "log_rsvp": ("outcome",),
}

# READ tools: the model can't know the answer until the lookup runs, so unlike the write-tools
# below, these get a genuine second Gemini turn with the real result instead of a same-turn
# fast-path guess. No call-limit — the prompt tells the model to call it again per new place.
_QUERY_TOOLS = {"find_nearest_fap"}


def _reask(lang: str) -> str:
    """Generic 'sorry, could you say that again?' in the scenario's language."""
    return {
        "telugu": "క్షమించండి అండి, ఒక్కసారి మళ్ళీ చెప్తారా?",
        "hindi": "माफ़ कीजिए जी, एक बार फिर बता दीजिए?",
    }.get(lang, "Sorry, could you say that again?")


def _fallback_for(tool: str | None, args: dict | None, lang: str = "english") -> str:
    """Tailored confirmation for a successful tool call, in the CHOSEN language — every tool ×
    every language (also the fallback if a follow-up generation fails AFTER the tool already
    saved). Built locally from the tool args, so it is instant and never needs a second Gemini
    call."""
    a = args or {}
    lang = lang if lang in ("english", "hindi") else "english"

    if tool == "log_fap_enquiry":
        outcome = str(a.get("outcome") or "").strip().lower()
        if lang == "hindi":
            return {
                "bulk_enquiry": "नोट कर लिया जी — टीम बल्क ऑर्डर पर बात करेगी।",
                "complaint": "नोट कर लिया जी — टीम इसे देखेगी और आपसे संपर्क करेगी।",
                "wrong_number": "कोई बात नहीं जी, परेशान करने के लिए माफ़ी!",
                "off_topic": "किसान सेवा केंद्र को कॉल करने के लिए धन्यवाद जी!",
            }.get(outcome, "कॉल के लिए धन्यवाद जी — दुकान पर ज़रूर आइए!")
        return {
            "bulk_enquiry": "Noted sir — our team will follow up on the bulk order.",
            "complaint": "I've noted it, sir — our team will look into this and get back to you.",
            "wrong_number": "No problem, sir — sorry to bother you!",
            "off_topic": "Thanks for calling Kisan Sewa Kendra, sir!",
        }.get(outcome, "Thanks for calling, sir — do visit the store!")

    if tool == "log_rsvp":
        outcome = str(a.get("outcome") or "").strip().lower()
        g = GOSTHI_CASE
        # Fully-silent call → leave a complete voicemail-style message: who called, the Gosthi
        # date/time/place. The whole value of the call in one go.
        if "no response" in str(a.get("notes") or "").lower():
            if lang == "hindi":
                return (f"{g['name']} जी, शायद आवाज़ नहीं आ रही — मैं कविता, किसान सेवा केंद्र से। "
                        f"{g['date_hi']} {g['time_hi']} {g['location_hi']} किसान गोष्ठी है — "
                        "ज़रूर आइएगा। धन्यवाद!")
            return (f"{g['name']}, seems I can't hear you — this is Kavita from Kisan Sewa "
                    f"Kendra. There's a Kisan Gosthi {g['date_en']} at {g['time']} at "
                    f"{g['location']} — do come by if you can. Thank you!")
        if lang == "hindi":
            return {
                "attending": "बहुत बढ़िया जी — तो वहाँ मिलते हैं!",
                "declined": "कोई बात नहीं जी — ख़याल रखिएगा!",
                "callback_requested": "ज़रूर जी — मैं फिर कॉल करूँगी। धन्यवाद!",
            }.get(outcome, "ठीक है जी — उम्मीद है आप आएँगे। धन्यवाद!")
        return {
            "attending": "Wonderful — we'll see you there!",
            "declined": "No problem at all — take care!",
            "callback_requested": "Of course — I'll call you again later. Thank you!",
        }.get(outcome, "Alright — hope to see you there. Thank you!")

    return {"hindi": "ठीक है जी, हो गया।"}.get(lang, "Done.")


_JUNK_NAMES = {"n/a", "na", "none", "null", "unknown", "customer", "guest", "test", "xxx", "abc"}


def _validate_args(tool: str, args: dict) -> str | None:
    """Deterministic guards the model can't rush past. Neither tool in this project needs one
    (find_nearest_fap just needs a non-empty place, already enforced by _REQUIRED_BY_TOOL;
    log_rsvp's identity is force-set, never user-supplied) — kept as the extension point the
    retry loop already calls generically. Returns an error message or None."""
    return None


def llm_available() -> bool:
    return bool(_KEYS)


def key_count() -> int:
    return len(_KEYS)


_IST = timezone(timedelta(hours=5, minutes=30))


def _today() -> str:
    """Current date AND time in Hyderabad (IST) — explicit tz because Vercel runs in UTC."""
    now = datetime.now(_IST)
    return now.strftime("%A, %Y-%m-%d, current time %I:%M %p IST")


def _should_rotate(status: int, text: str) -> bool:
    """Rotate to the next key on quota (429), key-permission errors, a per-key model
    retirement (404 'no longer available to new users' — other keys may still have it), or a
    dead/disabled key (401 — e.g. its bound service account was deleted; other keys are fine)."""
    if status in (401, 429, 404):
        return True
    if status in (400, 403):
        t = (text or "").upper()
        return any(s in t for s in ("API_KEY_INVALID", "API KEY NOT VALID", "QUOTA", "PERMISSION_DENIED"))
    return False


async def _generate(contents: list, scenario: str = "lead", lang: str = "",
                    force_tool: bool = False) -> dict:
    global _key_idx
    if not _KEYS:
        raise RuntimeError("No Gemini API key set")
    system_text = build_system_prompt(_today(), scenario, lang)
    tools = [{"functionDeclarations": tools_for(scenario)}]
    # "ANY" FORCES a function call — used for must-record turns (the client's close note),
    # where AUTO mode too often speaks the goodbye and skips the tool.
    tool_config = {"functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}}
    last_err = None
    client = _http.client()  # shared keep-alive client (no per-call TLS handshake)
    # Round-robin: each new request starts on the NEXT key, so the per-key free-tier rate
    # limit (RPM/RPD) is spread across all keys instead of hammering one until it 429s —
    # effective throughput ≈ single-key limit × number of keys.
    if len(_KEYS) > 1:
        _key_idx = (_key_idx + 1) % len(_KEYS)
    # Then try keys starting there; rotate past any that are quota'd/invalid this turn. Each
    # key uses ITS OWN model (see _model_for_key_idx) — a mixed pool serves every key's real
    # capacity instead of wasting a call on every 404 before finding one that works.
    for _ in range(len(_KEYS)):
        model = _model_for_key_idx(_key_idx)
        gen_config = {"temperature": 0.7, "maxOutputTokens": _max_tokens_for(model)}
        thinking = _thinking_config_for(model)
        if thinking:
            gen_config["thinkingConfig"] = thinking
        body = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
            "generationConfig": gen_config,
        }
        url = _URL.format(model=model)
        resp = await client.post(url, params={"key": _KEYS[_key_idx]}, json=body)
        if resp.status_code < 400:
            return resp.json()
        last_err = f"Gemini {resp.status_code} (key {_key_idx + 1}/{len(_KEYS)}, {model}): {resp.text[:160]}"
        if _should_rotate(resp.status_code, resp.text):
            _key_idx = (_key_idx + 1) % len(_KEYS)
            continue
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("All Gemini keys exhausted — " + (last_err or "quota/invalid"))


async def gemini_turn(contents: list, user_text: str, handlers: dict, scenario: str = "sales",
                      lang: str = "") -> str:
    """Run one customer turn.

    handlers: {tool_name: async fn(args)->result_dict}. Returns the agent's reply text.
    `scenario` (sales/gosthi) selects the persona and tool set; `lang` (english/hindi) selects
    the spoken language — empty falls back to the scenario's showcase default.
    """
    lang = norm_lang(lang, scenario)
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    last_tool, last_args = None, None
    # The client's close note names the tool it needs ("… CALL log_fap_enquiry …" / "… CALL
    # log_rsvp …") — force function-calling on those turns so the outcome is ALWAYS recorded.
    force_tool = "(System note" in (user_text or "") and "CALL " in (user_text or "")

    for _ in range(5):  # allow a couple of tool round-trips
        try:
            data = await _generate(contents, scenario, lang,
                                   force_tool=force_tool and last_tool is None)
        except Exception:
            # If a tool already saved this turn, give a graceful spoken confirmation instead
            # of surfacing a raw error (e.g. when the follow-up call hits a Gemini 429).
            if last_tool:
                return _fallback_for(last_tool, last_args, lang)
            raise
        candidates = data.get("candidates") or []
        if not candidates:
            break
        parts = (candidates[0].get("content") or {}).get("parts") or []

        text_chunks, fcall = [], None
        for p in parts:
            if "text" in p:
                text_chunks.append(p["text"])
            if "functionCall" in p:
                fcall = p["functionCall"]

        # Persist the model's turn (echo functionCall back so Gemini keeps the thread).
        contents.append({"role": "model", "parts": parts})

        if fcall and fcall.get("name") in handlers:
            name = fcall["name"]
            args = dict(fcall.get("args") or {})
            if name == "log_rsvp":              # outbound Gosthi call — the farmer is known;
                args["name"] = GOSTHI_CASE["name"]    # FORCE, don't setdefault — the model
                args["phone"] = GOSTHI_CASE["phone"]  # sometimes fills a placeholder-looking
                                                       # guess instead of leaving these for us.
            if name == "log_fap_enquiry":       # inbound, ANONYMOUS caller — never invent a
                nm = str(args.get("name") or "").strip()     # name/phone they didn't actually
                if len(nm) < 2 or nm.lower() in _JUNK_NAMES:  # offer; silently blank a
                    args["name"] = ""                          # placeholder-looking value
                digits = re.sub(r"\D", "", str(args.get("phone") or ""))  # instead of
                if digits and len(digits) < 10:                            # reject-and-retry
                    args["phone"] = ""       # (that would force an awkward re-ask of info an
                                              # anonymous caller never offered in the first place)
            missing = [k for k in _REQUIRED_BY_TOOL.get(name, ()) if not args.get(k)]
            if missing:
                response = {
                    "status": "error",
                    "message": "Missing " + ", ".join(missing)
                    + ". Politely ask the customer for these before proceeding.",
                }
                contents.append({"role": "user",
                                 "parts": [{"functionResponse": {"name": name, "response": response}}]})
                continue  # let the model ask for the missing fields

            problem = _validate_args(name, args)
            if problem:
                response = {"status": "error", "message": problem}
                contents.append({"role": "user",
                                 "parts": [{"functionResponse": {"name": name, "response": response}}]})
                continue  # let the model relay the problem and re-collect

            row = await handlers[name](args)
            if row is None:
                response = {
                    "status": "error",
                    "message": "Could not save the record. Apologise briefly in the caller's "
                    "language and offer to note the details again.",
                }
                contents.append({"role": "user",
                                 "parts": [{"functionResponse": {"name": name, "response": response}}]})
                continue  # let the model explain / recover

            if name in _QUERY_TOOLS:
                # READ, not a write — the model could not know the answer before the lookup ran,
                # so (unlike the write-tools' fast path below) it needs a genuine second turn
                # with the REAL result to compose an informed reply, not a same-turn guess.
                last_tool, last_args = name, args
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": name, "response": row}}]})
                continue

            # SUCCESS — speak and SKIP the second Gemini call (halves tool-turn latency).
            # Prefer the model's OWN text from this same turn when it wrote one (it answers
            # whatever the customer just asked — e.g. "how much?" — far smarter than a canned
            # line); the local confirmation is the fallback when the turn was tool-only.
            last_tool, last_args = name, args
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": name, "response": {"status": "success", "id": row.get("id")}}}]})
            own = re.sub(r"\(System[^)]*\)", "", "".join(text_chunks))
            own = re.sub(r"\s*\n+\s*", " ", own).strip()  # one spoken line, never split
            spoken = own if len(own) >= 8 else _fallback_for(name, args, lang)
            contents.append({"role": "model", "parts": [{"text": spoken}]})
            return spoken

        final = "".join(text_chunks).strip()
        # flash-lite sometimes parrots internal "(System note …)" instructions into its reply —
        # strip them so they are never shown or spoken to the customer.
        final = re.sub(r"\(System[^)]*\)", "", final)
        final = re.sub(r"\s*\n+\s*", " ", final).strip()  # one spoken line, never split
        if not final:
            final = (
                _fallback_for(last_tool, last_args, lang)
                if last_tool
                else _reask(lang)
            )
        return final

    return _reask(lang)
