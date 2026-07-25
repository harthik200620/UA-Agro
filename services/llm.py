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

import asyncio
import json
import os
import re
import time
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
    numbered GEMINI_API_KEY / GEMINI_API_KEY_2 … GEMINI_API_KEY_150 vars (add more keys by
    just adding env vars — no code change, as long as you stay under _150). Deduped, empties
    dropped. Order matters: it's preserved end-to-end (see _key_tiers below), so append new
    keys, never insert/reorder existing ones."""
    raw = []
    combo = _clean("GEMINI_API_KEYS")
    if combo:
        raw += [p.strip() for p in combo.split(",")]
    raw.append(_clean("GEMINI_API_KEY"))
    for n in range(2, 151):
        raw.append(_clean(f"GEMINI_API_KEY_{n}"))
    out, seen = [], set()
    for k in raw:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_KEYS = _load_keys()

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


# LAST-RESORT MODEL. GEMINI_MODEL_FALLBACK is deliberately the SAME model as GEMINI_MODEL (see
# .env for why moving the whole tier to a second model failed), which means the stack has had NO
# model-level redundancy at all — and production logs show that costing real turns: a 503 "this
# model is currently experiencing high demand" killed a turn outright, and no number of keys can
# route around a model-wide outage.
#
# So: a different model, tried ONLY after _LAST_RESORT_AFTER keys have already failed this turn.
# That ordering is the whole design. It can never become the workhorse (which is exactly the
# mistake recorded in .env), it costs a healthy turn nothing, and it only ever runs on turns that
# were otherwise about to be lost — where a different model with its own separate quota is the
# only thing that can still answer. gemini-3-flash-preview measured 1556ms TTFT on 6/6 keys.
_LAST_RESORT_MODEL = _clean("GEMINI_LAST_RESORT_MODEL", "gemini-3-flash-preview")
if not re.fullmatch(r"gemini-[A-Za-z0-9.\-]*", _LAST_RESORT_MODEL or ""):
    _LAST_RESORT_MODEL = ""
_LAST_RESORT_AFTER = max(2, _int_env("GEMINI_LAST_RESORT_AFTER", 6))


# PRIORITY KEY TIERS: newly-issued keys are far less contended than the original pool, so
# they should absorb traffic first. GEMINI_FRESH_KEY_COUNT keys are the FRESH tier — the LAST
# that many keys loaded (freshest batches are always appended, never inserted, see
# _load_keys) — tried first on every request. The OTHER tier (everything before them) is pure
# reserve: touched only once every fresh-tier key has 429'd/failed on THIS request, never
# time-shared evenly with the fresh pool the way a flat round-robin would.
_FRESH_KEY_COUNT = _int_env("GEMINI_FRESH_KEY_COUNT", 0)


def _key_tiers() -> tuple[list[int], list[int]]:
    n = len(_KEYS)
    if _FRESH_KEY_COUNT <= 0 or _FRESH_KEY_COUNT >= n:
        return list(range(n)), []
    split = n - _FRESH_KEY_COUNT
    return list(range(split, n)), list(range(split))


_FRESH_ORDER, _OTHER_ORDER = _key_tiers()
_fresh_idx = 0
_other_idx = 0

# HEDGED REQUESTS (staggered by default): a backup request on a second key fires only if the
# first is slower than GEMINI_HEDGE_AFTER_MS. Simultaneous racing (GEMINI_HEDGE=2+ with a tiny
# HEDGE_AFTER_MS) was measured harmful on a shared free-tier pool elsewhere in this project
# family: doubled burst rate 429'd the fresh tier within a handful of turns, median got WORSE,
# then cascaded into failed turns. The stagger keeps the median untouched and still cuts the
# rare multi-second stalls.
_HEDGE = max(1, _int_env("GEMINI_HEDGE", 1))
_HEDGE_AFTER_MS = max(250, _int_env("GEMINI_HEDGE_AFTER_MS", 3500))


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
    API); 2.5-series keeps the tunable budget; -lite gets no thinking (latency).
    "-latest" aliases now resolve to 3-series server-side (thinkingBudget → 400 invalid-
    argument; thinkingLevel minimal → ~1.2s; NO config → ~4s because default thinking turns
    on) — so -latest is treated as 3-series, never given a budget."""
    if _is_gemini3(model) or model.endswith("-latest"):
        return {"thinkingLevel": "minimal"}
    if "2.5" in model:
        eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
        return {"thinkingBudget": eff}
    return None


def _max_tokens_for(model: str) -> int:
    """When thinking is on, the visible answer shares the token pool with the thinking, so give
    it generous headroom (the prompt still keeps the spoken reply to one short sentence)."""
    if _is_gemini3(model) or model.endswith("-latest"):
        return 1024
    eff = 0 if "lite" in model.lower() else _THINKING_BUDGET
    return max(1024, eff + 512) if eff > 0 else 220

# Fields each tool needs before it may fire; the server enforces this even if the model rushes.
_REQUIRED_BY_TOOL = {
    "find_nearest_fap": ("place",),
    "log_fap_enquiry": ("outcome",),
    "log_rsvp": ("outcome",),
    "log_farmer_issue": ("issue_type", "outcome"),
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

    if tool == "log_farmer_issue":
        outcome = str(a.get("outcome") or "").strip().lower()
        if lang == "hindi":
            return {
                "resolved_on_call": "ठीक है जी — यह तरीका आज़माइए, फ़र्क़ दिखेगा।",
                "needs_followup": "नोट कर लिया जी — हमारी टीम आपसे जल्द संपर्क करेगी।",
            }.get(outcome, "नोट कर लिया जी — टीम आपसे संपर्क करेगी।")
        return {
            "resolved_on_call": "Alright sir — try that, you should see the difference.",
            "needs_followup": "Noted sir — our team will get in touch with you shortly.",
        }.get(outcome, "Noted sir — our team will follow up.")

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
    retirement (404 'no longer available to new users' — other keys may still have it), a
    dead/disabled key (401 — e.g. its bound service account was deleted; other keys are fine),
    or a server-side outage (5xx — 'model experiencing high demand' hits one model's pool, and
    the tiers run different models, so falling through to the next tier usually still answers)."""
    if status in (401, 429, 404) or status >= 500:
        return True
    if status in (400, 403):
        t = (text or "").upper()
        return any(s in t for s in ("API_KEY_INVALID", "API KEY NOT VALID", "QUOTA", "PERMISSION_DENIED"))
    return False


# Per-instance latency telemetry + quota memory. `last_attempt_count`/`last_served_by` describe
# the most recent _generate() call (read by main.py's timing line). `_cooldown` remembers keys
# that just 429'd (60s) or 5xx'd (15s) so a quota-dead key costs ONE probe per window instead of
# one failed round-trip on EVERY request.
last_attempt_count = 0
last_served_by = ""
_cooldown: dict[int, float] = {}


async def _generate(contents: list, scenario: str = "lead", lang: str = "",
                    force_tool: bool = False, hedge: bool = True) -> dict:
    global _fresh_idx, _other_idx, last_attempt_count, last_served_by
    if not _KEYS:
        raise RuntimeError("No Gemini API key set")
    system_text = build_system_prompt(_today(), scenario, lang)
    tools = [{"functionDeclarations": tools_for(scenario)}]
    # "ANY" FORCES a function call — used for must-record turns (the client's close note),
    # where AUTO mode too often speaks the goodbye and skips the tool.
    tool_config = {"functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}}
    last_err = None
    client = _http.client()  # shared keep-alive client (no per-call TLS handshake)
    last_attempt_count = 0
    last_served_by = ""
    now = time.time()
    all_cooling = all(_cooldown.get(i, 0) > now for i in range(len(_KEYS)))

    def _body_for(key_idx: int) -> tuple[str, dict]:
        # Once this turn has already burned _LAST_RESORT_AFTER keys, the pool is not going to
        # answer — the remaining keys share the same exhausted project-wide quota, or the model
        # itself is down (a 503 "high demand" killed a real turn in production). Switch models
        # rather than keep paying for the same refusal on key after key.
        model = _model_for_key_idx(key_idx)
        if _LAST_RESORT_MODEL and last_attempt_count >= _LAST_RESORT_AFTER:
            model = _LAST_RESORT_MODEL
        gen_config = {"temperature": 0.7, "maxOutputTokens": _max_tokens_for(model)}
        thinking = _thinking_config_for(model)
        if thinking:
            gen_config["thinkingConfig"] = thinking
        return model, {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
            "generationConfig": gen_config,
        }

    # STAGGERED hedge: launch on ONE key; only if it hasn't answered within
    # GEMINI_HEDGE_AFTER_MS launch a backup on a second key and take whichever answers first.
    # SIMULTANEOUS racing (GEMINI_HEDGE>=2 with a tiny HEDGE_AFTER_MS) was measured harmful
    # elsewhere in this project family — doubled burst rate 429-cascaded the free-tier pool and
    # made the median WORSE. The stagger cuts the tail at ~zero steady-state cost.
    if hedge and _FRESH_ORDER and _HEDGE_AFTER_MS < 60_000:
        picks = []
        cur = _fresh_idx
        for _ in range(len(_FRESH_ORDER)):
            if len(picks) >= max(2, _HEDGE):
                break
            cur = (cur + 1) % len(_FRESH_ORDER)
            k = _FRESH_ORDER[cur]
            if k in (p[0] for p in picks):
                continue
            if not all_cooling and _cooldown.get(k, 0) > time.time():
                continue
            picks.append((k,) + _body_for(k))
        _fresh_idx = cur
        if len(picks) >= 2:
            async def _race_one(key_idx: int, model: str, body: dict):
                resp = await client.post(_URL.format(model=model),
                                          params={"key": _KEYS[key_idx]}, json=body)
                return key_idx, model, resp

            primary = asyncio.ensure_future(_race_one(*picks[0]))
            tasks = [primary]
            try:
                done, _p = await asyncio.wait({primary}, timeout=_HEDGE_AFTER_MS / 1000)
                if not done:  # primary is slow — fire the backup and race them
                    tasks.append(asyncio.ensure_future(_race_one(*picks[1])))
                pending = set(tasks)
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        try:
                            key_idx, model, resp = t.result()
                        except Exception:
                            continue
                        last_attempt_count += 1
                        if resp.status_code < 400:
                            _cooldown.pop(key_idx, None)
                            tag = "~backup" if len(tasks) > 1 else ""
                            last_served_by = f"key{key_idx + 1}/{model}{tag}"
                            return resp.json()
                        last_err = f"Gemini {resp.status_code} (key {key_idx + 1}, {model}): {resp.text[:160]}"
                        if _should_rotate(resp.status_code, resp.text):
                            _cooldown[key_idx] = time.time() + (60 if resp.status_code == 429 else 15)
                # every racer failed → fall through to the sequential walk
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()

    # Try the FRESH tier first (round-robin among just those keys), then fall through to the
    # OTHER tier only once every fresh key has failed this turn. Each key uses ITS OWN model
    # (see _model_for_key_idx) regardless of tier. Pass 1 respects cooldowns; pass 2 (reached
    # only if pass 1 found NOTHING usable) ignores them — a cooldown cascade must degrade to
    # "try anyway", never to a failed turn.
    for respect_cooldowns in (True, False):
        for order, cur in ((_FRESH_ORDER, _fresh_idx), (_OTHER_ORDER, _other_idx)):
            if not order:
                continue
            if len(order) > 1:
                cur = (cur + 1) % len(order)
            for _ in range(len(order)):
                key_idx = order[cur]
                if respect_cooldowns and _cooldown.get(key_idx, 0) > time.time():
                    cur = (cur + 1) % len(order)
                    continue
                model = _model_for_key_idx(key_idx)
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
                last_attempt_count += 1
                resp = await client.post(url, params={"key": _KEYS[key_idx]}, json=body)
                if order is _FRESH_ORDER:
                    _fresh_idx = cur
                else:
                    _other_idx = cur
                if resp.status_code < 400:
                    _cooldown.pop(key_idx, None)
                    last_served_by = f"key{key_idx + 1}/{model}"
                    return resp.json()
                last_err = f"Gemini {resp.status_code} (key {key_idx + 1}/{len(_KEYS)}, {model}): {resp.text[:160]}"
                if _should_rotate(resp.status_code, resp.text):
                    _cooldown[key_idx] = time.time() + (60 if resp.status_code == 429 else 15)
                    cur = (cur + 1) % len(order)
                    continue
                raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("All Gemini keys exhausted — " + (last_err or "quota/invalid"))


# ── STREAMING PATH ────────────────────────────────────────────────────────────────────────
# The single biggest latency win in the whole pipeline. With :generateContent nothing exists
# until the WHOLE reply has been written (measured 1.3-1.9s here) and only then can TTS start.
# With :streamGenerateContent the FIRST CLAUSE is typically ready in 350-550ms and goes
# straight to the synthesiser while the rest is still being generated. STREAM_LLM=0 restores
# the old blocking path exactly.
_URL_STREAM = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:streamGenerateContent")
STREAM_LLM = _clean("STREAM_LLM", "1").lower() not in ("0", "false", "no", "off")

# No FIRST TOKEN within this long means the primary key is stalling — fire a backup and race.
# This supersedes the whole-response hedge timer on the streaming path: a stall is visible at
# the first token, so we no longer wait 3.5s of total silence to notice one. Same anti-burst
# property as the staggered hedge (the backup only ever fires on a real stall).
# MEASURED, not guessed. Healthy first-token on this pool is ~1.25-1.5s and it is FIXED overhead
# (a one-word reply costs the same as a full sentence). 1200ms was tried first and fired the
# backup on every single turn — doubling quota burn for nothing and feeding the very 429 pressure
# the hedge exists to cover. 2500ms was the opposite problem: a stalled key held the caller for a
# full 2.5s before anything else was tried. 1800 sits above the healthy band and still gets off a
# bad key quickly. 1800 was then MEASURED and reverted: escalating that eagerly burns noticeably
# more keys per minute, and on a rate-limited pool the extra 429s made p50 WORSE (english p50
# 5.8s vs 2-3s at 2500) — the same trap simultaneous hedging fell into here once before. Racing
# only looks free until the requests you add are the thing exhausting the pool.
_TTFT_STALL_MS = max(250, _int_env("GEMINI_TTFT_STALL_MS", 2500))
# Most streams a single turn may have IN FLIGHT at once. MEASURED constraint: a free-tier key
# serves ~7 requests per MINUTE (verified — the 8th returns 429 whether it carries 10 tokens or
# 4,500). Every extra racer therefore spends another key's minute-budget, which is the same
# reason simultaneous hedging was measured harmful on this pool once before. 2 = one backup.
_MAX_RACE = max(1, _int_env("GEMINI_MAX_RACE", 2))
# …and the hard cap on how many keys ONE turn may consume in total, so a failing turn can't walk
# the whole 104-key pool and spend everyone's minute-quota. A 429 comes back in ~200ms, so 20
# keys still fits inside the deadline below and gives a real answer a genuine chance. This was
# briefly set to 6, tuned against a test harness firing back-to-back turns — far too tight for a
# single live caller, who then got a 7s "sorry, the line broke" while 90+ healthy keys sat idle.
_MAX_KEYS_PER_TURN = max(2, _int_env("GEMINI_MAX_KEYS_PER_TURN", 20))
# How many requests may be IN FLIGHT while replacing keys that have already been REJECTED. This
# is the cascade brake, and it is deliberately separate from _MAX_RACE: _MAX_RACE governs
# speculation (firing an extra key at a stream that has gone quiet but may still answer), whereas
# this governs recovery (something came back 429/404 and that key is out of the running). Walking
# rejections one at a time cost a measured 6978ms on a single turn — 13 keys x ~500ms, strictly
# serial — while the caller listened to nothing. 3 keeps the recovery bounded without turning a
# cascade into a burst that makes the cascade worse.
#
# DEFAULT 1 — MEASURED, not assumed. At 3 the p50 went 2501ms -> 7483ms: a turn walked 15-16 keys
# instead of 13, every one 429, and it hit the 7s give-up deadline. Recovering FASTER from a
# rejection cascade also means spending the shared quota faster, which deepens the cascade for the
# NEXT turn — and on this pool the 429s are project-wide, not per-key, so more keys is not more
# quota. At 1 this is exactly the old serial behaviour, just handled inside the loop instead of
# falling out to the outer one. Raise it only against a pool with genuinely per-key quota.
_MAX_INFLIGHT_ERR = max(1, _int_env("GEMINI_MAX_INFLIGHT_ERR", 1))
# Hard ceiling on "no first token from ANY key". Without it, a turn where every raced stream
# stalls waits out the httpx read timeout once per rotation — measured 24s of pure silence on a
# quota-stressed pool, which is a dead call. At the ceiling we stop and let gemini_turn speak its
# graceful "say that again?" line instead: bad, but an order of magnitude less bad than silence.
# 7000 -> 4500. Production logs show every quota cascade running the deadline out in full, so this
# is not a theoretical ceiling — it is exactly how long the caller sits in silence before hearing
# "sorry, could you say that again". A healthy turn reaches first token in ~1.5s and a hedge adds
# ~2.5s, so 4.5s still covers the slowest legitimate path; past that the turn is not coming back
# and the only question is how long the caller is made to wait to be told so.
_TTFT_GIVEUP_MS = max(2000, _int_env("GEMINI_TTFT_GIVEUP_MS", 4500))
_DEBUG = _clean("GEMINI_DEBUG", "0").lower() not in ("0", "false", "no", "off", "")
# TRIED AND REMOVED (2026-07-25): a "last resort" pass that re-asked an emergency model
# (gemini-flash-lite-latest) on untouched keys once every key had failed on the real one. The
# idea was sound — lite was verified serving 200 on 12/12 keys while the primary 503'd — but
# MEASURED WORSE end to end: it appends up to three more sequential calls to the failure path,
# and turns that used to fail fast with a graceful line instead produced NO reply at all inside
# 40s. Failing fast and saying "sorry, the line broke" beats an unbounded rescue attempt.

_CLAUSE_HARD, _CLAUSE_SOFT = "।?!.…\n", ",;:—"
# _CLAUSE_SOFT_MIN 40 -> 75. At 40 a comma anywhere past the 40th character split the sentence,
# which on a one-sentence reply (Rule #2 caps them at one sentence) meant almost every reply was
# cut at its last comma — precisely where the trailing honorific lives. At 75 a soft split only
# happens in a genuinely long sentence, so the common case now streams whole and ElevenLabs gives
# it a single, properly-falling intonation contour instead of two half-phrases.
_CLAUSE_MIN, _CLAUSE_SOFT_MIN, _CLAUSE_MAX = 12, 75, 110


# RECORD TALK — the CRM row leaking into the caller's ear.
#
# The close-out prompt the client sends contains the record verbatim: "… CALL log_fap_enquiry
# with outcome='off_topic' and notes='no response on call'". The pre-existing guard only strips a
# parrot that KEEPS ITS PARENTHESES — `\(System[^)]*\)`. A model that paraphrases the instruction
# without them (which is exactly what "reading out the note" looks like) sailed straight through
# it and got spoken. Rule #3 now forbids this outright; this is the net under it, because a
# prompt rule is a request and the caller hearing "outcome off_topic, notes no response on call"
# is not a cosmetic defect.
#
# DELIBERATELY NARROW — it must never eat a legitimate line. "नोट कर लिया जी" and "I've noted it,
# sir" are the real canned confirmations in _fallback_for, so bare "note"/"noted" is NOT a
# pattern here; only unambiguous record machinery is.
_RECORD_TALK = re.compile(
    r"log_fap_enquiry|log_farmer_issue|log_rsvp|\bCRM\b"
    r"|\b(?:outcome|notes|product_interest|guest_note)\s*[:=]"
    # stems, not whole words: "saving" is sav+ing, "logging" is logg+ing — spelling the verbs out
    # as save|log only ("save"+"ing") missed the most natural phrasing of all, "Saving it to the
    # record now", which is exactly the sentence a model reaches for here.
    r"|\b(?:record|logg|log|enter|save|sav|add)(?:ing|ed|s)?\s+(?:this|it|these|that|the\s+\w+)?\s*"
    r"(?:in|into|to|onto)\s+(?:the\s+|our\s+|your\s+)?(?:system|crm|record|records|register|database|file)"
    r"|सिस्टम\s*में|रिकॉर्ड\s*में|दर्ज\s*कर\s*(?:रहा|रही|रहे)",
    re.IGNORECASE)

# Sentence split that respects Devanagari danda as well as Latin terminators.
_SENT_SPLIT = re.compile(r"(?<=[।?!.])\s+")


def _strip_record_talk(s: str) -> str:
    """Drop whole sentences that narrate the CRM record; keep everything else.

    Sentence-level rather than whole-reply, so one leaked clause never costs the caller the
    genuine answer sitting next to it."""
    if not s or not _RECORD_TALK.search(s):
        return s
    kept = [p for p in _SENT_SPLIT.split(s) if p.strip() and not _RECORD_TALK.search(p)]
    return " ".join(kept).strip()


_WORD = re.compile(r"[\wऀ-ॿ]{3,}", re.UNICODE)


def _echoes_record(spoken: str, args: dict | None) -> bool:
    """Is this 'reply' just the CRM note restated as a sentence?

    _RECORD_TALK catches the STRUCTURAL leak (field names, tool names). This catches the semantic
    one: no give-away keywords, the model simply says the summary it is about to file — "किसान ने
    डीएपी के बारे में पूछा, दो एकड़ ज़मीन" — which is the thing the caller should never hear.
    Deliberately requires a substantial note and heavy overlap: a real reply naturally shares a
    few words with the note (the product, the village), and losing a genuine answer would be far
    worse than letting one summary through."""
    note = " ".join(str((args or {}).get(k) or "") for k in ("notes", "description"))
    want = {w.lower() for w in _WORD.findall(note)}
    if len(want) < 5:
        return False
    have = {w.lower() for w in _WORD.findall(spoken or "")}
    return len(want & have) / len(want) >= 0.7


def _speakable(s: str) -> str:
    """The same sanitisation gemini_turn applies to the final text, applied per clause — so
    what gets spoken early is always a prefix of what the turn ultimately returns.

    That prefix invariant is why _strip_record_talk MUST also run on gemini_turn's final text:
    if streaming dropped a clause the final text kept, main.py's "what has the caller already
    heard" comparison would diverge and re-synthesise the whole reply — making the caller hear
    the leak TWICE instead of never."""
    s = re.sub(r"\(System[^)]*\)?", "", s or "")
    return _strip_record_talk(re.sub(r"\s+", " ", s).strip())


# A trailing honorific must NEVER become its own clause. Reported from a real call as "it takes a
# gap and says sir in a high pitch", and reproduced exactly: "…per bag, sir." split into
# ["…per bag,"] + ["sir."]. The first fragment is what gets flush:true in ElevenStream.feed, so
# ElevenLabs synthesises a COMMA-TERMINATED FRAGMENT as a finished utterance — which is why the
# sentence never falls at the end — and then renders "sir." as a separate utterance, which is the
# gap and the high pitch. prompts.py actively asks for 'sir'/'जी' mid-call, so this fired
# constantly rather than occasionally.
_VOCATIVE = re.compile(r"^[\s,]*(?:sir|madam|ji|जी|जनाब|अन्नदाता)\b[\s,;:—.!?]*$", re.IGNORECASE)
# A soft split also must not leave a stub behind it. 14 chars is comfortably longer than any
# honorific plus its punctuation, and short enough that genuine second clauses still stream early.
_CLAUSE_TAIL_MIN = 14


def _tail_ok(buf: str, i: int) -> bool:
    """May we split after index `i`? Only if what follows is a real clause, not an orphan.

    Two ways to be an orphan: too short to stand on its own, or a bare honorific. Both produce the
    same audible defect. Note this can only DELAY a split — the tail is flushed intact at
    end-of-stream — so the worst case is a fractionally later first chunk, never lost words."""
    tail = buf[i + 1:]
    return len(tail.strip()) >= _CLAUSE_TAIL_MIN and not _VOCATIVE.match(tail)


def _next_clause(buf: str) -> tuple[str, str]:
    """Peel one speakable clause off a growing buffer -> (clause, remainder)."""
    if len(buf) < _CLAUSE_MIN:
        return "", buf
    # Never emit a half-written "(System …)" leak: gemini_turn strips those from the final text,
    # so a partial one must not reach the speaker either.
    sys_at = buf.rfind("(System")
    if sys_at >= 0 and ")" not in buf[sys_at:]:
        return "", buf
    for i, ch in enumerate(buf):
        if i + 1 < _CLAUSE_MIN:
            continue
        if ch in _CLAUSE_HARD:
            # "2.5 लीटर" and "₹1,200" must not be mistaken for a sentence end.
            if ch == "." and (buf[i - 1:i].isdigit() or buf[i + 1:i + 2].isdigit()):
                continue
            return buf[:i + 1], buf[i + 1:]
        if ch in _CLAUSE_SOFT and i + 1 >= _CLAUSE_SOFT_MIN and _tail_ok(buf, i):
            return buf[:i + 1], buf[i + 1:]
    if len(buf) >= _CLAUSE_MAX:                    # no punctuation in sight — break on a word
        cut = buf.rfind(" ", _CLAUSE_MIN, _CLAUSE_MAX)
        if cut > 0 and _tail_ok(buf, cut):
            return buf[:cut + 1], buf[cut + 1:]
    return "", buf


def norm_spoken(s: str) -> str:
    """Public alias — main.py uses it to compare what was already streamed to the speaker
    against the turn's final text, so it can synthesise only the remainder."""
    return _speakable(s)


def _parts_of(data: dict) -> list:
    cands = data.get("candidates") or []
    return ((cands[0].get("content") or {}).get("parts") or []) if cands else []


async def _sse_pump(key_idx: int, model: str, body: dict, out: asyncio.Queue, tag: int) -> None:
    """Run ONE streaming request, pushing (tag, kind, status, payload) onto a shared queue.
    'done' is pushed only after the response is fully closed, so a finished pump is safe to
    cancel without leaving a half-read connection in the keep-alive pool."""
    finished = False
    try:
        async with _http.client().stream(
            "POST", _URL_STREAM.format(model=model),
            params={"key": _KEYS[key_idx], "alt": "sse"}, json=body,
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")
                await out.put((tag, "err", resp.status_code, detail[:200]))
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    await out.put((tag, "chunk", 0, json.loads(raw)))
                except json.JSONDecodeError:
                    continue
            finished = True
        if finished:
            await out.put((tag, "done", 0, None))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await out.put((tag, "err", 0, str(exc)[:200]))


async def _generate_stream(contents: list, scenario: str = "lead", lang: str = "",
                           force_tool: bool = False, on_clause=None) -> dict:
    """Streaming twin of _generate(): same inputs, and deliberately the SAME return shape.

    The only difference is that complete clauses are handed to `on_clause` the moment they are
    ready, instead of the caller waiting for the whole reply. Returning an identical
    {"candidates":[{"content":{"parts":[…]}}]} dict is the point — gemini_turn's tool dispatch,
    required-field gate, anonymous-caller blanking and sanitisation all keep working unchanged.
    """
    global _fresh_idx, last_attempt_count, last_served_by
    if not _KEYS:
        raise RuntimeError("No Gemini API key set")
    system_text = build_system_prompt(_today(), scenario, lang)
    tools = [{"functionDeclarations": tools_for(scenario)}]
    tool_config = {"functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}}
    last_attempt_count = 0
    last_served_by = ""
    last_err = None
    all_cooling = all(_cooldown.get(i, 0) > time.time() for i in range(len(_KEYS)))

    def _body_for(key_idx: int) -> tuple[str, dict]:
        # Once this turn has already burned _LAST_RESORT_AFTER keys, the pool is not going to
        # answer — the remaining keys share the same exhausted project-wide quota, or the model
        # itself is down (a 503 "high demand" killed a real turn in production). Switch models
        # rather than keep paying for the same refusal on key after key.
        model = _model_for_key_idx(key_idx)
        if _LAST_RESORT_MODEL and last_attempt_count >= _LAST_RESORT_AFTER:
            model = _LAST_RESORT_MODEL
        gen_config = {"temperature": 0.7, "maxOutputTokens": _max_tokens_for(model)}
        thinking = _thinking_config_for(model)
        if thinking:
            gen_config["thinkingConfig"] = thinking
        return model, {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "tools": tools,
            "toolConfig": tool_config,
            "generationConfig": gen_config,
        }

    # Fresh tier first, reserve tier second, cooling keys last — a cooldown cascade must degrade
    # to "try anyway", never to a failed turn (the same rule _generate's two-pass walk enforces).
    hot, cold = [], []
    for tier in (_FRESH_ORDER, _OTHER_ORDER):
        for k in tier:
            (cold if (not all_cooling and _cooldown.get(k, 0) > time.time()) else hot).append(k)
    if hot:                                   # rotate so traffic spreads across the fresh tier
        _fresh_idx = (_fresh_idx + 1) % len(hot)
        hot = hot[_fresh_idx:] + hot[:_fresh_idx]
    order = (hot + cold) or list(range(len(_KEYS)))

    turn_deadline = time.monotonic() + _TTFT_GIVEUP_MS / 1000
    for attempt, primary in enumerate(order):
        if time.monotonic() >= turn_deadline or last_attempt_count >= _MAX_KEYS_PER_TURN:
            break
        q: asyncio.Queue = asyncio.Queue()
        tasks: dict[int, asyncio.Task] = {}
        meta: dict[int, tuple[int, str]] = {}
        winner = None

        def _launch(key_idx: int, tag: int) -> None:
            model, body = _body_for(key_idx)
            meta[tag] = (key_idx, model)
            tasks[tag] = asyncio.ensure_future(_sse_pump(key_idx, model, body, q, tag))

        try:
            _launch(primary, 0)
            last_attempt_count += 1
            # `fired` = keys launched in this block (also the index into `order` for the next one).
            # `raced` counts only SPECULATIVE launches — the stall escalations — and is what
            # _MAX_RACE bounds. They were the same counter until error-replacement was added
            # below, which would otherwise have spent the speculation budget on rejections and
            # left a genuine stall with no backup.
            first_chunk, fired, raced = None, 1, 1
            next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
            while tasks and winner is None:
                # ESCALATING stagger. Firing exactly one backup wasn't enough: measured stalls of
                # 6s and 11s happened with the backup already in flight and also stalling. Each
                # further _TTFT_STALL_MS of total silence adds one more key, up to _MAX_RACE.
                # Still nothing like simultaneous racing — on a healthy turn (~1.4s TTFT) not one
                # extra request is ever sent, so the burst rate that 429s this pool is unchanged.
                more = order[attempt + fired] if attempt + fired < len(order) else None
                budget = turn_deadline - time.monotonic()
                if budget <= 0:
                    break
                can_escalate = (more is not None and raced < _MAX_RACE
                                and last_attempt_count < _MAX_KEYS_PER_TURN)
                wait_s = max(0.01, min(next_deadline - time.monotonic(), budget)
                             if can_escalate else budget)
                try:
                    tag, kind, status, payload = await asyncio.wait_for(q.get(), timeout=wait_s)
                except asyncio.TimeoutError:
                    if not can_escalate or time.monotonic() >= turn_deadline:
                        if _DEBUG:
                            print(f"  [gem] gave up: tried keys "
                                  f"{[m[0] + 1 for m in meta.values()]}", flush=True)
                        break                       # out of budget — stop waiting on this turn
                    if _DEBUG:
                        print(f"  [gem] key{meta[fired - 1][0] + 1} silent >"
                              f"{_TTFT_STALL_MS}ms — escalating to key{more + 1}", flush=True)
                    _launch(more, fired)
                    fired += 1
                    raced += 1
                    last_attempt_count += 1
                    next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
                    continue
                if kind == "chunk":
                    winner, first_chunk = tag, payload
                    break
                k, m = meta[tag]
                if kind == "err":
                    last_err = f"Gemini {status or ''} (key {k + 1}, {m}): {payload}"
                    if _DEBUG:
                        print(f"  [gem] key{k + 1} {m} ERR {status} {str(payload)[:110]}",
                              flush=True)
                    if _should_rotate(status, payload or ""):
                        _cooldown[k] = time.time() + (60 if status == 429 else 15)
                elif _DEBUG:
                    print(f"  [gem] key{k + 1} {m} stream ended with no content", flush=True)
                tasks.pop(tag, None)
                # REPLACE A FAILED KEY IMMEDIATELY, IN PLACE. Letting `tasks` drain to empty fell
                # out to the outer loop, which relaunched exactly one key — so a 429 cascade ran
                # STRICTLY SERIALLY at ~500ms per rejection. Measured: one turn spent 6978ms
                # walking 13 keys that way, and the tail, not the median, is what a caller
                # experiences as "it hangs sometimes". Refilling here keeps _MAX_INFLIGHT_ERR
                # requests moving at once during a cascade only. This is NOT speculative racing:
                # it fires solely in response to a rejection already received, so a healthy turn
                # (one key, one request) sends exactly as much traffic as before and the burst
                # rate that 429s this pool is untouched.
                if (len(tasks) < _MAX_INFLIGHT_ERR
                        and last_attempt_count < _MAX_KEYS_PER_TURN
                        and time.monotonic() < turn_deadline):
                    nxt = order[attempt + fired] if attempt + fired < len(order) else None
                    if nxt is not None:
                        _launch(nxt, fired)
                        fired += 1
                        last_attempt_count += 1
                        next_deadline = time.monotonic() + _TTFT_STALL_MS / 1000
            if winner is None:
                continue                        # nothing came back on this key — try the next
            for tag, t in tasks.items():
                if tag != winner and not t.done():
                    t.cancel()
            key_idx, model = meta[winner]
            _cooldown.pop(key_idx, None)
            last_served_by = f"key{key_idx + 1}/{model}" + ("~backup" if winner else "")

            full_text, held, extra_parts, saw_tool = "", "", [], False
            chunk = first_chunk
            while True:
                for part in _parts_of(chunk):
                    piece = part.get("text")
                    if isinstance(piece, str):
                        full_text += piece
                        held += piece
                    elif part:
                        extra_parts.append(part)   # functionCall (+ thoughtSignature) kept whole
                        saw_tool = True
                # Stop speaking the moment a tool call appears: what is said after one is decided
                # by gemini_turn (its own text, a fallback line, a re-ask, or a second turn) —
                # never by this stream.
                if on_clause and not saw_tool:
                    while True:
                        clause, held = _next_clause(held)
                        if not clause:
                            break
                        say = _speakable(clause)
                        if say:
                            await on_clause(say)
                nxt = None
                while True:                        # next event from the winning stream only
                    tag, kind, status, payload = await q.get()
                    if tag != winner:
                        continue                   # a cancelled loser's straggler — ignore
                    if kind == "chunk":
                        nxt = payload
                    elif kind == "err":
                        last_err = f"Gemini stream cut (key {key_idx + 1}): {payload}"
                    break
                if nxt is None:
                    break
                chunk = nxt
            if on_clause and not saw_tool:
                say = _speakable(held)             # flush the tail
                if say:
                    await on_clause(say)

            parts = ([{"text": full_text}] if full_text else []) + extra_parts
            return {"candidates": [{"content": {"parts": parts or [{"text": ""}]}}]}
        finally:
            for t in tasks.values():
                if not t.done():
                    t.cancel()
            # A key that never produced a first token is stalling, even though it never returned
            # a status we'd rotate on. Cool it off too, or the next turn picks the same throttled
            # keys straight away — measured: four consecutive turns all timing out on keys 81-85
            # while the other 99 keys in the pool sat idle and healthy.
            for tag, (k, _m) in meta.items():
                if tag != winner:
                    _cooldown[k] = max(_cooldown.get(k, 0), time.time() + 20)

    raise RuntimeError("All Gemini keys exhausted (stream) — " + (last_err or "quota/invalid"))


async def gemini_turn(contents: list, user_text: str, handlers: dict, scenario: str = "sales",
                      lang: str = "", on_clause=None) -> str:
    """Run one customer turn.

    handlers: {tool_name: async fn(args)->result_dict}. Returns the agent's reply text.
    `scenario` (sales/gosthi) selects the persona and tool set; `lang` (english/hindi) selects
    the spoken language — empty falls back to the scenario's showcase default.
    `on_clause`: optional async fn(str) called with each speakable clause AS IT IS GENERATED.
    Passing it switches this turn to :streamGenerateContent; leaving it None keeps the original
    blocking path byte-for-byte. What it receives is always a prefix of the returned text (see
    _speakable), so the caller can synthesise the remainder and never repeat itself.
    """
    lang = norm_lang(lang, scenario)
    contents.append({"role": "user", "parts": [{"text": user_text}]})
    last_tool, last_args = None, None
    # The client's close note names the tool it needs ("… CALL log_fap_enquiry …" / "… CALL
    # log_rsvp …") — force function-calling on those turns so the outcome is ALWAYS recorded.
    force_tool = "(System note" in (user_text or "") and "CALL " in (user_text or "")

    for turn_i in range(5):  # allow a couple of tool round-trips
        try:
            if on_clause is not None and STREAM_LLM:
                # Streaming: clauses reach the synthesiser as they are written. EVERY turn
                # streams, the find_nearest_fap follow-up included — that second call is a full
                # Gemini round-trip the caller would otherwise sit through in silence.
                data = await _generate_stream(contents, scenario, lang,
                                              force_tool=force_tool and last_tool is None,
                                              on_clause=on_clause)
            else:
                # Hedge only the first call of the turn — tool-followup calls are rare and the
                # racing setup isn't worth doubling their quota cost.
                data = await _generate(contents, scenario, lang,
                                       force_tool=force_tool and last_tool is None,
                                       hedge=turn_i == 0)
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
            if name in ("log_fap_enquiry", "log_farmer_issue"):  # inbound, ANONYMOUS caller — never invent a
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
            own = _strip_record_talk(own)                 # never read the row back to the caller
            # A same-turn line that merely RESTATES what is being filed is not a reply — it is
            # the record wearing a sentence. Fall back to the proper confirmation instead.
            if own and _echoes_record(own, args):
                own = ""
            spoken = own if len(own) >= 8 else _fallback_for(name, args, lang)
            contents.append({"role": "model", "parts": [{"text": spoken}]})
            return spoken

        final = "".join(text_chunks).strip()
        # flash-lite sometimes parrots internal "(System note …)" instructions into its reply —
        # strip them so they are never shown or spoken to the customer.
        final = re.sub(r"\(System[^)]*\)", "", final)
        final = re.sub(r"\s*\n+\s*", " ", final).strip()  # one spoken line, never split
        final = _strip_record_talk(final)                 # keep the prefix invariant _speakable relies on
        if not final:
            final = (
                _fallback_for(last_tool, last_args, lang)
                if last_tool
                else _reask(lang)
            )
        return final

    return _reask(lang)
