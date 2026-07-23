# Verba AI — Voice Agents for UA Agro (Kisan Sewa Kendra)

One app, two agents in two languages, plus a live CRM — built for **UA Agro Solutions Pvt
Ltd** (retail brand **Naveen Khushhali Kisan Sewa Kendra**, 80+ Farmer Access Points across
15+ districts of Central & Eastern Uttar Pradesh).

| Route | What it shows |
|---|---|
| `/` | Scenario picker → **🌾 Store enquiry line** — Meera answers inbound calls about products, prices, which product suits which crop, and looks up the caller's nearest store by village/town (real function-calling lookup, not a guess) · **📢 Kisan Gosthi outreach** — Kavita calls a farmer, checks in warmly, and invites them to the upcoming village meeting, capturing the RSVP. **Both agents speak English and Hindi** |
| `/crm` | **Verba CRM** — every call's outcome writes back here live (store enquiries incl. bulk/complaints, Gosthi RSVPs) |

**Stack:** Sarvam Saaras v3 (speech-to-text) · Google Gemini (brain + tools, 80-key hybrid
rotation) · ElevenLabs (voice, auto-falls back to Sarvam Bulbul) · FastAPI + SQLite. No
telephony, no ffmpeg, no Node — the browser captures 16 kHz WAV itself.

Product/pricing data (`services/catalog.py`) and the FAP store directory
(`services/directory.py`) are a representative demo dataset built from UA Agro's verified,
public footprint — not a scrape of a real internal store/price list (none is public). Urea
and DAP prices use the real government-notified MRPs; everything else is a realistic
market-rate estimate. Swap in real data before a production rollout.

## Setup
```powershell
cd "C:\Users\HP\Claude\Projects\AI service clients\uaagro-voice-agent"
& "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt

copy .env.example .env      # then open .env and paste your keys
```
Fill `.env`:
- `GEMINI_API_KEY` (+ optional `_2`…`_100` for rotation) — required (aistudio.google.com/apikey)
- `SARVAM_API_KEY` — speech-to-text. Without it the page uses the browser's built-in
  recognition (Chrome/Edge only).
- `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` (+ `_HI`) — the voice (English/Hindi).
- `ADMIN_PASSWORD` — kept for compatibility; the page currently has no access gate.

## Run
```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe" -m uvicorn main:app --reload --port 8000
```
Open **http://localhost:8000** in Chrome/Edge (use `localhost`, not a file:// path — the mic
needs a secure context, which localhost satisfies). Pick a scenario — the **agent speaks
first** (it "picks up").

You can also **type** in the box at the bottom to run without a microphone.

## Architecture notes
- `services/directory.py` — `find_nearest_fap(place)`: exact/alias town match → fuzzy
  (STT-garble-tolerant) match → district match → a small nearby-district fallback table →
  `not_found`. Deterministic, called by the model as a real tool — never guessed.
- `services/catalog.py` — the product catalog and crop guide are embedded directly into the
  sales prompt as reference text (static data, not a tool call).
- `services/llm.py` — `find_nearest_fap` is the one *read* tool in this build; unlike the
  write tools (`log_fap_enquiry`, `log_rsvp`), it gets a genuine second Gemini turn with the
  real lookup result (`_QUERY_TOOLS`) instead of the same-turn fast path, since the model
  can't know the answer before the lookup runs.

## Deploy (Vercel)
```powershell
npx vercel --prod
```
New Vercel projects default Deployment Protection to ON — disable it or the public link
302s to a login page:
```powershell
npx vercel project protection disable uaagro-voice-agent --sso
```
