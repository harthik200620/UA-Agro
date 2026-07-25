"""One shared, connection-pooled httpx.AsyncClient reused across turns.

Opening a fresh AsyncClient per call (the old pattern) paid a new TLS handshake to
Gemini / ElevenLabs / Sarvam on every request. Reusing one keep-alive client shaves
~100-300 ms off each call after the first. Created lazily inside the event loop and never
explicitly closed — the process owns it, and on Vercel it's reused across warm invocations.
"""
from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            # 40s read was pointless: the platform kills the function around 10-15s, so a stall
            # could never recover — it just 504'd the caller. 12s lets a stalled call drop into
            # the key rotation / hedge while there is still budget left to answer with.
            timeout=httpx.Timeout(12.0, connect=3.0),
            limits=httpx.Limits(max_keepalive_connections=16, keepalive_expiry=90.0),
        )
    return _client
