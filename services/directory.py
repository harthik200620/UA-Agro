"""services/directory.py — UA Agro's Farmer Access Point (FAP) store directory + a deterministic
nearest-match lookup for the `sales` scenario.

No exhaustive real public FAP address list exists for UA Agro (checked this session — the
company publishes only a store COUNT and district COUNT, not a full directory). This is a
representative sample built across the real, verified footprint (80+ stores across 15+
districts of Central & Eastern Uttar Pradesh) using real town/tehsil names within those real
districts. Deliberately NO per-FAP phone numbers — inventing 19 fake direct-dial numbers for
real towns is a credibility risk; any caller asking for a number should be pointed to the one
real, verified toll-free helpline instead (see COMPANY in services/catalog.py).
"""
from __future__ import annotations

import difflib

FAP_DIRECTORY = [
    {"id": "KNP-01", "name": "Kisan Sewa Kendra – Bilhaur", "district": "Kanpur Nagar",
     "town": "Bilhaur", "landmark": "near the tehsil chauraha, Bilhaur–Kanpur road",
     "aliases": ["bilhaur", "बिलहौर", "बिल्हौर"]},
    {"id": "KNP-02", "name": "Kisan Sewa Kendra – Ghatampur", "district": "Kanpur Nagar",
     "town": "Ghatampur", "landmark": "opposite the mandi samiti gate",
     "aliases": ["ghatampur", "घाटमपुर"]},
    {"id": "KND-01", "name": "Kisan Sewa Kendra – Akbarpur", "district": "Kanpur Dehat",
     "town": "Akbarpur", "landmark": "on the Kanpur–Kalyanpur road, near the block office",
     "aliases": ["akbarpur", "अकबरपुर"]},
    {"id": "LKO-01", "name": "Kisan Sewa Kendra – Malihabad", "district": "Lucknow",
     "town": "Malihabad", "landmark": "near the mango mandi, Malihabad crossing",
     "aliases": ["malihabad", "मलिहाबाद"]},
    {"id": "LKO-02", "name": "Kisan Sewa Kendra – Mohanlalganj", "district": "Lucknow",
     "town": "Mohanlalganj", "landmark": "on the Lucknow–Raebareli road",
     "aliases": ["mohanlalganj", "मोहनलालगंज"]},
    {"id": "UNO-01", "name": "Kisan Sewa Kendra – Purwa", "district": "Unnao",
     "town": "Purwa", "landmark": "near the tehsil office, Purwa",
     "aliases": ["purwa", "पुरवा"]},
    {"id": "HRD-01", "name": "Kisan Sewa Kendra – Sandila", "district": "Hardoi",
     "town": "Sandila", "landmark": "on the Lucknow–Hardoi highway",
     "aliases": ["sandila", "sandeela", "सांडीला", "सांडी"]},
    {"id": "STP-01", "name": "Kisan Sewa Kendra – Biswan", "district": "Sitapur",
     "town": "Biswan", "landmark": "near the Biswan bus stand",
     "aliases": ["biswan", "बिसवां", "बीसवां"]},
    {"id": "BBK-01", "name": "Kisan Sewa Kendra – Zaidpur", "district": "Barabanki",
     "town": "Zaidpur", "landmark": "opposite the Zaidpur grain market",
     "aliases": ["zaidpur", "जैदपुर"]},
    {"id": "BBK-02", "name": "Kisan Sewa Kendra – Barabanki", "district": "Barabanki",
     "town": "Barabanki", "landmark": "near the district collectorate road",
     "aliases": ["barabanki town", "barabanki city", "बाराबंकी"]},
    {"id": "AYO-01", "name": "Kisan Sewa Kendra – Sohawal", "district": "Ayodhya",
     "town": "Sohawal", "landmark": "on the Ayodhya–Sohawal road",
     "aliases": ["sohawal", "faizabad sohawal", "सोहावल"]},
    {"id": "AYO-02", "name": "Kisan Sewa Kendra – Ayodhya", "district": "Ayodhya",
     "town": "Ayodhya", "landmark": "near the Ayodhya bypass, Rae Bareli road side",
     "aliases": ["ayodhya town", "faizabad", "अयोध्या", "फैज़ाबाद", "फैजाबाद"]},
    {"id": "RBL-01", "name": "Kisan Sewa Kendra – Unchahar", "district": "Raebareli",
     "town": "Unchahar", "landmark": "near the Unchahar railway station road",
     "aliases": ["unchahar", "ऊंचाहार", "उंचाहार"]},
    {"id": "PTG-01", "name": "Kisan Sewa Kendra – Kunda", "district": "Pratapgarh",
     "town": "Kunda", "landmark": "near the Kunda tehsil gate",
     "aliases": ["kunda", "कुंडा"]},
    {"id": "FTP-01", "name": "Kisan Sewa Kendra – Bindki", "district": "Fatehpur",
     "town": "Bindki", "landmark": "on the Fatehpur–Bindki road",
     "aliases": ["bindki", "बिंदकी", "बिन्दकी"]},
    {"id": "AUR-01", "name": "Kisan Sewa Kendra – Bidhuna", "district": "Auraiya",
     "town": "Bidhuna", "landmark": "near the Bidhuna block office",
     "aliases": ["bidhuna", "बिधुना"]},
    {"id": "KNJ-01", "name": "Kisan Sewa Kendra – Chhibramau", "district": "Kannauj",
     "town": "Chhibramau", "landmark": "near the Chhibramau grain mandi",
     "aliases": ["chhibramau", "chibramau", "छिबरामऊ"]},
    {"id": "FBD-01", "name": "Kisan Sewa Kendra – Kaimganj", "district": "Farrukhabad",
     "town": "Kaimganj", "landmark": "on the Farrukhabad–Kaimganj road",
     "aliases": ["kaimganj", "कायमगंज"]},
    {"id": "ETW-01", "name": "Kisan Sewa Kendra – Jaswantnagar", "district": "Etawah",
     "town": "Jaswantnagar", "landmark": "near the Jaswantnagar bus stand",
     "aliases": ["jaswantnagar", "जसवंतनगर"]},
]

# Devanagari forms of each served district — Hindi-language calls (this scenario's default)
# can produce a find_nearest_fap(place=...) call in EITHER script depending on how the model
# renders the conversation, so district lookup must work in both, not just Roman script.
_DISTRICT_HI = {
    "Kanpur Nagar": ["कानपुर नगर", "कानपुर"],
    "Kanpur Dehat": ["कानपुर देहात"],
    "Lucknow": ["लखनऊ"],
    "Unnao": ["उन्नाव"],
    "Hardoi": ["हरदोई"],
    "Sitapur": ["सीतापुर"],
    "Barabanki": ["बाराबंकी"],
    "Ayodhya": ["अयोध्या", "फैज़ाबाद", "फैजाबाद"],
    "Raebareli": ["रायबरेली"],
    "Pratapgarh": ["प्रतापगढ़", "प्रतापगढ"],
    "Fatehpur": ["फ़तेहपुर", "फतेहपुर"],
    "Auraiya": ["औरैया"],
    "Kannauj": ["कन्नौज"],
    "Farrukhabad": ["फ़र्रुख़ाबाद", "फर्रुखाबाद"],
    "Etawah": ["इटावा"],
}

# A few well-known UP districts we DON'T serve, mapped to the nearest district we DO — used only
# as a fallback hint when a caller names a place outside our footprint entirely. Both scripts.
NEARBY_DISTRICT_HINT = {
    "lakhimpur kheri": "Sitapur", "shahjahanpur": "Hardoi", "mainpuri": "Etawah",
    "kaushambi": "Fatehpur", "amethi": "Raebareli", "sultanpur": "Ayodhya",
    "ambedkar nagar": "Ayodhya", "kanpur": "Kanpur Nagar", "faizabad": "Ayodhya",
    "लखीमपुर खीरी": "Sitapur", "शाहजहाँपुर": "Hardoi", "शाहजहांपुर": "Hardoi",
    "मैनपुरी": "Etawah", "कौशाम्बी": "Fatehpur", "कौशांबी": "Fatehpur",
    "अमेठी": "Raebareli", "सुल्तानपुर": "Ayodhya", "अम्बेडकर नगर": "Ayodhya",
    "अंबेडकर नगर": "Ayodhya",
}


def _normalize(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().lower() if ch.isalnum() or ch.isspace()).strip()


_TOWN_INDEX: dict[str, dict] = {}
_DISTRICT_INDEX: dict[str, list[dict]] = {}
for _fap in FAP_DIRECTORY:
    _DISTRICT_INDEX.setdefault(_normalize(_fap["district"]), []).append(_fap)
    _TOWN_INDEX[_normalize(_fap["town"])] = _fap
    for _alias in _fap.get("aliases", []):
        _TOWN_INDEX[_normalize(_alias)] = _fap
# Devanagari district names point at the SAME fap list as their English key.
for _dist_en, _dist_hi_forms in _DISTRICT_HI.items():
    _faps_for_dist = _DISTRICT_INDEX.get(_normalize(_dist_en), [])
    for _form in _dist_hi_forms:
        _DISTRICT_INDEX[_normalize(_form)] = _faps_for_dist

_SERVED_DISTRICTS = sorted({fap["district"] for fap in FAP_DIRECTORY})


def _found(fap: dict, note: str) -> dict:
    return {"status": "found", "id": fap["id"], "name": fap["name"], "district": fap["district"],
            "town": fap["town"], "landmark": fap["landmark"], "distance_note": note}


def find_nearest_fap(place: str) -> dict:
    """Deterministic lookup — never guessed by the model. Tries, in order: exact town/alias
    match; STT-garbled-name fuzzy match (difflib, stdlib); exact/fuzzy district-name match;
    finally a 'not found' response listing what we DO serve, so the model can reason over real
    data (including its own general geography knowledge) instead of inventing a location."""
    q = _normalize(place)
    if not q:
        return {"status": "not_found", "served_districts": _SERVED_DISTRICTS}

    hit = _TOWN_INDEX.get(q)
    if hit:
        return _found(hit, "in " + hit["town"])

    close_towns = difflib.get_close_matches(q, _TOWN_INDEX.keys(), n=3, cutoff=0.72)
    if close_towns:
        matched = [_TOWN_INDEX[c] for c in close_towns]
        districts = {m["district"] for m in matched}
        if len(districts) > 1:
            return {"status": "ambiguous",
                    "candidates": [{"district": m["district"], "town": m["town"]} for m in matched[:3]]}
        return _found(matched[0], "near " + place.strip())

    dist_hit = _DISTRICT_INDEX.get(q)
    if dist_hit:
        return _found(dist_hit[0], "in " + dist_hit[0]["district"] + " district")

    close_districts = difflib.get_close_matches(q, _DISTRICT_INDEX.keys(), n=1, cutoff=0.72)
    if close_districts:
        fap = _DISTRICT_INDEX[close_districts[0]][0]
        return _found(fap, "in " + fap["district"] + " district")

    # Caller said something like "Bilhaur, Kanpur" or "Bilhaur kasbah" — no single index key
    # matches the WHOLE phrase, but one of its words alone might (try town index, then district).
    words = [w for w in q.split() if len(w) > 2]
    for w in words:
        hit = _TOWN_INDEX.get(w)
        if hit:
            return _found(hit, "near " + place.strip())
    for w in words:
        dist_hit = _DISTRICT_INDEX.get(w)
        if dist_hit:
            return _found(dist_hit[0], "in " + dist_hit[0]["district"] + " district")

    hint_district = NEARBY_DISTRICT_HINT.get(q)
    if not hint_district:
        close_hint = difflib.get_close_matches(q, NEARBY_DISTRICT_HINT.keys(), n=1, cutoff=0.75)
        hint_district = NEARBY_DISTRICT_HINT.get(close_hint[0]) if close_hint else None
    if hint_district:
        fap = _DISTRICT_INDEX[_normalize(hint_district)][0]
        return _found(fap, "our nearest FAP to that area is in " + fap["district"] + " district")

    return {"status": "not_found", "served_districts": _SERVED_DISTRICTS}
