"""services/catalog.py — UA Agro company facts + product catalog + crop guide, embedded
directly into the `sales` scenario's prompt as static reference text (not a tool call — this is
exactly the CLINIC["services"] precedent from the sibling repos, just a bigger category set).

Pricing: urea and DAP use the real, long-stable government-notified MRPs (spot-check/date-stamp
these before a live pitch — subsidy rates do get revised). Everything else — branded seeds,
crop-protection products, cattle feed, equipment — is a constructed, realistic market estimate,
the same "illustrative demo data" precedent every sibling repo's case data already uses.
"""
from __future__ import annotations

COMPANY = {
    "legal_name": "UA Agro Solutions Private Limited",
    "retail_brand": "Naveen Khushhali Kisan Sewa Kendra",
    "retail_brand_hi": "नवीन खुशहाली किसान सेवा केंद्र",
    "short_name": "Kisan Sewa Kendra",
    "short_name_hi": "किसान सेवा केंद्र",
    "helpline": "18002127074",
    "region": "Central & Eastern Uttar Pradesh",
    "store_count": "80+",
    "district_count": "15+",
}

PRODUCT_CATALOG = {
    "Fertilizers & Nutrients": [
        {"name": "Urea (46% N)", "price": "₹266.50 for a 45kg bag",
         "crops": ["wheat", "rice", "paddy", "sugarcane", "potato"],
         "note": "government-fixed MRP"},
        {"name": "DAP — Di-Ammonium Phosphate (18-46-0)", "price": "₹1,350 for a 50kg bag",
         "crops": ["wheat", "mustard", "potato"], "note": "government-fixed MRP"},
        {"name": "MOP — Muriate of Potash (60% K2O)", "price": "around ₹1,700 for a 50kg bag",
         "crops": ["sugarcane", "potato"]},
        {"name": "NPK Complex (12:32:16)", "price": "around ₹1,470 for a 50kg bag",
         "crops": ["wheat", "rice", "paddy"]},
        {"name": "Zinc Sulphate", "price": "around ₹120 per kg",
         "crops": ["rice", "paddy", "wheat"], "note": "for zinc-deficient soil, common in UP"},
    ],
    "Seeds": [
        {"name": "Certified wheat seed (HD-2967)", "price": "₹35 to ₹40 per kg", "crops": ["wheat"]},
        {"name": "Hybrid paddy seed", "price": "₹250 to ₹300 per kg", "crops": ["rice", "paddy"]},
        {"name": "Certified paddy seed (Sarju-52)", "price": "₹60 to ₹70 per kg", "crops": ["rice", "paddy"]},
        {"name": "Mustard seed (Pusa Bold)", "price": "₹110 to ₹130 per kg", "crops": ["mustard"]},
        {"name": "Potato seed tuber", "price": "₹28 to ₹35 per kg", "crops": ["potato"]},
        {"name": "Vegetable seed packets (tomato, okra, cauliflower)", "price": "₹150 to ₹400 per packet",
         "crops": ["vegetables"]},
    ],
    "Crop Protection": [
        {"name": "Tricyclazole fungicide", "price": "₹800 to ₹900 per kg",
         "crops": ["rice", "paddy"], "note": "for paddy blast disease"},
        {"name": "Imidacloprid insecticide", "price": "₹450 to ₹600 per litre",
         "crops": ["rice", "paddy", "sugarcane"]},
        {"name": "Pretilachlor herbicide", "price": "₹350 to ₹450 per litre",
         "crops": ["rice", "paddy"], "note": "for weed control"},
        {"name": "Neem-based bio-pesticide", "price": "₹300 to ₹350 per litre",
         "crops": ["rice", "paddy", "wheat", "vegetables"], "note": "organic-leaning farmers"},
    ],
    "Cattle Feed": [
        {"name": "Balanced cattle feed pellets", "price": "₹1,100 to ₹1,250 for a 40kg bag"},
        {"name": "Mineral mixture supplement", "price": "around ₹450 for a 25kg bag"},
    ],
    "Farm Equipment": [
        {"name": "Power weeder (small, paddy-field)", "price": "₹28,000 to ₹35,000"},
        {"name": "Knapsack sprayer (16L manual)", "price": "₹1,200 to ₹1,500"},
        {"name": "Battery-operated sprayer (16L)", "price": "₹3,500 to ₹4,200"},
        {"name": "Hand tool sets (sickle etc.)", "price": "₹150 to ₹300 each"},
    ],
}

# UP's major crops → the products above that actually apply — the "what should I use for my
# crop" answer, keyed loosely (lowercase, a couple of common spellings/synonyms per crop).
CROP_GUIDE = {
    "wheat": ["Urea (46% N)", "DAP — Di-Ammonium Phosphate (18-46-0)", "Zinc Sulphate",
              "Certified wheat seed (HD-2967)", "Neem-based bio-pesticide"],
    "rice": ["Urea (46% N)", "NPK Complex (12:32:16)", "Zinc Sulphate", "Hybrid paddy seed",
             "Certified paddy seed (Sarju-52)", "Tricyclazole fungicide", "Imidacloprid insecticide",
             "Pretilachlor herbicide"],
    "paddy": ["Urea (46% N)", "NPK Complex (12:32:16)", "Zinc Sulphate", "Hybrid paddy seed",
              "Certified paddy seed (Sarju-52)", "Tricyclazole fungicide", "Imidacloprid insecticide",
              "Pretilachlor herbicide"],
    "sugarcane": ["Urea (46% N)", "MOP — Muriate of Potash (60% K2O)", "Imidacloprid insecticide"],
    "potato": ["DAP — Di-Ammonium Phosphate (18-46-0)", "MOP — Muriate of Potash (60% K2O)",
               "Potato seed tuber"],
    "mustard": ["DAP — Di-Ammonium Phosphate (18-46-0)", "Mustard seed (Pusa Bold)"],
    "pulses": ["Neem-based bio-pesticide"],
    "vegetables": ["Vegetable seed packets (tomato, okra, cauliflower)", "Neem-based bio-pesticide"],
}


def format_catalog_for_prompt() -> str:
    lines = []
    for category, items in PRODUCT_CATALOG.items():
        lines.append(f"{category}:")
        for it in items:
            crops = f" — for {', '.join(it['crops'])}" if it.get("crops") else ""
            note = f" ({it['note']})" if it.get("note") else ""
            lines.append(f"  - {it['name']}: {it['price']}{crops}{note}")
    return "\n".join(lines)


def format_crop_guide_for_prompt() -> str:
    lines = []
    for crop, products in CROP_GUIDE.items():
        lines.append(f"  - {crop}: {', '.join(products)}")
    return "\n".join(lines)
