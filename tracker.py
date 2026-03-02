import os
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests

# =========================
# CONFIG (editable via env)
# =========================
TRACK_QUERY = os.getenv(
    "TRACK_QUERY",
    "Iran Middle East escalation Red Sea shipping Strait of Hormuz sanctions strikes drones missiles diplomacy"
)
EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "12"))
FETCH_TIMEOUT_SEC = int(os.getenv("FETCH_TIMEOUT", "20"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "20000"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # safer default
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"

# =========================
# SECRETS (GitHub Secrets)
# =========================
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def validate_lon_lat(lon: Any, lat: Any) -> bool:
    try:
        lon = float(lon)
        lat = float(lat)
        return -180 <= lon <= 180 and -90 <= lat <= 90
    except Exception:
        return False


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# Exa Search
# -------------------------
def exa_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """
    Minimal Exa search call.
    If Exa changes endpoint/shape, adjust this.
    """
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {
        "query": query,
        "numResults": num_results,
        "type": "neural",
        "useAutoprompt": True,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=FETCH_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"Exa error {r.status_code}: {r.text}")
    data = r.json()
    return data.get("results", [])


# -------------------------
# Fetch + strip HTML
# -------------------------
def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SEC,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MiddleEastTracker/1.0)"},
        )
        if r.status_code != 200:
            return ""

        text = r.text
        # remove scripts/styles
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        # remove tags
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_PAGE_CHARS]
    except Exception:
        return ""


# -------------------------
# OpenAI (Chat Completions)
# -------------------------
def openai_json(prompt_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI Chat Completions with JSON mode (response_format json_object).
    More compatible than /v1/responses for many accounts.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    system = (
        "You are an OSINT analyst for geopolitics -> emerging markets.\n"
        "Return STRICT JSON ONLY (no markdown, no extra keys).\n"
        "Never invent facts. If unsure, reduce confidence or omit.\n"
        "Never invent coordinates. Only include events with trustworthy coords."
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt_obj)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        # show full server message in Actions logs
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


# -------------------------
# Main pipeline
# -------------------------
def build_prompt(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    We ask for 3 outputs:
      - latest_json: summary panel
      - events_geojson: map layer
      - history_record: one row for charts
    """
    return {
        "task": "Generate a Middle East tracker snapshot + map events + trend record.",
        "time_window": "Prioritize last 24-72 hours; pick the most recent credible sources.",
        "output_schema": {
            "latest_json": {
                "generated_at_utc": "ISO-8601 UTC",
                "headline_summary": "array <=10 bullets",
                "market_implications": {
                    "Energy": "array <=5 bullets",
                    "FX": "array <=5 bullets",
                    "Rates": "array <=5 bullets",
                    "Equities": "array <=5 bullets"
                },
                "risk_score": "int 0-100",
                "risk_drivers": "array 3-6 short items",
                "watchlist": "array <=8 objects {asset, direction, why}",
                "sources": "array <=15 objects {title,url,publisher,published_at}"
            },
            "events_geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": "[lon, lat]"},
                        "properties": {
                            "id": "string",
                            "event_type": "string (strike|shipping_disruption|sanctions|diplomacy|protest|cyber|other)",
                            "severity": "int 0-100",
                            "confidence": "low|med|high",
                            "title": "string",
                            "timestamp_utc": "ISO-8601 UTC or null",
                            "location_name": "string",
                            "implication": "1-2 lines",
                            "source_urls": "array of urls"
                        }
                    }
                ]
            },
            "history_record": {
                "generated_at_utc": "ISO-8601 UTC",
                "risk_score": "int 0-100",
                "top_drivers": "array <=3",
                "event_counts_by_type": "object map type->count"
            }
        },
        "rules": [
            "Use only the supplied sources; include URLs in sources/source_urls.",
            "Do NOT invent coordinates. If you only have a vague region, omit the event.",
            "If a claim is unconfirmed, set confidence=low and say so in title/implication.",
            "Prefer fewer, higher-quality events over many guessed ones."
        ],
        "inputs": sources
    }


def clean_geojson(raw_geo: Dict[str, Any]) -> Dict[str, Any]:
    features = raw_geo.get("features", []) or []
    cleaned = []

    for f in features:
        try:
            geom = f.get("geometry", {}) or {}
            coords = geom.get("coordinates", None)
            if not coords or len(coords) != 2:
                continue

            lon, lat = coords[0], coords[1]
            if not validate_lon_lat(lon, lat):
                continue

            p = f.get("properties", {}) or {}
            # ensure id
            p["id"] = p.get("id") or stable_id(
                p.get("title", ""),
                p.get("timestamp_utc", "") or "",
                ",".join(p.get("source_urls", []) or [])
            )

            cleaned.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": p
            })
        except Exception:
            continue

    return {"type": "FeatureCollection", "features": cleaned}


def main() -> None:
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY. Add it in Repo Settings → Secrets and variables → Actions.")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY. Add it in Repo Settings → Secrets and variables → Actions.")

    print(f"[1/4] Exa search: {TRACK_QUERY}")
    results = exa_search(TRACK_QUERY, EXA_MAX_RESULTS)
    print(f"      Exa returned {len(results)} results")

    print("[2/4] Fetching pages…")
    sources = []
    for r in results:
        url = r.get("url")
        if not url:
            continue

        text = fetch_url_text(url)
        if not text:
            continue

        sources.append({
            "title": r.get("title") or "",
            "url": url,
            "publisher": r.get("source") or r.get("author") or "",
            "published_at": r.get("publishedDate") or r.get("published_date") or None,
            "content": text
        })

    print(f"      Fetched {len(sources)} pages with usable text")
    if not sources:
        raise RuntimeError("No sources fetched. Try updating TRACK_QUERY or increase EXA_MAX_RESULTS.")

    print("[3/4] Calling OpenAI…")
    prompt = build_prompt(sources)
    out = openai_json(prompt)

    latest = out.get("latest_json")
    geo = out.get("events_geojson")
    hist = out.get("history_record")

    if not latest or not geo or not hist:
        raise RuntimeError("Model output missing latest_json/events_geojson/history_record.")

    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    hist["generated_at_utc"] = hist.get("generated_at_utc") or latest["generated_at_utc"]

    geo_clean = clean_geojson(geo)

    print("[4/4] Writing output files…")
    write_json(OUT_LATEST, latest)
    write_json(OUT_GEOJSON, geo_clean)
    append_jsonl(OUT_HISTORY, hist)

    print("✅ Done.")
    print(f"   {OUT_LATEST} written")
    print(f"   {OUT_GEOJSON} written with {len(geo_clean.get('features', []))} events")
    print(f"   {OUT_HISTORY} appended")


if __name__ == "__main__":
    main()
