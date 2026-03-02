import os
import json
import re
import hashlib
from datetime import datetime, timezone

import requests

# ========= Required secrets (GitHub → Settings → Secrets → Actions) =========
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========= Config (optional) =========
TRACK_QUERY = os.getenv(
    "TRACK_QUERY",
    "Iran Middle East escalation Red Sea shipping Strait of Hormuz sanctions strikes drones missiles diplomacy"
)
EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "10"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "20"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "12000"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def validate_lon_lat(lon, lat) -> bool:
    try:
        lon = float(lon)
        lat = float(lat)
        return -180 <= lon <= 180 and -90 <= lat <= 90
    except Exception:
        return False


def exa_search(query: str, num_results: int):
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {"query": query, "numResults": num_results, "type": "neural", "useAutoprompt": True}
    r = requests.post(url, headers=headers, json=payload, timeout=FETCH_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Exa error {r.status_code}: {r.text}")
    return r.json().get("results", [])


def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        text = r.text
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def openai_json(prompt_obj: dict) -> dict:
    """
    IMPORTANT: Uses /v1/chat/completions ONLY. No /v1/responses anywhere.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    system = (
        "You are an OSINT analyst for geopolitics -> emerging markets.\n"
        "Return STRICT JSON only (no markdown, no extra keys).\n"
        "Never invent coordinates. If unsure, omit the event.\n"
        "Use only the supplied sources and include URLs."
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
        # show the real error in Actions logs
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY secret.")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY secret.")

    # 1) Search
    results = exa_search(TRACK_QUERY, EXA_MAX_RESULTS)

    # 2) Fetch pages
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
            "content": text,
        })

    if not sources:
        raise RuntimeError("No sources fetched. Change TRACK_QUERY or increase EXA_MAX_RESULTS.")

    # 3) Ask LLM for structured outputs
    prompt = {
        "task": "Generate (1) snapshot JSON, (2) GeoJSON events, (3) a trend record.",
        "output_schema": {
            "latest_json": {
                "generated_at_utc": "ISO-8601 UTC",
                "headline_summary": "array <=10 bullets",
                "market_implications": {
                    "Energy": "array <=5",
                    "FX": "array <=5",
                    "Rates": "array <=5",
                    "Equities": "array <=5",
                },
                "risk_score": "int 0-100",
                "risk_drivers": "array 3-6",
                "watchlist": "array <=8 objects {asset, direction, why}",
                "sources": "array <=15 objects {title,url,publisher,published_at}",
            },
            "events_geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": "[lon, lat]"},
                        "properties": {
                            "id": "string",
                            "event_type": "string",
                            "severity": "int 0-100",
                            "confidence": "low|med|high",
                            "title": "string",
                            "timestamp_utc": "ISO-8601 UTC or null",
                            "location_name": "string",
                            "implication": "1-2 lines",
                            "source_urls": "array of urls",
                        },
                    }
                ],
            },
            "history_record": {
                "generated_at_utc": "ISO-8601 UTC",
                "risk_score": "int 0-100",
                "top_drivers": "array <=3",
                "event_counts_by_type": "object map type->count",
            },
        },
        "rules": [
            "Do not invent coordinates. Only include events you can confidently locate.",
            "Use only the provided sources; include their URLs.",
        ],
        "inputs": sources,
    }

    out = openai_json(prompt)

    latest = out.get("latest_json")
    geo = out.get("events_geojson")
    hist = out.get("history_record")
    if not latest or not geo or not hist:
        raise RuntimeError("Missing keys in model output. Expected latest_json/events_geojson/history_record.")

    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    hist["generated_at_utc"] = hist.get("generated_at_utc") or latest["generated_at_utc"]

    # Clean/validate GeoJSON
    cleaned = []
    for f in (geo.get("features") or []):
        try:
            coords = (f.get("geometry") or {}).get("coordinates")
            if not coords or len(coords) != 2:
                continue
            lon, lat = coords
            if not validate_lon_lat(lon, lat):
                continue
            p = f.get("properties") or {}
            p["id"] = p.get("id") or stable_id(p.get("title", ""), p.get("timestamp_utc") or "", ",".join(p.get("source_urls") or []))
            cleaned.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}, "properties": p})
        except Exception:
            continue

    geo_clean = {"type": "FeatureCollection", "features": cleaned}

    # Write outputs
    write_json(OUT_LATEST, latest)
    write_json(OUT_GEOJSON, geo_clean)
    append_jsonl(OUT_HISTORY, hist)

    print("OK:", OUT_LATEST, OUT_GEOJSON, OUT_HISTORY, "events:", len(cleaned))


if __name__ == "__main__":
    main()
