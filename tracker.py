import os, json, re, time, hashlib
from datetime import datetime, timezone, timedelta

import requests

# ----------------------------
# CONFIG (edit if you want)
# ----------------------------
QUERY = os.getenv(
    "TRACK_QUERY",
    "Iran Middle East conflict escalation Red Sea shipping Strait of Hormuz sanctions strikes drones missiles diplomacy"
)
EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "12"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "20"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"  # append-only

# ----------------------------
# REQUIRED SECRETS
# ----------------------------
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def exa_search(query: str, num_results: int = 10):
    """
    Minimal Exa Search call.
    If Exa changes their endpoint/shape, adjust this function.
    """
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {
        "query": query,
        "numResults": num_results,
        "type": "neural",
        "useAutoprompt": True,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_url(url: str) -> str:
    """
    Fetches a page and crudely strips HTML to reduce tokens.
    Some sites block bots/paywall; those will return empty.
    """
    try:
        r = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TrackerBot/1.0)"},
        )
        if r.status_code != 200:
            return ""
        text = r.text
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:20000]  # cap to control cost
    except Exception:
        return ""


def openai_json(prompt_obj: dict) -> dict:
    """
    Uses OpenAI Responses API with json_object output.
    """
    url = f"{OPENAI_BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    system = (
        "You are an OSINT analyst. Output STRICT JSON only. "
        "No markdown, no commentary, no extra keys."
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt_obj)},
        ],
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    out_text = None
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_text = c.get("text")
                break
        if out_text:
            break

    if not out_text:
        raise RuntimeError("Model returned no JSON text.")
    return json.loads(out_text)


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


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY (set in GitHub Secrets).")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY (set in GitHub Secrets).")

    # 1) Search
    results = exa_search(QUERY, EXA_MAX_RESULTS)

    # 2) Fetch content
    sources = []
    for r in results:
        url = r.get("url")
        if not url:
            continue
        text = fetch_url(url)
        if not text:
            continue
        sources.append(
            {
                "title": r.get("title") or "",
                "url": url,
                "publisher": r.get("source") or r.get("author") or "",
                "published_at": r.get("publishedDate") or r.get("published_date") or None,
                "content": text,
            }
        )

    if not sources:
        raise RuntimeError(
            "No pages fetched. Try changing TRACK_QUERY or increase EXA_MAX_RESULTS."
        )

    # 3) Ask LLM to produce both latest.json + events.geojson + a trend record
    prompt = {
        "task": "Create a Middle East tracker snapshot and a GeoJSON event layer.",
        "time_window": "Focus on last 24-72 hours if possible; prioritize most recent.",
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
                            "event_type": "string (e.g., strike, shipping_disruption, sanctions, diplomacy, protest, cyber)",
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
            "Use only supplied sources; put URLs into sources/source_urls.",
            "Do NOT invent coordinates. Only include events with known coordinates in sources or universally-known city coordinates.",
            "If uncertain, lower confidence or omit.",
        ],
        "inputs": sources,
    }

    out = openai_json(prompt)

    latest = out.get("latest_json")
    geo = out.get("events_geojson")
    hist = out.get("history_record")

    if not latest or not geo or not hist:
        raise RuntimeError("Model output missing latest_json/events_geojson/history_record")

    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    hist["generated_at_utc"] = hist.get("generated_at_utc") or latest["generated_at_utc"]

    # 4) Clean/validate GeoJSON
    cleaned_features = []
    for f in geo.get("features", []) or []:
        try:
            coords = f.get("geometry", {}).get("coordinates", None)
            if not coords or len(coords) != 2:
                continue
            lon, lat = coords[0], coords[1]
            if not validate_lon_lat(lon, lat):
                continue

            p = f.get("properties", {}) or {}
            p["id"] = p.get("id") or stable_id(p.get("title", ""), p.get("timestamp_utc", ""), ",".join(p.get("source_urls", []) or []))

            cleaned_features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": p,
                }
            )
        except Exception:
            continue

    geo_clean = {"type": "FeatureCollection", "features": cleaned_features}

    # 5) Write files
    write_json(OUT_LATEST, latest)
    write_json(OUT_GEOJSON, geo_clean)
    append_jsonl(OUT_HISTORY, hist)

    print("Wrote:", OUT_LATEST, OUT_GEOJSON, OUT_HISTORY)
    print("Events on map:", len(cleaned_features))


if __name__ == "__main__":
    main()
