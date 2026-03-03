import os, json, re, time, hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import requests

# ===== Secrets =====
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== Config =====
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "25"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "18000"))

# More coverage
EXA_PER_QUERY = int(os.getenv("EXA_PER_QUERY", "10"))  # 6 queries => ~60 candidates
MAX_FETCH_URLS = int(os.getenv("MAX_FETCH_URLS", "30"))  # fetch top N pages (token cost control)

# Nominatim (polite usage)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "SensingInsightsTracker/1.0 (demo)")
NOMINATIM_MIN_DELAY_SEC = float(os.getenv("NOMINATIM_MIN_DELAY_SEC", "1.1"))

# Output files
OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"
OUT_GRAPH = "graph.json"
OUT_GEOCODE_CACHE = "geocode_cache.json"

# Time zones
IST = timezone(timedelta(hours=5, minutes=30))
def utc_now_iso(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def ist_now_iso(): return datetime.now(IST).isoformat()

# ---- Fallback centroids (so map is NEVER empty) ----
# (lon, lat)
CENTROIDS = {
    "iran": (53.6880, 32.4279),
    "israel": (34.8516, 31.0461),
    "gaza": (34.3088, 31.3547),
    "west bank": (35.2, 31.9),
    "lebanon": (35.8623, 33.8547),
    "syria": (38.9968, 34.8021),
    "iraq": (43.6793, 33.2232),
    "yemen": (48.5164, 15.5527),
    "oman": (55.9233, 21.4735),
    "uae": (54.3773, 24.4539),
    "qatar": (51.1839, 25.3548),
    "saudi arabia": (45.0792, 23.8859),
    "jordan": (36.2384, 30.5852),
    "egypt": (30.8025, 26.8206),
    "turkey": (35.2433, 38.9637),
    "kuwait": (47.4818, 29.3117),
    "bahrain": (50.5577, 26.0667),
    "red sea": (38.0, 20.0),
    "strait of hormuz": (56.25, 26.58),
    "bab el-mandeb": (43.33, 12.64),
}

MARKET_NODES = {
    "oil_up_risk": "Oil up-risk",
    "em_fx_pressure": "EM FX pressure",
    "risk_off_flows": "Risk-off flows",
    "rates_credit_spreads": "Rates/Credit spreads",
    "shipping_risk": "Shipping risk",
    "energy_supply_risk": "Energy supply risk",
}

def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def clean_text_from_html(html: str) -> str:
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"(?is)<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_PAGE_CHARS]

def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        return clean_text_from_html(r.text)
    except Exception:
        return ""

def exa_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {"query": query, "numResults": num_results, "type": "neural", "useAutoprompt": True}
    r = requests.post(url, headers=headers, json=payload, timeout=FETCH_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Exa error {r.status_code}: {r.text}")
    return r.json().get("results", [])

def openai_json(prompt_obj: dict) -> dict:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system = (
        "You are an OSINT + markets analyst focused on Middle East events and EM market impacts.\n"
        "Output STRICT JSON only (no markdown, no extra keys).\n"
        "Extract MANY events (aim 25-80 if sources allow). Prefer breadth WITH confidence labels.\n"
        "Never invent facts. Use only provided sources.\n"
        "For locations: produce geocodable strings (city/port/strait/country). If unsure, use country.\n"
        "Also output causal links: shipping->oil->EM FX and sanctions->risk-off->rates/spreads when applicable.\n"
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt_obj)}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    return json.loads(r.json()["choices"][0]["message"]["content"])

class Geocoder:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, "r", encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.last_call = 0.0

    def save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def geocode(self, place: str) -> Optional[Tuple[float, float, str]]:
        if not place:
            return None
        key = place.strip().lower()
        if key in self.cache:
            v = self.cache[key]
            if v is None: return None
            return (v["lon"], v["lat"], v.get("display_name", place))

        # throttle
        now = time.time()
        wait = NOMINATIM_MIN_DELAY_SEC - (now - self.last_call)
        if wait > 0: time.sleep(wait)

        try:
            r = requests.get(
                NOMINATIM_URL,
                params={"q": place, "format": "json", "limit": 1},
                headers={"User-Agent": NOMINATIM_USER_AGENT},
                timeout=FETCH_TIMEOUT
            )
            self.last_call = time.time()
            if r.status_code != 200:
                self.cache[key] = None
                return None
            js = r.json()
            if not js:
                self.cache[key] = None
                return None
            top = js[0]
            lon = float(top["lon"]); lat = float(top["lat"])
            disp = top.get("display_name", place)
            self.cache[key] = {"lon": lon, "lat": lat, "display_name": disp}
            return (lon, lat, disp)
        except Exception:
            self.cache[key] = None
            return None

def centroid_fallback(location_name: str) -> Optional[Tuple[float, float, str]]:
    if not location_name:
        return None
    k = location_name.strip().lower()
    # direct match
    if k in CENTROIDS:
        lon, lat = CENTROIDS[k]
        return (lon, lat, f"Centroid: {location_name}")
    # partial match
    for key, (lon, lat) in CENTROIDS.items():
        if key in k:
            return (lon, lat, f"Centroid: {key}")
    return None

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_queries() -> List[str]:
    # multiple queries => less repetitive sources
    return [
        "Middle East Israel Iran Hezbollah Hamas strikes missiles drones last 72 hours",
        "Red Sea Bab el-Mandeb shipping disruption attacks insurance rerouting last 72 hours",
        "Strait of Hormuz shipping disruption tanker risk last 30 days",
        "Iran sanctions designation enforcement US EU last 30 days",
        "Israel Gaza ceasefire talks diplomacy hostage negotiations last 30 days",
        "Middle East conflict spillover oil price risk emerging markets fx spreads last 30 days",
    ]

def main():
    if not EXA_API_KEY: raise SystemExit("Missing EXA_API_KEY")
    if not OPENAI_API_KEY: raise SystemExit("Missing OPENAI_API_KEY")

    # 1) Multi-query Exa
    urls = {}
    for q in build_queries():
        for r in exa_search(q, EXA_PER_QUERY):
            u = r.get("url")
            if not u: continue
            # keep the best title/source metadata we saw
            if u not in urls:
                urls[u] = {
                    "title": r.get("title") or "",
                    "url": u,
                    "publisher": r.get("source") or r.get("author") or "",
                    "published_at": r.get("publishedDate") or r.get("published_date") or None,
                }

    # 2) Fetch top N pages
    fetched = []
    for meta in list(urls.values())[:MAX_FETCH_URLS]:
        text = fetch_url_text(meta["url"])
        if not text: continue
        fetched.append({**meta, "content": text})

    if not fetched:
        raise RuntimeError("No pages fetched. Lower MAX_FETCH_URLS or change queries.")

    # 3) Ask LLM for many events + links + summary
    prompt = {
        "task": "Extract events + market impacts + causal links.",
        "output_schema": {
            "latest": {
                "generated_at_utc": "ISO-8601 UTC",
                "generated_at_ist": "ISO-8601 IST",
                "headline_summary": "array <=10",
                "risk_score": "0-100",
                "risk_drivers": "array 3-8",
                "market_implications": {"Energy": "array", "FX": "array", "Rates": "array", "Equities": "array"},
                "sources": "array <=20 objects {title,url,publisher,published_at}"
            },
            "events": "array of event objects",
            "links": "array of causal edges"
        },
        "event_object": {
            "event_type": "strike|shipping_disruption|sanctions|diplomacy|protest|cyber|other",
            "severity": "0-100",
            "confidence": "low|med|high",
            "title": "string",
            "timestamp_utc": "ISO-8601 or null",
            "location_name": "string (geocodable; if unsure use a country or strait name)",
            "actors": "array",
            "implication": "1-2 lines",
            "source_urls": "array"
        },
        "required_market_nodes": list(MARKET_NODES.keys()),
        "canonical_chains": [
            "shipping_disruption -> shipping_risk -> oil_up_risk -> em_fx_pressure",
            "sanctions -> risk_off_flows -> rates_credit_spreads"
        ],
        "inputs": fetched
    }

    out = openai_json(prompt)
    latest = out.get("latest") or {}
    events = out.get("events") or []
    links = out.get("links") or []

    # timestamps
    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    latest["generated_at_ist"] = latest.get("generated_at_ist") or ist_now_iso()

    # 4) Geocode events (with fallback so map is not empty)
    geocoder = Geocoder(OUT_GEOCODE_CACHE)
    features = []
    node_events = []

    for e in events:
        try:
            loc = (e.get("location_name") or "").strip()
            if not loc:
                continue

            eid = stable_id(e.get("title",""), loc, e.get("timestamp_utc") or "", e.get("event_type") or "")
            geo = geocoder.geocode(loc)

            if geo is None:
                # try fallback centroid based on location string
                geo = centroid_fallback(loc)

            if geo is None:
                # last resort: if actors mention a country, try centroid
                for a in (e.get("actors") or []):
                    fb = centroid_fallback(str(a))
                    if fb:
                        geo = fb
                        break

            if geo is None:
                continue  # truly unplaceable

            lon, lat, disp = geo
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": eid,
                    "event_type": e.get("event_type", "other"),
                    "severity": int(e.get("severity", 0)),
                    "confidence": e.get("confidence", "low"),
                    "title": e.get("title", "Event"),
                    "timestamp_utc": e.get("timestamp_utc"),
                    "location_name": loc,
                    "location_resolved": disp,
                    "actors": e.get("actors") or [],
                    "implication": e.get("implication") or "",
                    "source_urls": e.get("source_urls") or [],
                }
            })

            node_events.append({
                "id": eid,
                "label": (e.get("title") or "Event")[:80],
                "group": "event",
                "severity": int(e.get("severity", 0)),
                "confidence": e.get("confidence", "low"),
                "event_type": e.get("event_type", "other"),
            })
        except Exception:
            continue

    geocoder.save()

    # 5) Build dense graph snapshot
    nodes = []
    edges = []

    # market nodes always present
    for nid, label in MARKET_NODES.items():
        nodes.append({"id": nid, "label": label, "group": "market"})

    # event nodes
    nodes.extend(node_events)

    # helper to see if an event id exists
    event_ids = set(n["id"] for n in node_events)

    # add canonical edges from event types (always)
    for f in features:
        p = f["properties"]
        eid = p["id"]
        if p["event_type"] == "shipping_disruption":
            edges += [
                {"from": eid, "to": "shipping_risk", "label": "increases_risk", "confidence": "med", "why": "Disruption raises route risk/insurance/rerouting."},
                {"from": "shipping_risk", "to": "oil_up_risk", "label": "pressures", "confidence": "med", "why": "Shipping risk can add risk premia to energy flows."},
                {"from": "oil_up_risk", "to": "em_fx_pressure", "label": "pressures", "confidence": "med", "why": "Higher oil can worsen EM import bills and FX."},
            ]
        if p["event_type"] == "sanctions":
            edges += [
                {"from": eid, "to": "risk_off_flows", "label": "triggers", "confidence": "med", "why": "Sanctions raise uncertainty and risk premia."},
                {"from": "risk_off_flows", "to": "rates_credit_spreads", "label": "amplifies", "confidence": "med", "why": "Risk-off often widens spreads and tightens financial conditions."},
            ]

    # include model links (normalize “from/to” if it’s a known market node or event title)
    def norm_node(x: str) -> str:
        if not x: return x
        xl = x.strip()
        if xl in MARKET_NODES: return xl
        if xl in event_ids: return xl
        return xl  # keep; D3 will still show it as isolated if not in nodes

    # ensure nodes exist for any non-market non-event endpoints
    node_ids = set(n["id"] for n in nodes)
    for l in links:
        frm = norm_node(str(l.get("from","")))
        to = norm_node(str(l.get("to","")))
        if not frm or not to: continue
        if frm not in node_ids:
            nodes.append({"id": frm, "label": frm[:40], "group": "market"})
            node_ids.add(frm)
        if to not in node_ids:
            nodes.append({"id": to, "label": to[:40], "group": "market"})
            node_ids.add(to)
        edges.append({
            "from": frm, "to": to,
            "label": l.get("relation","causes"),
            "confidence": l.get("confidence","low"),
            "why": l.get("why","")
        })

    # 6) Write outputs
    write_json(OUT_GEOJSON, {"type": "FeatureCollection", "features": features})
    write_json(OUT_GRAPH, {"nodes": nodes, "edges": edges})
    write_json(OUT_LATEST, latest)
    append_jsonl(OUT_HISTORY, {
        "generated_at_utc": latest["generated_at_utc"],
        "generated_at_ist": latest["generated_at_ist"],
        "risk_score": latest.get("risk_score", 0),
        "top_drivers": (latest.get("risk_drivers") or [])[:3],
    })

    print("OK: events:", len(features), "nodes:", len(nodes), "edges:", len(edges))

if __name__ == "__main__":
    main()
