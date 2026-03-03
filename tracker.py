import os
import json
import re
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests

# ==========================
# REQUIRED SECRETS
# ==========================
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================
# CONFIG
# ==========================
TRACK_QUERY = os.getenv(
    "TRACK_QUERY",
    "Iran Middle East escalation Red Sea shipping Strait of Hormuz sanctions strikes drones missiles diplomacy"
)
EXA_MAX_RESULTS = int(os.getenv("EXA_MAX_RESULTS", "12"))
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "20"))
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "15000"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Nominatim (free demo geocoder) — must be polite
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "SensingInsightsTracker/1.0 (contact: github-pages-demo)")
NOMINATIM_MIN_DELAY_SEC = float(os.getenv("NOMINATIM_MIN_DELAY_SEC", "1.1"))  # throttle

# View / analysis scope
REGION_FOCUS = "Middle East"
ALLOW_GLOBAL_PLAYERS = True  # mark global actors if involved

# Optional perspective focus (kept neutral, used only for framing)
PERSPECTIVE_NOTE = os.getenv("PERSPECTIVE_NOTE", "Focus on implications relevant to US/Israel-aligned stakeholders, but keep facts neutral.")

# ==========================
# OUTPUT FILES (repo)
# ==========================
OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"          # append-only time series
OUT_EVENTS = "events.jsonl"            # append-only event memory
OUT_LINKS = "links.jsonl"              # append-only causal edges memory
OUT_GRAPH = "graph.json"               # current graph snapshot
OUT_GEOCODE_CACHE = "geocode_cache.json"  # persistent cache

# ==========================
# TIME HELPERS
# ==========================
IST = timezone(timedelta(hours=5, minutes=30))

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def ist_now_iso() -> str:
    return datetime.now(IST).isoformat()

# ==========================
# UTIL
# ==========================
def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def validate_lon_lat(lon, lat) -> bool:
    try:
        lon = float(lon); lat = float(lat)
        return -180 <= lon <= 180 and -90 <= lat <= 90
    except Exception:
        return False

def read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit is not None:
        return rows[-limit:]
    return rows

# ==========================
# EXA SEARCH
# ==========================
def exa_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "content-type": "application/json"}
    payload = {"query": query, "numResults": num_results, "type": "neural", "useAutoprompt": True}
    r = requests.post(url, headers=headers, json=payload, timeout=FETCH_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Exa error {r.status_code}: {r.text}")
    return r.json().get("results", [])

# ==========================
# FETCH PAGE TEXT (basic)
# ==========================
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

# ==========================
# OPENAI JSON (Chat Completions)
# ==========================
def openai_json(prompt_obj: dict) -> dict:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    system = (
        "You are an OSINT + markets analyst.\n"
        f"Region focus: {REGION_FOCUS}. If global actors are materially involved, mark them.\n"
        f"Perspective note: {PERSPECTIVE_NOTE}\n\n"
        "RULES:\n"
        "1) Output STRICT JSON only. No markdown. No extra keys.\n"
        "2) Use ONLY the supplied sources.\n"
        "3) If a claim is unconfirmed, mark confidence low.\n"
        "4) Do NOT invent coordinates.\n"
        "5) Prefer precision over breadth.\n"
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
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    return json.loads(data["choices"][0]["message"]["content"])

# ==========================
# NOMINATIM GEOCODER (cached + throttled)
# ==========================
class Geocoder:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = read_json(cache_path, default={})
        self.last_call = 0.0

    def save(self):
        write_json(self.cache_path, self.cache)

    def geocode(self, place: str) -> Optional[Tuple[float, float, str]]:
        """
        Returns (lon, lat, display_name) or None.
        Cached by normalized place string.
        """
        if not place or not isinstance(place, str):
            return None
        key = place.strip().lower()

        if key in self.cache:
            v = self.cache[key]
            if v and validate_lon_lat(v["lon"], v["lat"]):
                return (float(v["lon"]), float(v["lat"]), v.get("display_name", place))
            return None

        # throttle
        now = time.time()
        wait = NOMINATIM_MIN_DELAY_SEC - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)

        params = {
            "q": place,
            "format": "json",
            "limit": 1,
        }
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=FETCH_TIMEOUT)
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

# ==========================
# MEMORY: DEDUP EVENTS
# ==========================
def event_fingerprint(e: dict) -> str:
    # stable-ish identity for dedup across runs
    title = (e.get("title") or "").strip().lower()
    loc = (e.get("location_name") or "").strip().lower()
    ts = (e.get("timestamp_utc") or "").strip()
    et = (e.get("event_type") or "").strip().lower()
    return stable_id(title, loc, ts, et)

# ==========================
# BUILD AGENT PROMPT
# ==========================
def build_prompt(sources: List[dict], recent_events: List[dict]) -> dict:
    """
    The model must:
    1) Extract events (no coords needed; we'll geocode)
    2) Create causal links & market chains
    3) Produce market summary + risk scores
    """
    return {
        "task": "Track Middle East developments + generate market insights + causal chains.",
        "inputs": {
            "sources": sources,
            "recent_events_memory": recent_events  # last N events for backtracking/dedup context
        },
        "required_outputs": {
            "latest_json": {
                "generated_at_utc": "ISO-8601 UTC",
                "generated_at_ist": "ISO-8601 IST (+05:30)",
                "headline_summary": "array <=10 bullets",
                "market_implications": {
                    "Energy": "array <=6 bullets",
                    "FX": "array <=6 bullets (focus EM FX pressure channels)",
                    "Rates": "array <=6 bullets (include credit spreads when relevant)",
                    "Equities": "array <=6 bullets"
                },
                "risk_score": "int 0-100",
                "risk_drivers": "array 3-8 short items",
                "watchlist": "array <=10 objects {asset, direction, why}",
                "sources": "array <=15 objects {title,url,publisher,published_at}"
            },
            "events": [
                {
                    "event_type": "strike|shipping_disruption|sanctions|diplomacy|protest|cyber|other",
                    "severity": "int 0-100",
                    "confidence": "low|med|high",
                    "title": "string",
                    "timestamp_utc": "ISO-8601 UTC or null",
                    "location_name": "string (must be geocodable: city/port/strait/base/country + subregion)",
                    "actors": "array of involved parties (include global actors if relevant)",
                    "implication": "1-2 lines",
                    "source_urls": "array of urls"
                }
            ],
            "causal_links": [
                {
                    "from": "string (event_id or market_node)",
                    "to": "string (event_id or market_node)",
                    "relation": "string (causes|increases_risk|pressures|triggers|amplifies)",
                    "confidence": "low|med|high",
                    "why": "one sentence"
                }
            ],
            "market_nodes": [
                "oil_up_risk",
                "em_fx_pressure",
                "risk_off_flows",
                "rates_credit_spreads"
            ],
            "history_record": {
                "generated_at_utc": "ISO-8601 UTC",
                "generated_at_ist": "ISO-8601 IST",
                "risk_score": "int 0-100",
                "top_drivers": "array <=3",
                "counts_by_type": "object map type->count"
            }
        },
        "linking_rules": [
            "Create explicit chains when supported by sources or strong economic reasoning.",
            "Include these canonical chains whenever applicable:",
            "shipping_disruption -> oil_up_risk -> em_fx_pressure",
            "sanctions -> risk_off_flows -> rates_credit_spreads",
            "When linking, prefer concrete mechanism in 'why'."
        ],
        "dedup_rule": "If an event is effectively the same as an event in memory, output it but keep title/location consistent and confidence appropriate.",
        "hard_rules": [
            "Use only supplied sources; cite URLs in source_urls/sources.",
            "Do NOT invent coordinates.",
            "If location is too vague, drop the event."
        ]
    }

# ==========================
# GRAPH BUILD
# ==========================
def build_graph(events_now: List[dict], links: List[dict]) -> dict:
    # nodes: events + market nodes
    nodes = []
    edges = []

    # Market nodes (fixed)
    market_nodes = {
        "oil_up_risk": {"label": "Oil up-risk", "group": "market"},
        "em_fx_pressure": {"label": "EM FX pressure", "group": "market"},
        "risk_off_flows": {"label": "Risk-off flows", "group": "market"},
        "rates_credit_spreads": {"label": "Rates/Credit spreads", "group": "market"},
    }
    for nid, meta in market_nodes.items():
        nodes.append({"id": nid, **meta})

    # Event nodes
    for e in events_now:
        eid = e["id"]
        nodes.append({
            "id": eid,
            "label": (e.get("title") or "Event")[:60],
            "group": "event",
            "severity": e.get("severity", 0),
            "confidence": e.get("confidence", "low"),
            "event_type": e.get("event_type", "other"),
            "timestamp_utc": e.get("timestamp_utc"),
            "location_name": e.get("location_name")
        })

    # Edges
    for l in links:
        edges.append({
            "from": l.get("from"),
            "to": l.get("to"),
            "label": l.get("relation", ""),
            "confidence": l.get("confidence", "low"),
            "why": l.get("why", "")
        })

    return {"nodes": nodes, "edges": edges}

# ==========================
# MAIN
# ==========================
def main():
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY secret")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY secret")

    # Load memory
    past_events = read_jsonl(OUT_EVENTS, limit=500)  # last 500 for context
    past_links = read_jsonl(OUT_LINKS, limit=1000)

    geocoder = Geocoder(OUT_GEOCODE_CACHE)

    # 1) Search + fetch
    results = exa_search(TRACK_QUERY, EXA_MAX_RESULTS)

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

    if not sources:
        raise RuntimeError("No sources fetched. Try changing TRACK_QUERY or increasing EXA_MAX_RESULTS.")

    # 2) LLM extraction + linking
    prompt = build_prompt(sources=sources, recent_events=past_events[-80:])
    out = openai_json(prompt)

    latest = out.get("latest_json")
    events = out.get("events") or []
    causal_links = out.get("causal_links") or []
    hist = out.get("history_record")

    if not latest or not isinstance(events, list) or not hist:
        raise RuntimeError("Model output missing required keys: latest_json/events/history_record.")

    # 3) Add timestamps in both zones
    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    latest["generated_at_ist"] = latest.get("generated_at_ist") or ist_now_iso()

    hist["generated_at_utc"] = hist.get("generated_at_utc") or latest["generated_at_utc"]
    hist["generated_at_ist"] = hist.get("generated_at_ist") or latest["generated_at_ist"]

    # 4) Normalize events, geocode, assign ids, dedup
    existing_fps = {e.get("fingerprint"): True for e in past_events if e.get("fingerprint")}
    events_now_norm = []
    geo_features = []

    for e in events:
        try:
            # Minimal validation
            e_type = e.get("event_type", "other")
            title = e.get("title") or "Event"
            loc = (e.get("location_name") or "").strip()
            if not loc:
                continue

            # Create fingerprint/id
            fp = event_fingerprint(e)
            eid = e.get("id") or stable_id(fp)

            # Geocode (cache + throttle)
            geo = geocoder.geocode(loc)
            if not geo:
                # If we can't geocode, skip plotting, but still keep memory event if you want
                # We'll skip it entirely to avoid empty map noise.
                continue
            lon, lat, disp = geo
            if not validate_lon_lat(lon, lat):
                continue

            # Normalize
            ev = {
                "id": eid,
                "fingerprint": fp,
                "event_type": e_type,
                "severity": int(e.get("severity", 0)),
                "confidence": e.get("confidence", "low"),
                "title": title,
                "timestamp_utc": e.get("timestamp_utc"),
                "location_name": loc,
                "location_resolved": disp,
                "actors": e.get("actors") or [],
                "implication": e.get("implication") or "",
                "source_urls": e.get("source_urls") or []
            }

            # Save to current batch
            events_now_norm.append(ev)

            # GeoJSON feature
            geo_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": eid,
                    "event_type": ev["event_type"],
                    "severity": ev["severity"],
                    "confidence": ev["confidence"],
                    "title": ev["title"],
                    "timestamp_utc": ev["timestamp_utc"],
                    "location_name": ev["location_name"],
                    "location_resolved": ev["location_resolved"],
                    "actors": ev["actors"],
                    "implication": ev["implication"],
                    "source_urls": ev["source_urls"],
                }
            })

            # Append to memory only if new fingerprint
            if fp not in existing_fps:
                append_jsonl(OUT_EVENTS, {**ev, "seen_at_utc": latest["generated_at_utc"], "seen_at_ist": latest["generated_at_ist"]})
                existing_fps[fp] = True

        except Exception:
            continue

    geocoder.save()

    # 5) Create canonical market chains + use model links
    # Ensure these market nodes exist for linking
    market_nodes = ["oil_up_risk", "em_fx_pressure", "risk_off_flows", "rates_credit_spreads"]

    # Map event types to canonical chains when appropriate
    derived_links = []
    for ev in events_now_norm:
        if ev["event_type"] == "shipping_disruption":
            derived_links.append({
                "from": ev["id"], "to": "oil_up_risk",
                "relation": "increases_risk", "confidence": "med",
                "why": "Shipping disruption can constrain supply routes and raise oil risk premia."
            })
            derived_links.append({
                "from": "oil_up_risk", "to": "em_fx_pressure",
                "relation": "pressures", "confidence": "med",
                "why": "Higher oil prices often worsen import bills and pressure EM FX for net importers."
            })
        if ev["event_type"] == "sanctions":
            derived_links.append({
                "from": ev["id"], "to": "risk_off_flows",
                "relation": "triggers", "confidence": "med",
                "why": "Sanctions can increase uncertainty and drive risk-off positioning."
            })
            derived_links.append({
                "from": "risk_off_flows", "to": "rates_credit_spreads",
                "relation": "amplifies", "confidence": "med",
                "why": "Risk-off flows can widen credit spreads and push rates/term premia higher."
            })

    # Normalize model-provided causal links to actual ids when possible
    # If model used titles or placeholders, we try to map by substring to event ids.
    title_to_id = {ev["title"].lower(): ev["id"] for ev in events_now_norm}

    def resolve_node(x: str) -> str:
        if not x:
            return x
        if x in market_nodes:
            return x
        # try direct match to event id
        if any(ev["id"] == x for ev in events_now_norm):
            return x
        # try title match
        xl = x.strip().lower()
        for t, eid in title_to_id.items():
            if xl == t or xl in t or t in xl:
                return eid
        return x  # keep as-is; graph will still show it

    normalized_links = []
    for l in (causal_links or []):
        try:
            frm = resolve_node(l.get("from", ""))
            to = resolve_node(l.get("to", ""))
            if not frm or not to:
                continue
            normalized_links.append({
                "from": frm,
                "to": to,
                "relation": l.get("relation", "causes"),
                "confidence": l.get("confidence", "low"),
                "why": l.get("why", "")
            })
        except Exception:
            continue

    # Combine + persist links (append-only, dedup lightly by hash)
    existing_link_ids = {stable_id(x.get("from",""), x.get("to",""), x.get("relation",""), x.get("why","")): True for x in past_links}
    for l in derived_links + normalized_links:
        lid = stable_id(l.get("from",""), l.get("to",""), l.get("relation",""), l.get("why",""))
        if lid in existing_link_ids:
            continue
        append_jsonl(OUT_LINKS, {**l, "id": lid, "seen_at_utc": latest["generated_at_utc"], "seen_at_ist": latest["generated_at_ist"]})
        existing_link_ids[lid] = True

    # 6) Write snapshot files
    geojson = {"type": "FeatureCollection", "features": geo_features}
    write_json(OUT_GEOJSON, geojson)
    write_json(OUT_LATEST, latest)
    append_jsonl(OUT_HISTORY, hist)

    # 7) Build graph snapshot from last N events + last N links
    all_events_for_graph = read_jsonl(OUT_EVENTS, limit=200)
    all_links_for_graph = read_jsonl(OUT_LINKS, limit=400)

    # Graph should focus on recent visible events (last 72-ish)
    graph_events = all_events_for_graph[-120:]
    graph = build_graph(events_now=graph_events, links=all_links_for_graph)
    write_json(OUT_GRAPH, graph)

    print("OK: wrote latest.json, events.geojson, graph.json; appended history/events/links.")
    print("Map events this run:", len(geo_features))


if __name__ == "__main__":
    main()
