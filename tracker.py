import os
import json
import re
import time
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set

import requests

# ==========================
# REQUIRED SECRETS
# ==========================
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================
# CONFIG
# ==========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "25"))

# Output files (repo root)
OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_HISTORY = "history.jsonl"
OUT_GRAPH = "graph.json"
OUT_GEOCODE_CACHE = "geocode_cache.json"
OUT_EXTRACT = "op_extract.json"  # <-- OpenPlanter writes this

# Time zones
IST = timezone(timedelta(hours=5, minutes=30))

# Nominatim (polite usage)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "SensingInsightsTracker/1.0 (github-actions)")
NOMINATIM_MIN_DELAY_SEC = float(os.getenv("NOMINATIM_MIN_DELAY_SEC", "1.2"))

# Fallback centroids (lon, lat)
CENTROIDS: Dict[str, Tuple[float, float]] = {
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

MARKET_NODES = [
    {"id": "oil_up_risk", "label": "Oil up-risk", "group": "market"},
    {"id": "em_fx_pressure", "label": "EM FX pressure", "group": "market"},
    {"id": "risk_off_flows", "label": "Risk-off flows", "group": "market"},
    {"id": "rates_credit_spreads", "label": "Rates/Credit spreads", "group": "market"},
]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def ist_now_iso() -> str:
    return datetime.now(IST).isoformat()

def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def centroid_fallback(location_name: str) -> Optional[Tuple[float, float, str]]:
    if not location_name:
        return None
    k = location_name.strip().lower()
    if k in CENTROIDS:
        lon, lat = CENTROIDS[k]
        return lon, lat, f"Centroid: {location_name}"
    for key, (lon, lat) in CENTROIDS.items():
        if key in k:
            return lon, lat, f"Centroid: {key}"
    return None

class Geocoder:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
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
            if v is None:
                return None
            return (float(v["lon"]), float(v["lat"]), v.get("display_name", place))

        now = time.time()
        wait = NOMINATIM_MIN_DELAY_SEC - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)

        try:
            r = requests.get(
                NOMINATIM_URL,
                params={"q": place, "format": "json", "limit": 1},
                headers={"User-Agent": NOMINATIM_USER_AGENT},
                timeout=FETCH_TIMEOUT,
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
            lon = float(top["lon"])
            lat = float(top["lat"])
            disp = top.get("display_name", place)
            self.cache[key] = {"lon": lon, "lat": lat, "display_name": disp}
            return (lon, lat, disp)
        except Exception:
            self.cache[key] = None
            return None

def normalize_links(raw_links: Any) -> List[Dict[str, Any]]:
    # Only accept dicts; ignore strings to prevent crashes.
    if not isinstance(raw_links, list):
        return []
    cleaned = []
    for item in raw_links:
        if isinstance(item, dict):
            cleaned.append(item)
    return cleaned

def ensure_graph_nodes(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    id_set: Set[str] = set(n.get("id") for n in nodes if n.get("id"))
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm and frm not in id_set:
            nodes.append({"id": frm, "label": str(frm).replace("_", " "), "group": "event_or_concept"})
            id_set.add(frm)
        if to and to not in id_set:
            nodes.append({"id": to, "label": str(to).replace("_", " "), "group": "event_or_concept"})
            id_set.add(to)
    return nodes

def run_openplanter_extract(workspace: str):
    # Force OpenAI so Anthropic default doesn’t break runs.
    task = f"""
You are an OSINT + markets tracking agent.

Goal: Track Middle East events and market implications, and infer relationships between events.

You MUST write EXACTLY one JSON file to the workspace root:
- {OUT_EXTRACT}

Use a tool to write the file to path "./{OUT_EXTRACT}" (repo root).

The JSON MUST have exactly these top-level keys:
- latest (object)
- events (array)
- links (array)

latest object:
- headline_summary (array <=10 strings)
- risk_score (0-100 int)
- risk_drivers (array 3-8 strings)
- market_implications (object with keys Energy, FX, Rates, Equities each array of strings)
- sources (array of objects: title,url,publisher,published_at)

events array: each item has
- event_type (strike|shipping_disruption|sanctions|diplomacy|protest|cyber|other)
- severity (0-100 int)
- confidence (low|med|high)
- title (string)
- timestamp_utc (ISO-8601 or null)
- location_name (geocodable string; if unsure use country/strait)
- actors (array of strings)
- implication (1-2 lines)
- source_urls (array of urls)

links array: infer relationships between events AND event->market concepts.
Each link is an object with:
- from (string: an event title OR a market concept id)
- to (string: an event title OR a market concept id)
- relation (string: e.g., triggers|retaliates|escalates|disrupts|pressures|raises_risk)
- confidence (low|med|high)
- why (short justification)

Market concept ids you may use:
- oil_up_risk
- em_fx_pressure
- risk_off_flows
- rates_credit_spreads

Rules:
- Use ONLY credible sources you can cite in source_urls.
- Produce at least 15 events if sources allow.
- links must be objects (no strings).
"""

    cmd = [
        "openplanter-agent",
        "--headless",
        "--provider", "openai",
        "--model", OPENAI_MODEL,
        "--task", task,
        "--workspace", workspace,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY")

    workspace = os.path.abspath(".")
    run_openplanter_extract(workspace)

    if not os.path.exists(OUT_EXTRACT):
        raise SystemExit(f"OpenPlanter did not write {OUT_EXTRACT} to repo root.")

    with open(OUT_EXTRACT, "r", encoding="utf-8") as f:
        out = json.load(f)

    latest = out.get("latest") if isinstance(out.get("latest"), dict) else {}
    events = out.get("events") if isinstance(out.get("events"), list) else []
    links = normalize_links(out.get("links"))

    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    latest["generated_at_ist"] = latest.get("generated_at_ist") or ist_now_iso()

    # Geocode -> GeoJSON + event nodes
    geocoder = Geocoder(OUT_GEOCODE_CACHE)
    features = []
    event_nodes = []

    # map from event title to event_id so links can refer to titles
    title_to_id: Dict[str, str] = {}

    for e in events:
        if not isinstance(e, dict):
            continue
        loc = (e.get("location_name") or "").strip()
        title = (e.get("title") or "Event").strip()
        if not loc or not title:
            continue

        eid = stable_id(title, loc, e.get("timestamp_utc") or "", e.get("event_type") or "")
        title_to_id[title] = eid

        geo = geocoder.geocode(loc) or centroid_fallback(loc)
        if geo is None:
            # try actor-based fallback
            for a in (e.get("actors") or []):
                fb = centroid_fallback(str(a))
                if fb:
                    geo = fb
                    break
        if geo is None:
            continue

        lon, lat, disp = geo

        props = {
            "id": eid,
            "event_type": e.get("event_type", "other"),
            "severity": int(e.get("severity", 0) or 0),
            "confidence": e.get("confidence", "low"),
            "title": title,
            "timestamp_utc": e.get("timestamp_utc"),
            "location_name": loc,
            "location_resolved": disp,
            "actors": e.get("actors") or [],
            "implication": e.get("implication") or "",
            "source_urls": e.get("source_urls") or [],
        }

        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})

        event_nodes.append({
            "id": eid,
            "label": title[:80],
            "group": "event",
            "severity": props["severity"],
            "confidence": props["confidence"],
            "event_type": props["event_type"],
            "timestamp_utc": props["timestamp_utc"],
            "location_name": props["location_name"],
        })

    geocoder.save()

    # Build graph from OpenPlanter links
    nodes = MARKET_NODES + event_nodes
    edges = []

    for l in links:
        frm = (l.get("from") or "").strip()
        to = (l.get("to") or "").strip()
        if not frm or not to:
            continue

        # If link refers to event title, convert to event id
        frm_id = title_to_id.get(frm, frm)
        to_id = title_to_id.get(to, to)

        edges.append({
            "from": frm_id,
            "to": to_id,
            "label": (l.get("relation") or "causes"),
            "confidence": (l.get("confidence") or "low"),
            "why": (l.get("why") or ""),
        })

    nodes = ensure_graph_nodes(nodes, edges)

    # Write outputs your HTML uses
    write_json(OUT_GEOJSON, {"type": "FeatureCollection", "features": features})
    write_json(OUT_GRAPH, {"nodes": nodes, "edges": edges})
    write_json(OUT_LATEST, latest)

    append_jsonl(OUT_HISTORY, {
        "generated_at_utc": latest["generated_at_utc"],
        "generated_at_ist": latest["generated_at_ist"],
        "risk_score": latest.get("risk_score", 0),
        "top_drivers": (latest.get("risk_drivers") or [])[:3],
    })

    print(f"OK: events_plotted={len(features)} nodes={len(nodes)} edges={len(edges)} wrote={OUT_LATEST},{OUT_GEOJSON},{OUT_GRAPH},{OUT_HISTORY}")

if __name__ == "__main__":
    main()
