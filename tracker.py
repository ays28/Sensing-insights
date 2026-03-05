import os
import json
import time
import hashlib
import math
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ==========================
# REQUIRED SECRETS
# ==========================
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================
# CONFIG
# ==========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "25"))

# Output files (repo root)
OUT_EXTRACT = "op_extract.json"
OUT_LATEST = "latest.json"
OUT_GEOJSON = "events.geojson"
OUT_GRAPH = "graph.json"
OUT_HISTORY = "history.jsonl"
OUT_GEOCODE_CACHE = "geocode_cache.json"

# Time zones
IST = timezone(timedelta(hours=5, minutes=30))

# Nominatim (polite usage)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "SensingInsightsTracker/1.0 (github-actions)")
NOMINATIM_MIN_DELAY_SEC = float(os.getenv("NOMINATIM_MIN_DELAY_SEC", "1.2"))

# Accumulation caps (prevent infinite growth)
MAX_ACCUMULATED_EVENTS = int(os.getenv("MAX_ACCUMULATED_EVENTS", "1500"))
MAX_GRAPH_EVENT_NODES = int(os.getenv("MAX_GRAPH_EVENT_NODES", "250"))
MAX_GRAPH_EDGES = int(os.getenv("MAX_GRAPH_EDGES", "1200"))

# Deterministic map jitter for stacked markers (degrees; ~0.02 ≈ a couple km)
JITTER_DEG = float(os.getenv("JITTER_DEG", "0.02"))

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
    # safe fallback so events never disappear
    "middle east": (45.0, 29.5),
}

MARKET_NODES = [
    {"id": "oil_up_risk", "label": "Oil up-risk", "group": "market"},
    {"id": "em_fx_pressure", "label": "EM FX pressure", "group": "market"},
    {"id": "risk_off_flows", "label": "Risk-off flows", "group": "market"},
    {"id": "rates_credit_spreads", "label": "Rates/Credit spreads", "group": "market"},
]


# ==========================
# TIME + ID
# ==========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ist_now_iso() -> str:
    return datetime.now(IST).isoformat()


def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ==========================
# FILE I/O
# ==========================
def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# ==========================
# GEO FALLBACKS + GEOCODER
# ==========================
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

    def save(self) -> None:
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

        # throttle
        now = time.time()
        wait = NOMINATIM_MIN_DELAY_SEC - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)

        try:
            r = requests.get(
                NOMINATIM_URL,
                params={"q": place, "format": "json", "limit": 1, "accept-language": "en"},
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


# ==========================
# GRAPH HELPERS
# ==========================
def normalize_links(raw_links: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_links, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw_links:
        if isinstance(item, dict):
            frm = item.get("from")
            to = item.get("to")
            if frm and to:
                out.append(item)
    return out


def ensure_graph_nodes(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    id_set: Set[str] = set(n.get("id") for n in nodes if n.get("id"))
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm and frm not in id_set:
            nodes.append({"id": frm, "label": str(frm).replace("_", " "), "group": "market"})
            id_set.add(frm)
        if to and to not in id_set:
            nodes.append({"id": to, "label": str(to).replace("_", " "), "group": "market"})
            id_set.add(to)
    return nodes


def load_existing_features() -> List[Dict[str, Any]]:
    existing = safe_read_json(OUT_GEOJSON, {})
    if isinstance(existing, dict) and existing.get("type") == "FeatureCollection":
        feats = existing.get("features")
        if isinstance(feats, list):
            return [f for f in feats if isinstance(f, dict)]
    return []


def feature_ts(f: Dict[str, Any]) -> str:
    p = f.get("properties") or {}
    return str(p.get("timestamp_utc") or "")


def merge_features(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def get_id(feat: Dict[str, Any]) -> str:
        p = feat.get("properties") or {}
        return str(p.get("id") or "")

    for f in existing:
        fid = get_id(f)
        if fid:
            merged[fid] = f

    for f in new:
        fid = get_id(f)
        if not fid:
            continue
        if fid not in merged:
            merged[fid] = f
        else:
            if feature_ts(f) >= feature_ts(merged[fid]):
                merged[fid] = f

    out = list(merged.values())
    out.sort(key=feature_ts, reverse=True)
    return out[:MAX_ACCUMULATED_EVENTS]


def build_event_nodes_from_features(features: List[Dict[str, Any]], cap: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    sorted_feats = sorted(features, key=feature_ts, reverse=True)
    nodes: List[Dict[str, Any]] = []
    title_to_id: Dict[str, str] = {}

    for f in sorted_feats[:cap]:
        p = f.get("properties") or {}
        eid = p.get("id")
        title = p.get("title") or ""
        if not eid:
            continue

        if title and title not in title_to_id:
            title_to_id[title] = eid

        nodes.append({
            "id": eid,
            "label": (title[:80] if title else str(eid)),
            "group": "event",
            "severity": int(p.get("severity", 0) or 0),
            "confidence": p.get("confidence", "low"),
            "event_type": p.get("event_type", "other"),
            "timestamp_utc": p.get("timestamp_utc"),
            "location_name": p.get("location_name", ""),
        })

    return nodes, title_to_id


def load_existing_graph() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    g = safe_read_json(OUT_GRAPH, {})
    if not isinstance(g, dict):
        return [], []
    nodes = g.get("nodes") if isinstance(g.get("nodes"), list) else []
    edges = g.get("edges") if isinstance(g.get("edges"), list) else []
    nodes = [n for n in nodes if isinstance(n, dict) and n.get("id")]
    edges = [e for e in edges if isinstance(e, dict) and e.get("from") and e.get("to")]
    return nodes, edges


def merge_edges(existing: List[Dict[str, Any]], new: List[Dict[str, Any]], node_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    Accumulate graph edges across runs.
    Key = (from,to,label). Keep latest 'why/confidence' when duplicate occurs.
    Filter to edges whose endpoints exist in node_ids (or are market ids).
    """
    def edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
        return (str(e.get("from")), str(e.get("to")), str(e.get("label") or ""))

    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def accept(e: Dict[str, Any]) -> bool:
        frm = str(e.get("from") or "")
        to = str(e.get("to") or "")
        if not frm or not to:
            return False
        # allow market concept ids always (they're in MARKET_NODES anyway)
        return (frm in node_ids or any(frm == m["id"] for m in MARKET_NODES)) and (to in node_ids or any(to == m["id"] for m in MARKET_NODES))

    for e in existing:
        if accept(e):
            merged[edge_key(e)] = e

    for e in new:
        if accept(e):
            merged[edge_key(e)] = e

    out = list(merged.values())
    # keep deterministic ordering (optional) — prefer edges with higher "confidence" lexical
    out.sort(key=lambda x: (str(x.get("confidence") or ""), str(x.get("label") or "")), reverse=True)
    return out[:MAX_GRAPH_EDGES]


# ==========================
# LATEST ROBUSTNESS
# ==========================
def is_latest_weak(latest: Dict[str, Any]) -> bool:
    if not isinstance(latest, dict):
        return True
    risk = int(latest.get("risk_score", 0) or 0)
    headlines = latest.get("headline_summary") or []
    sources = latest.get("sources") or []
    # weak if it basically contains nothing
    return (risk == 0) and (len(headlines) == 0) and (len(sources) == 0)


def make_fallback_latest_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If agent returns weak latest, create a minimal usable latest from events.
    Keeps UI populated without changing schema.
    """
    # pick top events by severity
    evs = [e for e in events if isinstance(e, dict)]
    evs.sort(key=lambda x: int(x.get("severity", 0) or 0), reverse=True)

    headlines: List[str] = []
    sources: List[Dict[str, Any]] = []
    drivers: List[str] = []

    for e in evs[:10]:
        title = str(e.get("title") or "").strip()
        sev = int(e.get("severity", 0) or 0)
        et = str(e.get("event_type") or "").strip()
        loc = str(e.get("location_name") or "").strip()
        if title:
            headlines.append(f"{title} ({et}, sev {sev}) — {loc}".strip(" —"))

        urls = e.get("source_urls") or []
        if isinstance(urls, list) and urls:
            u0 = str(urls[0] or "")
            if u0:
                sources.append({"title": title or u0, "url": u0, "publisher": "", "published_at": ""})

        if et and et not in drivers:
            drivers.append(et)

    # crude risk estimate
    top_sev = int(evs[0].get("severity", 0) or 0) if evs else 0
    risk_score = max(0, min(100, top_sev))

    return {
        "headline_summary": headlines[:10],
        "risk_score": risk_score,
        "risk_drivers": drivers[:8],
        "market_implications": {"Energy": [], "FX": [], "Rates": [], "Equities": []},
        "sources": sources[:12],
        "generated_at_utc": utc_now_iso(),
        "generated_at_ist": ist_now_iso(),
    }


# ==========================
# MAP VISIBILITY: DETERMINISTIC JITTER FOR STACKED POINTS
# ==========================
def deterministic_jitter(lon: float, lat: float, key: str, scale: float) -> Tuple[float, float]:
    """
    Spread points with identical coordinates so 144 doesn't look like 20.
    Deterministic based on event id key.
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    a = int(h[:8], 16) / 0xFFFFFFFF  # 0..1
    b = int(h[8:16], 16) / 0xFFFFFFFF
    # angle + radius
    angle = 2 * math.pi * a
    radius = scale * (0.2 + 0.8 * b)
    dlon = radius * math.cos(angle)
    dlat = radius * math.sin(angle)
    return lon + dlon, lat + dlat


def apply_jitter_to_stacked_features(features: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for f in features:
        g = f.get("geometry") or {}
        coords = g.get("coordinates") or None
        if not (isinstance(coords, list) and len(coords) == 2):
            continue
        lon, lat = float(coords[0]), float(coords[1])
        buckets.setdefault((lon, lat), []).append(f)

    for (lon, lat), group in buckets.items():
        if len(group) <= 1:
            continue
        for f in group:
            p = f.get("properties") or {}
            eid = str(p.get("id") or "")
            if not eid:
                continue
            jlon, jlat = deterministic_jitter(lon, lat, eid, scale)
            f["geometry"]["coordinates"] = [jlon, jlat]

    return features


# ==========================
# OPENPLANTER RUNNER
# ==========================
def run_openplanter(workspace: str) -> None:
    task = f"""
You are an OSINT + markets tracking agent.

Goal: Track Middle East events related to US, Israel & Iran war, and market implications, and infer relationships between events. Do not miss any event.

TIME / RECENCY:
- Prefer events from the last 24–72 hours.
- You may include a few older (up to 7 days) items ONLY if they are strategically critical context.
- If timestamp is unknown, set timestamp_utc to null (do not invent).

You MUST write EXACTLY one JSON file to the workspace root:
- {OUT_EXTRACT}

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

LOCATION RULES:
- location_name MUST be geocodable and specific.
- Prefer "City, Country". If not possible, use "Country" or "Strait/Sea name, near Country".
- Avoid vague regions.

- actors (array of strings)
- implication (1-2 lines)
- source_urls (array of urls)

EVENT QUALITY RULES:
- Avoid duplicates/near-duplicates. Merge multi-report into one event with multiple source_urls.

links array: infer relationships between events AND event->market concepts.
Each link is an object with:
- from (string: an event title OR a market concept id)
- to (string: an event title OR a market concept id)
- relation (string)
- confidence (low|med|high)
- why (short justification)

Market concept ids you may use:
- oil_up_risk
- em_fx_pressure
- risk_off_flows
- rates_credit_spreads

Rules:
- Use ONLY credible sources you can cite in source_urls.
- Produce at least 54 events; no filler.
- links must be objects only (no strings).
""".strip()

    cmd = [
        "openplanter-agent",
        "--headless",
        "--provider", "openai",
        "--model", OPENAI_MODEL,
        "--reasoning-effort", "none",
        "--task", task,
        "--workspace", workspace,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ==========================
# MAIN
# ==========================
def main() -> None:
    if not EXA_API_KEY:
        raise SystemExit("Missing EXA_API_KEY")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY")

    workspace = os.path.abspath(".")
    run_openplanter(workspace)

    if not os.path.exists(OUT_EXTRACT):
        raise SystemExit(f"OpenPlanter did not write {OUT_EXTRACT} to repo root.")

    with open(OUT_EXTRACT, "r", encoding="utf-8") as f:
        out = json.load(f)

    latest = out.get("latest") if isinstance(out.get("latest"), dict) else {}
    events = out.get("events") if isinstance(out.get("events"), list) else []
    links = normalize_links(out.get("links"))

    # Load previous latest to avoid "empty right panel" on weak runs
    prev_latest = safe_read_json(OUT_LATEST, {})
    if is_latest_weak(latest):
        # try fallback from events; if even that is empty, keep previous
        fallback_latest = make_fallback_latest_from_events([e for e in events if isinstance(e, dict)])
        if not is_latest_weak(fallback_latest):
            latest = fallback_latest
        elif isinstance(prev_latest, dict) and prev_latest:
            latest = prev_latest

    # Ensure timestamps used by UI exist
    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    latest["generated_at_ist"] = latest.get("generated_at_ist") or ist_now_iso()

    geocoder = Geocoder(OUT_GEOCODE_CACHE)

    # Build new features from THIS run
    new_features: List[Dict[str, Any]] = []

    dropped_missing_loc = 0
    hard_fallback_used = 0

    for e in events:
        if not isinstance(e, dict):
            continue

        title = (e.get("title") or "Event").strip()
        loc = (e.get("location_name") or "").strip()
        etype = (e.get("event_type") or "other").strip()
        ts = e.get("timestamp_utc") or ""

        # If location missing, try actor-based centroid keyword
        if not loc:
            actors = e.get("actors") or []
            for a in actors:
                fb = centroid_fallback(str(a))
                if fb:
                    loc = str(a)
                    break

        if not title or not loc:
            dropped_missing_loc += 1
            continue

        src_urls = e.get("source_urls") or []
        src0 = str(src_urls[0] or "") if isinstance(src_urls, list) and src_urls else ""
        eid = stable_id(title, loc, ts, etype, src0)

        geo = geocoder.geocode(loc)
        if geo is None:
            geo = centroid_fallback(loc)
        if geo is None:
            for a in (e.get("actors") or []):
                fb = centroid_fallback(str(a))
                if fb:
                    geo = fb
                    break
        if geo is None:
            geo = centroid_fallback("middle east")
            hard_fallback_used += 1

        lon, lat, disp = geo

        props = {
            "id": eid,
            "event_type": etype,
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

        new_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    geocoder.save()

    # Accumulate events.geojson
    existing_features = load_existing_features()
    merged_features = merge_features(existing_features, new_features)

    # Jitter stacked markers so "144 events" are actually visible
    merged_features = apply_jitter_to_stacked_features(merged_features, JITTER_DEG)

    # ==========================
    # ACCUMULATED GRAPH BEHAVIOR
    # ==========================
    # Build nodes from accumulated events, not only this run
    event_nodes, title_to_id = build_event_nodes_from_features(merged_features, cap=MAX_GRAPH_EVENT_NODES)
    node_ids = set(n["id"] for n in event_nodes) | set(m["id"] for m in MARKET_NODES)

    # Build new edges from current links, mapping event titles -> ids using accumulated mapping
    new_edges: List[Dict[str, Any]] = []
    for l in links:
        frm_raw = str(l.get("from") or "").strip()
        to_raw = str(l.get("to") or "").strip()
        if not frm_raw or not to_raw:
            continue
        frm = title_to_id.get(frm_raw, frm_raw)
        to = title_to_id.get(to_raw, to_raw)
        new_edges.append({
            "from": frm,
            "to": to,
            "label": str(l.get("relation") or "causes"),
            "confidence": str(l.get("confidence") or "low"),
            "why": str(l.get("why") or ""),
        })

    # Merge edges with previous graph.json (accumulation)
    prev_nodes, prev_edges = load_existing_graph()
    # We rebuild nodes from accumulated features, but we DO accumulate edges:
    merged_edges = merge_edges(prev_edges, new_edges, node_ids=node_ids)

    nodes: List[Dict[str, Any]] = MARKET_NODES + event_nodes
    nodes = ensure_graph_nodes(nodes, merged_edges)

    # Write outputs (schemas unchanged, index.html unchanged)
    write_json(OUT_GEOJSON, {"type": "FeatureCollection", "features": merged_features})
    write_json(OUT_GRAPH, {"nodes": nodes, "edges": merged_edges})
    write_json(OUT_LATEST, latest)

    append_jsonl(OUT_HISTORY, {
        "generated_at_utc": latest["generated_at_utc"],
        "generated_at_ist": latest["generated_at_ist"],
        "risk_score": latest.get("risk_score", 0),
        "top_drivers": (latest.get("risk_drivers") or [])[:3],
    })

    print(
        "OK:",
        f"events_extracted={len(events)}",
        f"events_new_plotted={len(new_features)}",
        f"events_total_accumulated={len(merged_features)}",
        f"graph_nodes={len(nodes)}",
        f"graph_edges={len(merged_edges)}",
        f"dropped_missing_loc={dropped_missing_loc}",
        f"hard_fallback_used={hard_fallback_used}",
    )


if __name__ == "__main__":
    main()
