import os
import json
import time
import math
import hashlib
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
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT", "SensingInsightsTracker/1.0 (github-actions)"
)
NOMINATIM_MIN_DELAY_SEC = float(os.getenv("NOMINATIM_MIN_DELAY_SEC", "1.2"))

# Accumulation caps (prevent infinite growth)
MAX_ACCUMULATED_EVENTS = int(os.getenv("MAX_ACCUMULATED_EVENTS", "2000"))

# Graph controls
MAX_GRAPH_EVENT_NODES = int(os.getenv("MAX_GRAPH_EVENT_NODES", "300"))  # nodes shown
MAX_GRAPH_EDGES = int(os.getenv("MAX_GRAPH_EDGES", "1600"))  # edges stored
EMERGENT_MAX_PAIR_WINDOW = int(os.getenv("EMERGENT_MAX_PAIR_WINDOW", "120"))  # comparisons per node
EMERGENT_MAX_EDGES = int(os.getenv("EMERGENT_MAX_EDGES", "700"))  # emergent edges per run
EMERGENT_MIN_SCORE = float(os.getenv("EMERGENT_MIN_SCORE", "4.0"))  # edge score threshold

# Deterministic map jitter for stacked markers (degrees; ~0.02 ≈ a few km)
JITTER_DEG = float(os.getenv("JITTER_DEG", "0.02"))

# Some actor stopwords / junk tokens
ACTOR_STOP = {
    "", "unknown", "unspecified", "n/a", "na", "none",
    "government", "military", "forces", "officials",
    "police", "army", "navy", "air force",
}

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
# TIME + ID HELPERS
# ==========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ist_now_iso() -> str:
    return datetime.now(IST).isoformat()


def stable_id(*parts: str) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def parse_ts(ts: Any) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


# ==========================
# FILE HELPERS
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
# GEO HELPERS
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
# GRAPH / ACCUMULATION HELPERS
# ==========================
def normalize_links(raw_links: Any) -> List[Dict[str, Any]]:
    # Keep dict links only; ignore strings/mixed.
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


def feature_ts_str(f: Dict[str, Any]) -> str:
    p = f.get("properties") or {}
    return str(p.get("timestamp_utc") or "")


def merge_features(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup by properties.id. Keep newest timestamp_utc if duplicate.
    """
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
            if feature_ts_str(f) >= feature_ts_str(merged[fid]):
                merged[fid] = f

    out = list(merged.values())
    out.sort(key=feature_ts_str, reverse=True)
    return out[:MAX_ACCUMULATED_EVENTS]


def load_existing_graph() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    g = safe_read_json(OUT_GRAPH, {})
    if not isinstance(g, dict):
        return [], []
    nodes = g.get("nodes") if isinstance(g.get("nodes"), list) else []
    edges = g.get("edges") if isinstance(g.get("edges"), list) else []
    nodes = [n for n in nodes if isinstance(n, dict) and n.get("id")]
    edges = [e for e in edges if isinstance(e, dict) and e.get("from") and e.get("to")]
    return nodes, edges


def merge_edges(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Accumulate edges across runs (dedup by from,to,label).
    """
    def ekey(e: Dict[str, Any]) -> Tuple[str, str, str]:
        return (str(e.get("from") or ""), str(e.get("to") or ""), str(e.get("label") or ""))

    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in existing:
        k = ekey(e)
        if k[0] and k[1]:
            merged[k] = e
    for e in new:
        k = ekey(e)
        if k[0] and k[1]:
            merged[k] = e

    out = list(merged.values())
    # Keep consistent ordering (not required, but helpful)
    out.sort(key=lambda x: (str(x.get("confidence") or ""), str(x.get("label") or "")), reverse=True)
    return out[:MAX_GRAPH_EDGES]


def build_event_nodes_from_features(features: List[Dict[str, Any]], cap: int) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build graph nodes from accumulated geojson features.
    Returns (nodes, title_to_id) for mapping OpenPlanter links by title.
    """
    sorted_feats = sorted(features, key=feature_ts_str, reverse=True)
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


# ==========================
# LATEST ROBUSTNESS (RIGHT PANEL)
# ==========================
def is_latest_weak(latest: Dict[str, Any]) -> bool:
    if not isinstance(latest, dict):
        return True
    risk = int(latest.get("risk_score", 0) or 0)
    headlines = latest.get("headline_summary") or []
    sources = latest.get("sources") or []
    return (risk == 0) and (len(headlines) == 0) and (len(sources) == 0)


def make_fallback_latest_from_features(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If OpenPlanter returns weak latest, build a usable summary from accumulated features.
    Keeps UI populated with no schema changes.
    """
    feats = [f for f in features if isinstance(f, dict)]
    feats.sort(key=lambda f: int((f.get("properties") or {}).get("severity", 0) or 0), reverse=True)

    headlines: List[str] = []
    sources: List[Dict[str, Any]] = []
    drivers: List[str] = []

    for f in feats[:10]:
        p = f.get("properties") or {}
        title = str(p.get("title") or "").strip()
        sev = int(p.get("severity", 0) or 0)
        et = str(p.get("event_type") or "").strip()
        loc = str(p.get("location_name") or "").strip()
        if title:
            headlines.append(f"{title} ({et}, sev {sev}) — {loc}".strip(" —"))

        urls = p.get("source_urls") or []
        if isinstance(urls, list) and urls:
            u0 = str(urls[0] or "")
            if u0:
                sources.append({"title": title or u0, "url": u0, "publisher": "", "published_at": ""})

        if et and et not in drivers:
            drivers.append(et)

    top_sev = int(((feats[0].get("properties") or {}).get("severity", 0) or 0)) if feats else 0
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
# MAP VISIBILITY: DETERMINISTIC JITTER
# ==========================
def deterministic_jitter(lon: float, lat: float, key: str, scale: float) -> Tuple[float, float]:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    a = int(h[:8], 16) / 0xFFFFFFFF
    b = int(h[8:16], 16) / 0xFFFFFFFF
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
# EMERGENT EDGES (EVENTS ACKNOWLEDGE EACH OTHER)
# ==========================
def km_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def conf_from_score(score: float) -> str:
    if score >= 6.5:
        return "high"
    if score >= 5.0:
        return "med"
    return "low"


def norm_actor(a: str) -> str:
    a = (a or "").strip().lower()
    a = a.replace(".", "").replace(",", "")
    return a


def extract_country_hint(loc: str) -> str:
    """
    Cheap heuristic: if known keys appear in location string, return that key.
    """
    if not loc:
        return ""
    s = loc.lower()
    for k in CENTROIDS.keys():
        if k in s:
            return k
    return ""


def build_emergent_edges_from_features(features: List[Dict[str, Any]], id_allow: Set[str]) -> List[Dict[str, Any]]:
    """
    Create relationship edges from:
      - shared actors
      - geo proximity
      - time proximity
      - same country hint
      - escalation-ish patterns
    Direction: older -> newer when timestamps exist.
    """
    recs = []
    for f in features:
        if not isinstance(f, dict):
            continue
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        coords = g.get("coordinates") or None
        if not (isinstance(coords, list) and len(coords) == 2):
            continue

        eid = str(p.get("id") or "")
        if not eid or eid not in id_allow:
            continue

        actors_raw = p.get("actors") or []
        actors = []
        if isinstance(actors_raw, list):
            for a in actors_raw:
                aa = norm_actor(str(a))
                if aa and aa not in ACTOR_STOP and len(aa) <= 60:
                    actors.append(aa)

        loc_name = str(p.get("location_name") or "")
        recs.append({
            "id": eid,
            "title": str(p.get("title") or ""),
            "etype": str(p.get("event_type") or ""),
            "sev": int(p.get("severity", 0) or 0),
            "ts": parse_ts(p.get("timestamp_utc")),
            "loc": loc_name.lower(),
            "country_hint": extract_country_hint(loc_name),
            "actors": actors,
            "lon": float(coords[0]),
            "lat": float(coords[1]),
        })

    # sort newest first for efficient pair window
    recs.sort(key=lambda r: (r["ts"] is not None, r["ts"]), reverse=True)

    edges_scored: List[Tuple[float, Dict[str, Any]]] = []
    N = len(recs)

    for i in range(N):
        a = recs[i]
        # compare only against next window to keep fast
        for j in range(i + 1, min(N, i + 1 + EMERGENT_MAX_PAIR_WINDOW)):
            b = recs[j]

            score = 0.0
            why_bits: List[str] = []

            # shared actors
            if a["actors"] and b["actors"]:
                shared = set(a["actors"]).intersection(set(b["actors"]))
                if shared:
                    score += 3.0
                    ex = list(shared)[:2]
                    why_bits.append(f"Shared actor: {', '.join(ex)}")

            # same event type
            if a["etype"] and a["etype"] == b["etype"]:
                score += 1.0
                why_bits.append(f"Same type: {a['etype']}")

            # country hint overlap
            if a["country_hint"] and a["country_hint"] == b["country_hint"]:
                score += 1.5
                why_bits.append(f"Same area: {a['country_hint']}")

            # time proximity
            if a["ts"] and b["ts"]:
                hours = abs((a["ts"] - b["ts"]).total_seconds()) / 3600.0
                if hours <= 72:
                    score += 2.0
                    why_bits.append("Within 72h")
                elif hours <= 168:
                    score += 1.0
                    why_bits.append("Within 7d")

            # geo proximity
            dist = km_distance(a["lon"], a["lat"], b["lon"], b["lat"])
            if dist <= 250:
                score += 2.0
                why_bits.append(f"Nearby (~{int(dist)}km)")
            elif dist <= 500:
                score += 1.0
                why_bits.append(f"Regionally close (~{int(dist)}km)")

            # escalation-ish priors
            # strike/cyber/protest nearby in time → could be response chain
            if a["etype"] in {"strike", "cyber", "protest"} and b["etype"] in {"strike", "cyber", "protest"}:
                if a["ts"] and b["ts"]:
                    hours = abs((a["ts"] - b["ts"]).total_seconds()) / 3600.0
                    if hours <= 72:
                        score += 0.8
                        why_bits.append("Escalation pattern")

            # severity bonus
            score += (min(a["sev"], b["sev"]) / 60.0)

            if score < EMERGENT_MIN_SCORE:
                continue

            # direction: older -> newer when possible
            frm_id, to_id = b["id"], a["id"]  # recs sorted newest first; b likely older
            label = "related_to"
            # strengthen label for some patterns
            if "Shared actor" in " ".join(why_bits):
                label = "actor_link"
            if "Escalation pattern" in " ".join(why_bits):
                label = "possible_retaliation"

            edges_scored.append((
                score,
                {
                    "from": frm_id,
                    "to": to_id,
                    "label": label,
                    "confidence": conf_from_score(score),
                    "why": "; ".join(why_bits)[:240],
                }
            ))

    edges_scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in edges_scored[:EMERGENT_MAX_EDGES]]


def build_market_edges_from_features(features: List[Dict[str, Any]], id_allow: Set[str]) -> List[Dict[str, Any]]:
    """
    Deterministic edges from events to market concept nodes.
    Keeps the graph meaningful even if OpenPlanter links are weak.
    """
    out: List[Dict[str, Any]] = []
    for f in features[: max(600, MAX_GRAPH_EVENT_NODES * 3)]:
        p = f.get("properties") or {}
        eid = str(p.get("id") or "")
        if not eid or eid not in id_allow:
            continue
        et = str(p.get("event_type") or "")
        sev = int(p.get("severity", 0) or 0)

        def c(sev_: int) -> str:
            return "high" if sev_ >= 75 else ("med" if sev_ >= 50 else "low")

        if et == "shipping_disruption":
            out.append({"from": eid, "to": "oil_up_risk", "label": "drives", "confidence": c(sev), "why": "Shipping disruption increases energy risk"})
            out.append({"from": eid, "to": "risk_off_flows", "label": "contributes_to", "confidence": "med", "why": "Heightened geopolitical risk can drive risk-off flows"})
        elif et == "sanctions":
            out.append({"from": eid, "to": "em_fx_pressure", "label": "drives", "confidence": c(sev), "why": "Sanctions can pressure EM FX via trade/flows expectations"})
            out.append({"from": eid, "to": "risk_off_flows", "label": "contributes_to", "confidence": "med", "why": "Sanctions uncertainty can add to risk-off sentiment"})
        elif et == "strike":
            out.append({"from": eid, "to": "risk_off_flows", "label": "contributes_to", "confidence": c(sev), "why": "Kinetic escalation can trigger risk-off positioning"})
            if sev >= 60:
                out.append({"from": eid, "to": "oil_up_risk", "label": "raises", "confidence": "med", "why": "Escalation can raise energy supply risk premium"})
        elif et == "diplomacy":
            # diplomacy often reduces risk
            out.append({"from": eid, "to": "risk_off_flows", "label": "reduces", "confidence": c(100 - sev), "why": "Diplomacy can reduce near-term risk-off pressure"})
        elif et == "cyber":
            out.append({"from": eid, "to": "rates_credit_spreads", "label": "contributes_to", "confidence": c(sev), "why": "Cyber incidents can widen credit risk perceptions"})
        # other: no forced edges

    return out


# ==========================
# OPENPLANTER RUNNER
# ==========================
def run_openplanter(workspace: str) -> None:
    """
    Calls OpenPlanter to produce ./op_extract.json in repo root.
    """
    task = f"""
You are an OSINT + markets tracking agent.

Goal: Track Middle East events and market implications, and infer relationships between events.

TIME / RECENCY:
- Prefer events from the last 24–72 hours.
- Include up to 7 days only if strategically critical context.
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
- actors (array of strings)
- implication (1-2 lines)
- source_urls (array of urls)

LOCATION RULES (VERY IMPORTANT):
- location_name MUST be geocodable and specific.
- Prefer "City, Country". If not possible, use "Country" or "Strait/Sea name, near Country".
- Avoid vague regions like "border area", "the region".

EVENT QUALITY:
- Avoid duplicates/near-duplicates. If multiple sources report the same story, merge into ONE event with multiple source_urls.

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
- Produce 30–60 events focused on recency and relevance (no filler).
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

    raw_latest = out.get("latest") if isinstance(out.get("latest"), dict) else {}
    raw_events = out.get("events") if isinstance(out.get("events"), list) else []
    raw_links = normalize_links(out.get("links"))

    # --------------------------
    # Geocode THIS RUN events -> features
    # --------------------------
    geocoder = Geocoder(OUT_GEOCODE_CACHE)

    new_features: List[Dict[str, Any]] = []
    title_to_id_run: Dict[str, str] = {}

    dropped_missing_loc = 0
    hard_fallback_used = 0

    for e in raw_events:
        if not isinstance(e, dict):
            continue

        title = (e.get("title") or "Event").strip()
        loc = (e.get("location_name") or "").strip()
        etype = (e.get("event_type") or "other").strip()
        ts = e.get("timestamp_utc") or ""

        # Try actors as fallback location hint
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

        # Stabilize id using first source URL too (helps dedupe across runs)
        src_urls = e.get("source_urls") or []
        src0 = str(src_urls[0] or "") if isinstance(src_urls, list) and src_urls else ""
        eid = stable_id(title, loc, ts, etype, src0)
        title_to_id_run[title] = eid

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

    # --------------------------
    # Accumulate events.geojson
    # --------------------------
    existing_features = load_existing_features()
    merged_features = merge_features(existing_features, new_features)

    # Make stacked markers visible
    merged_features = apply_jitter_to_stacked_features(merged_features, JITTER_DEG)

    # --------------------------
    # Latest.json (right panel)
    # --------------------------
    prev_latest = safe_read_json(OUT_LATEST, {})
    latest = raw_latest if isinstance(raw_latest, dict) else {}

    if is_latest_weak(latest):
        # fallback from accumulated features
        fb_latest = make_fallback_latest_from_features(merged_features)
        if not is_latest_weak(fb_latest):
            latest = fb_latest
        elif isinstance(prev_latest, dict) and prev_latest:
            latest = prev_latest

    latest["generated_at_utc"] = latest.get("generated_at_utc") or utc_now_iso()
    latest["generated_at_ist"] = latest.get("generated_at_ist") or ist_now_iso()

    # --------------------------
    # Graph: build nodes from accumulated store
    # --------------------------
    event_nodes, title_to_id_accum = build_event_nodes_from_features(merged_features, cap=MAX_GRAPH_EVENT_NODES)
    nodes: List[Dict[str, Any]] = MARKET_NODES + event_nodes
    node_ids: Set[str] = set(n["id"] for n in nodes if n.get("id"))

    # --------------------------
    # Edges: 3 sources
    #   1) OpenPlanter links (this run)
    #   2) Emergent edges from accumulated store
    #   3) Deterministic market edges
    # And we accumulate edges across runs.
    # --------------------------
    new_edges: List[Dict[str, Any]] = []

    # 1) OpenPlanter links: map titles -> ids using accumulated mapping (best)
    for l in raw_links:
        frm_raw = str(l.get("from") or "").strip()
        to_raw = str(l.get("to") or "").strip()
        if not frm_raw or not to_raw:
            continue

        frm = title_to_id_accum.get(frm_raw, title_to_id_run.get(frm_raw, frm_raw))
        to = title_to_id_accum.get(to_raw, title_to_id_run.get(to_raw, to_raw))

        new_edges.append({
            "from": frm,
            "to": to,
            "label": str(l.get("relation") or "causes"),
            "confidence": str(l.get("confidence") or "low"),
            "why": str(l.get("why") or ""),
        })

    # 2) Emergent edges: events acknowledge each other
    # only allow event ids in current node set (so graph remains readable)
    event_id_allow = set(n["id"] for n in event_nodes)
    emergent_edges = build_emergent_edges_from_features(merged_features, id_allow=event_id_allow)
    new_edges.extend(emergent_edges)

    # 3) Market edges: always meaningful
    new_edges.extend(build_market_edges_from_features(merged_features, id_allow=event_id_allow))

    # Accumulate edges across runs
    _, prev_edges = load_existing_graph()
    merged_edges = merge_edges(prev_edges, new_edges)

    # Make sure graph endpoints exist
    nodes = ensure_graph_nodes(nodes, merged_edges)

    # --------------------------
    # Write outputs consumed by index.html
    # --------------------------
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
        f"events_extracted={len(raw_events)}",
        f"events_new_plotted={len(new_features)}",
        f"events_total_accumulated={len(merged_features)}",
        f"graph_nodes={len(nodes)}",
        f"graph_edges={len(merged_edges)}",
        f"dropped_missing_loc={dropped_missing_loc}",
        f"hard_fallback_used={hard_fallback_used}",
        f"emergent_edges_added={len(emergent_edges)}",
    )


if __name__ == "__main__":
    main()
