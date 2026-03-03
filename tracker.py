import os
import sys
import subprocess
from pathlib import Path

# Files we expect OpenPlanter to write into the workspace (repo root)
EXPECTED = ["latest.json", "events.geojson", "graph.json", "history.jsonl"]

def main():
    # Basic env checks (OpenPlanter supports providers; you are using OpenAI+Exa in your workflow)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY")
    if not os.getenv("EXA_API_KEY"):
        raise SystemExit("Missing EXA_API_KEY")

    workspace = Path(os.getenv("WORKSPACE", ".")).resolve()

    task = (
        "You are a geopolitical + markets tracking agent.\n"
        "Goal: Track Middle East events and market implications.\n\n"
        "Do the following:\n"
        "1) Collect recent, credible sources (last 72 hours and last 30 days as relevant), prioritize primary/reputable outlets.\n"
        "2) Extract a structured summary and a list of events.\n"
        "3) Write EXACTLY these files into the workspace root:\n"
        "   - latest.json  (headline_summary<=10, risk_score 0-100, risk_drivers 3-8, market_implications by asset class, sources list)\n"
        "   - events.geojson (FeatureCollection of Point events with properties: id,event_type,severity,confidence,title,timestamp_utc,location_name,actors,implication,source_urls)\n"
        "   - graph.json (nodes+edges capturing causal links; nodes must include all edge endpoints)\n"
        "   - history.jsonl (append one line with generated_at_utc, generated_at_ist, risk_score, top_drivers)\n\n"
        "Rules:\n"
        "- Use ISO-8601 for timestamps. If unknown, set timestamp_utc null.\n"
        "- location_name must be geocodable (city/port/strait/country). If unsure, use country/strait.\n"
        "- If you cannot confidently produce a field, keep it empty/null but DO NOT crash.\n"
        "- Always include source_urls for events.\n"
    )

    # Run OpenPlanter headlessly
    cmd = [
        "openplanter-agent",
        "--task", task,
        "--workspace", str(workspace),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Validate outputs exist
    missing = [f for f in EXPECTED if not (workspace / f).exists()]
    if missing:
        raise SystemExit(f"OpenPlanter finished but missing outputs: {missing}")

    print("OK: OpenPlanter wrote:", ", ".join(EXPECTED))


if __name__ == "__main__":
    main()
