import os
import subprocess
from pathlib import Path

EXPECTED = ["latest.json", "events.geojson", "graph.json", "history.jsonl"]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY")
    if not os.getenv("EXA_API_KEY"):
        raise SystemExit("Missing EXA_API_KEY")

    workspace = Path(os.getenv("WORKSPACE", ".")).resolve()

    # Use your workflow env if set; else a safe default
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    task = (
        "You are a geopolitical + markets tracking agent.\n"
        "Goal: Track Middle East events and market implications.\n\n"
        "Do the following:\n"
        "1) Collect recent, credible sources (last 72 hours and last 30 days as relevant).\n"
        "2) Extract a structured summary and a list of events.\n"
        "3) Write EXACTLY these files into the workspace root:\n"
        "   - latest.json\n"
        "   - events.geojson\n"
        "   - graph.json\n"
        "   - history.jsonl\n"
        "Rules:\n"
        "- ISO-8601 timestamps (or null).\n"
        "- location_name must be geocodable.\n"
        "- Always include source_urls for events.\n"
    )

    cmd = [
        "openplanter-agent",
        "--headless",
        "--provider", "openai",          # ✅ force OpenAI
        "--model", model,                # ✅ force your model (e.g., gpt-4o-mini)
        "--task", task,
        "--workspace", str(workspace),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    missing = [f for f in EXPECTED if not (workspace / f).exists()]
    if missing:
        raise SystemExit(f"OpenPlanter finished but missing outputs: {missing}")

    print("OK:", ", ".join(EXPECTED))

if __name__ == "__main__":
    main()
