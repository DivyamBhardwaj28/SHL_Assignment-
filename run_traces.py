"""
Replays markdown conversation traces (like C1.md) against your local
POST /chat endpoint, and computes Recall@10 against each trace's
expected final shortlist.

Usage:
    1. Put all trace .md files (C1.md, C2.md, ...) in a folder, e.g. ./traces/
    2. Make sure your server is running: uvicorn main:app --reload
    3. Run: python run_traces.py ./traces

Note: this replays the SCRIPTED user lines from each trace verbatim.
The real SHL evaluator drives the user side with its own LLM reacting
to your agent's actual replies, so this is an approximation -- useful
for fast local iteration, not a perfect stand-in for the real harness.
"""

import sys
import re
import json
import time
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/chat"
TIMEOUT_S = 30
SECONDS_BETWEEN_REQUESTS = 2  # paid tier has a much higher rate limit than free;
                               # small delay still avoids bursting into it


def parse_trace(path: Path):
    """Extract the scripted user turns and the final expected shortlist
    (the markdown table in the last turn) from a trace file."""
    text = path.read_text(encoding="utf-8")

    # Split into turn blocks
    turn_blocks = re.split(r"### Turn \d+", text)[1:]

    user_messages = []
    final_recommendations = []

    for block in turn_blocks:
        # Extract the user line: "**User**\n\n> ..."
        user_match = re.search(r"\*\*User\*\*\s*\n+>\s*(.+)", block)
        if user_match:
            user_messages.append(user_match.group(1).strip())

        # Extract any markdown table rows in this block.
        # Header looks like: | # | Name | Test Type | Keys | Duration | Languages | URL |
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            # Skip header row and separator row (---- ----)
            if cells[0] in ("#", "") or set(cells[0]) <= {"-"}:
                continue
            if not cells[0].isdigit():
                continue
            name = cells[1]
            url_match = re.search(r"https?://\S+", line)
            url = url_match.group(0).strip("<>") if url_match else None
            rows.append({"name": name, "url": url})

        if rows:
            # Each new table overwrites; the LAST table found in the file
            # (the one at end_of_conversation: true) becomes the expected
            # final shortlist.
            final_recommendations = rows

    return user_messages, final_recommendations


def run_trace(user_messages):
    """Replay scripted user turns against the live API, building real
    conversation history from the agent's own responses as we go."""
    history = []
    last_response = None

    for msg in user_messages:
        history.append({"role": "user", "content": msg})
        
        resp = requests.post(API_URL, json={"messages": history}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        history.append({"role": "assistant", "content": data["reply"]})
        last_response = data
        time.sleep(SECONDS_BETWEEN_REQUESTS)  # avoid free-tier rate limit (5 req/min)

    return last_response


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def recall_at_10(expected, actual_recommendations):
    if not expected:
        return None  # nothing to score against

    expected_names = {normalize(e["name"]) for e in expected}
    actual_names = {normalize(r["name"]) for r in (actual_recommendations or [])[:10]}

    hits = expected_names & actual_names
    return len(hits) / len(expected_names), hits, expected_names - hits


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_traces.py <folder_with_trace_md_files>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    trace_files = sorted(folder.glob("*.md"))

    if not trace_files:
        print(f"No .md files found in {folder}")
        sys.exit(1)

    recalls = []

    for trace_path in trace_files:
        print(f"\n=== {trace_path.name} ===")
        user_messages, expected = parse_trace(trace_path)

        if not user_messages:
            print("  Could not parse any user turns, skipping.")
            continue
        if not expected:
            print("  Could not parse an expected shortlist, skipping.")
            continue

        try:
            result = run_trace(user_messages)
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            continue

        score, hits, misses = recall_at_10(expected, result.get("recommendations"))
        recalls.append(score)

        print(f"  Expected ({len(expected)}): {[e['name'] for e in expected]}")
        print(f"  Got ({len(result.get('recommendations', []))}): "
              f"{[r['name'] for r in result.get('recommendations', [])]}")
        print(f"  Recall@10: {score:.2f}")
        if misses:
            missed_names = [e["name"] for e in expected if normalize(e["name"]) in misses]
            print(f"  Missed: {missed_names}")
        print(f"  end_of_conversation: {result.get('end_of_conversation')}")

    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\n=== Mean Recall@10 across {len(recalls)} traces: {mean_recall:.3f} ===")
    else:
        print("\nNo traces scored.")


if __name__ == "__main__":
    main()
