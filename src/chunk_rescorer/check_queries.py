"""Verify assets/data/queries.json against the generated corpus.

Each symptom query's `entry` must be present in its target manual's Troubleshooting section,
every `unique_terms` item must appear in the target manual and in no other manual, and the
entry must be in the corpus's three-line format. Also reports how many query words the entry
shares (lexical chunk selection depends on this). Exit code 1 on any failure.

    uv run python src/chunk_rescorer/check_queries.py
"""

import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "assets" / "data"
STOP = {"the", "a", "an", "and", "but", "or", "my", "is", "are", "on", "of", "to", "in", "when", "from",
        "has", "no", "not", "i", "up", "any", "after", "at", "with", "that", "can", "which", "will"}


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", s.lower()) if w not in STOP}


def main() -> int:
    q = json.loads((DATA / "queries.json").read_text())
    manuals = {p.stem: p.read_text() for p in sorted((DATA / "manuals").glob("*.md"))}
    lower = {k: v.lower() for k, v in manuals.items()}
    ok = True
    print(f"{'query':24} {'target':9} {'in_TS':5} {'q∩entry':>7}  problems")
    for s in q["symptom_queries"]:
        problems = []
        lines = s["entry"].split("\n")
        if not (len(lines) == 3 and lines[0].startswith("Problem: ") and lines[1].startswith("Possible cause: ")
                and lines[2].startswith("Corrective action: ")):
            problems.append("bad entry format")
        text = manuals.get(s["target"], "")
        ts = text.split("\n## 6. Troubleshooting")[-1].split("\n## 7. ")[0] if text else ""
        in_ts = s["entry"] in ts
        if not in_ts:
            problems.append("entry missing from target §6")
        for t in s["unique_terms"]:
            hits = [m for m, txt in lower.items() if t.lower() in txt]
            if s["target"] not in hits:
                problems.append(f"'{t}' not in target")
            if others := [h for h in hits if h != s["target"]]:
                problems.append(f"'{t}' also in {others}")
        ew = words(s["entry"])
        qo = len(words(s["query"]) & ew)
        ok &= not problems
        print(f"{s['id']:24} {s['target']:9} {str(in_ts):5} {qo:>7}  {problems or '-'}")
    for s in q["spec_queries"]:
        if s["target"] not in manuals:
            ok = False
            print(f"{s['id']:24} {s['target']:9} target manual missing")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
