"""Load the synthetic manual corpus under assets/data/manuals/ for the notebook.

Each manual is a Markdown file plus an entry in manifest.json written by
generate_manuals.py. Token figures come from the manifest's measured ratio of
1.326 tokens per word (Qwen3 tokenizer), the model family jina-reranker-v3.5 uses.
"""

import json
from pathlib import Path

FIELDS = ("model", "brand", "category", "name")


def load_manuals(manuals_dir: str | Path) -> list[dict]:
    """One document per manual, in manifest (cluster) order, ready to index."""
    manuals_dir = Path(manuals_dir)
    manifest = json.loads((manuals_dir / "manifest.json").read_text())
    docs = []
    for m in manifest["manuals"]:
        doc = {k: m[k] for k in FIELDS}
        doc["content"] = (manuals_dir / m["file"]).read_text()
        doc["token_count"] = m["est_tokens"]
        doc["troubleshooting_start_tokens"] = m["troubleshooting_start_est_tokens"]
        docs.append(doc)
    return docs


def describe(docs: list[dict], cap_tokens: int = 8192) -> None:
    """Per-manual table: where the troubleshooting section sits relative to the cap."""
    print(f"{'model':10} {'category':22} {'tokens':>7} {'troubleshooting at':>19}  past the cap?")
    for d in docs:
        past = "yes" if d["troubleshooting_start_tokens"] > cap_tokens else "no"
        print(f"{d['model']:10} {d['category']:22} {d['token_count']:>7,} {d['troubleshooting_start_tokens']:>19,}  {past}")
    total = sum(d["token_count"] for d in docs)
    print(f"\n{len(docs)} manuals, ~{total:,} tokens in total")


def load_queries(queries_file: str | Path, kind: str) -> list[dict]:
    """The evaluation queries of one kind: "symptom" (answer in the troubleshooting
    section, past the cap) or "spec" (answer in the specifications, in view).

    Each row has ``id``, ``category``, ``target`` (the model whose manual answers it)
    and ``query``.
    """
    q = json.loads(Path(queries_file).read_text())
    return [
        {"id": s["id"], "category": s["cluster"], "target": s["target"], "query": s["query"]}
        for s in q[f"{kind}_queries"]
    ]
