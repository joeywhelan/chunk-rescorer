"""Printing helpers for notebook search results."""

import re
import textwrap

NOISE = 0.05  # the same document moves about this much between requests; smaller gaps are noise


def verdict(resp: dict, expected: str) -> str:
    """One sentence: did the correct manual win, and by how much?"""
    hits = resp["hits"]["hits"]
    ids = [h["_id"] for h in hits]
    scores = [h["_score"] for h in hits]
    if expected not in ids:
        return f"correct manual ({expected}) not in the results"
    rank = ids.index(expected) + 1
    if rank == 1:
        gap = scores[0] - scores[1]
        how = "a near tie: the reranker could not really tell them apart" if gap < NOISE else "a clear call"
        return f"correct manual first, {gap:.2f} ahead of #2 -> {how}"
    gap = scores[0] - scores[rank - 1]
    return f"correct manual only #{rank}, {gap:.2f} behind the winner -> wrong answer"


def show_comparison(query: str, expected: str, runs: list[tuple[str, str, dict]]) -> None:
    """Side-by-side readout of several configurations on one query.

    ``runs`` is a list of (label, what the reranker saw, search response).
    """
    print(f'Query:    "{query}"')
    print(f"Answer:   {expected} is the manual whose troubleshooting section has this fault\n")
    for label, saw, resp in runs:
        print(f"{label}")
        print(f"   reranker saw: {saw}")
        print(f"   took:         {resp['took']:,} ms")
        for rank, h in enumerate(resp["hits"]["hits"], start=1):
            mark = "  <- correct" if h["_id"] == expected else ""
            category = h.get("_source", {}).get("category", "")
            print(f"   #{rank:<3} {h['_id']:10} {category:22} score {h['_score']:.3f}{mark}")
        print(f"   verdict:      {verdict(resp, expected)}\n")
    print("Scores are only comparable within one block, not across blocks. Read the rank and the gap.")


def outcome(resp: dict, expected: str) -> dict:
    """Rank of the correct manual, its margin over the best other manual, and timing."""
    hits = resp["hits"]["hits"]
    ids = [h["_id"] for h in hits]
    scores = [h["_score"] for h in hits]
    rank = ids.index(expected) + 1 if expected in ids else None
    others = [s for i, s in zip(ids, scores) if i != expected]
    margin = scores[rank - 1] - max(others) if rank and others else None
    return {"rank": rank, "margin": margin, "took_ms": resp["took"]}


def show_query_table(rows: list[dict], configs: tuple[str, ...] = ("Rerank", "Chunk Rerank")) -> None:
    """One line per query: where each configuration put the correct manual.

    Each row needs ``id``, ``target`` and an ``outcome`` dict per configuration under
    the keys named in ``configs``. Margin is the correct manual's score minus the best
    other score: positive means it won by that much, negative that it lost by that much.
    """
    def cell(o: dict) -> str:
        if o["rank"] is None:
            return f"{'missing':>7} {'':>7} {o['took_ms']:>6,}"
        return f"{'#' + str(o['rank']):>7} {o['margin']:>+7.2f} {o['took_ms']:>6,}"

    print(f"{'query':24} {'answer':9}" + "".join(f" | {c:^22}" for c in configs))
    print(f"{'':34}" + f" | {'rank':>7} {'margin':>7} {'ms':>6}" * len(configs))
    print("-" * (34 + 25 * len(configs)))
    for r in rows:
        print(f"{r['id']:24} {r['target']:9}" + "".join(f" | {cell(r[c])}" for c in configs))
    print()
    for cfg in configs:
        first = sum(1 for r in rows if r[cfg]["rank"] == 1)
        close = sum(1 for r in rows if r[cfg]["rank"] == 1 and r[cfg]["margin"] < NOISE)
        note = f" ({close} of them by less than {NOISE:.2f}, i.e. a near tie)" if close else ""
        took = sorted(r[cfg]["took_ms"] for r in rows)
        print(f"{cfg}: correct manual first in {first} of {len(rows)}{note}; median Elasticsearch time {took[len(took) // 2]:,} ms")


def _service_message(err: dict) -> str:
    """The inference service's own message, dug out of Elasticsearch's nested error.

    Follows the caused_by / suppressed chain to the innermost reason, then keeps only
    the part the service returned (inside "Error message: [...]"), minus its validator prefix.
    """
    while True:
        nested = err.get("caused_by") or (err.get("suppressed") or [None])[0]
        if not nested:
            break
        err = nested
    reason = err.get("reason", "")
    m = re.search(r"Error message: \[(.*)\]$", reason)
    message = m.group(1) if m else reason
    return re.sub(r"^Validation error: '[^']*'\s*", "", message)


def show_rejection(label: str, saw: str, error) -> None:
    """Readout for a configuration whose request the inference service refused.

    ``error`` is the ``ApiError`` raised by the client; the service's message is printed
    as returned, wrapped to the readout's width.
    """
    print(f"{label}")
    print(f"   reranker saw: {saw}")
    print(f"   status:       {error.status_code} {type(error).__name__}")
    print(textwrap.fill(_service_message(error.body["error"]), width=88,
                        initial_indent="   reason:       ", subsequent_indent=" " * 17))
    print("   verdict:      no ranking at all -> this configuration cannot rerank a window this large\n")
