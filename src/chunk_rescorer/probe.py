"""Synthetic probe documents for measuring the reranker's per-document token cap.

A probe document is a fixed run of unrelated filler prose with, optionally, one
marker sentence inserted at a chosen token offset. Two probe documents built from
the same marker differ only in where (or whether) the marker appears.
"""

import random

# Measured for this filler with the Qwen3 tokenizer (the family jina-reranker-v3.5
# is built on). Offsets derived from it are accurate to within a few percent.
TOKENS_PER_WORD = 1.326

_FILLER_WORDS = (
    "the council approved quarterly budget for library renovation and park maintenance "
    "volunteers planted tulips near the fountain while the treasurer reconciled receipts "
    "from the bake sale and the choir rehearsed in the community hall on tuesday evening "
    "residents discussed recycling schedules street lighting and the annual harvest festival "
    "the committee reviewed minutes adopted the agenda and adjourned before nine o'clock "
    "gardeners recommended mulching roses in autumn and pruning apple trees in late winter "
    "the bookkeeper filed invoices sorted ledgers and archived the previous year's statements"
).split()


def tokens_to_words(tokens: int) -> int:
    return int(tokens / TOKENS_PER_WORD)


def filler_words(n_words: int, seed: int = 11) -> list[str]:
    """Deterministic unrelated prose, built from short sentences."""
    rng = random.Random(seed)
    words: list[str] = []
    while len(words) < n_words:
        sentence = rng.sample(_FILLER_WORDS, k=rng.randint(9, 16))
        sentence[0] = sentence[0].capitalize()
        sentence[-1] += "."
        words.extend(sentence)
    return words[:n_words]


def probe_doc(marker: str, marker_at_tokens: int | None, total_tokens: int, seed: int = 11) -> str:
    """Filler of ``total_tokens`` with ``marker`` inserted at ``marker_at_tokens``.

    ``marker_at_tokens=None`` returns the marker-free control. The insertion point
    snaps forward to the next sentence boundary so no filler sentence is split.
    """
    words = filler_words(tokens_to_words(total_tokens), seed)
    if marker_at_tokens is None:
        return " ".join(words)
    pos = min(tokens_to_words(marker_at_tokens), len(words))
    while pos < len(words) and not words[pos - 1].endswith("."):
        pos += 1
    return " ".join(words[:pos] + marker.split() + words[pos:])
