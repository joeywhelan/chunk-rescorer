"""
Probe jina-reranker-v3.5 behavior on Elastic Inference Service (EIS), via the
Elasticsearch _inference API on a Serverless project.

Usage:
    export ES_CLOUD_ID="<deployment-or-project cloud id>"
    export ES_API_KEY=...
    export INFERENCE_ID=.jina-reranker-v3-5     # optional; script lists candidates if unset
    pip install elasticsearch
    python eis_truncation_probe.py > eis_results.txt

What it tests (EIS returns no token usage, so everything is inferred from scores/errors):
  0  Sanity      : 3 short docs; confirms the endpoint works and shows raw score range.
  A  Boundary    : ONE 40K-token doc with the needle at increasing token offsets, each
                   sent alongside an identical no-needle control. Where the needle stops
                   lifting the score is the per-document truncation point.
                   Jina API showed a cliff at 8,192; EIS may differ.
  B  Request cap : N x 10K-token filler docs, N rising until the request errors.
                   Jina API rejected > 200K raw tokens with a 422.
  C  Doc-count cap: N short (~60-token) docs, N in 32/64/65/100/200, needle doc LAST.
                   Checks the "up to 64 documents per call" statement for listwise models.
  D  Position bias: 12 distinct filler docs sent in forward and reversed order.
                   If scores track list position rather than the document, that's bias.
"""

import json
import random
import sys
import time

from elasticsearch import ApiError, Elasticsearch

ES_CLOUD_ID = "your_cloud_id_here"
ES_API_KEY = "your_api_key_here"
INFERENCE_ID = ".jina-reranker-v3.5"
if not ES_CLOUD_ID or not ES_API_KEY:
    sys.exit("Set ES_CLOUD_ID and ES_API_KEY")

es = Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY, request_timeout=300)

# Measured against Jina API for this filler: 1.326. Qwen3 tokenizer is the same model
# family on EIS, so this should hold; treat offsets as approximate +/- 3%.
TOKENS_PER_WORD = 1.326

QUERY = "What is the maintenance interval for the Kestrel-7 hydraulic actuator?"
NEEDLE = (
    "The Kestrel-7 hydraulic actuator requires a complete fluid replacement and seal "
    "inspection every 1,450 operating hours according to the manufacturer's service bulletin."
)

WORDS = (
    "the council approved quarterly budget for library renovation and park maintenance "
    "volunteers planted tulips near the fountain while the treasurer reconciled receipts "
    "from the bake sale and the choir rehearsed in the community hall on tuesday evening "
    "residents discussed recycling schedules street lighting and the annual harvest festival "
    "the committee reviewed minutes adopted the agenda and adjourned before nine o'clock "
    "gardeners recommended mulching roses in autumn and pruning apple trees in late winter "
    "the bookkeeper filed invoices sorted ledgers and archived the previous year's statements"
).split()


def filler_words(n_words: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    while len(out) < n_words:
        sent = rng.sample(WORDS, k=rng.randint(9, 16))
        sent[0] = sent[0].capitalize()
        out.extend(sent)
        out[-1] = out[-1] + "."
    return out[:n_words]


def tok2words(tokens: int) -> int:
    return int(tokens / TOKENS_PER_WORD)


def doc_with_needle_at(total_tokens: int, needle_at_tokens: int | None, seed: int) -> str:
    words = filler_words(tok2words(total_tokens), seed)
    if needle_at_tokens is None:
        return " ".join(words)
    pos = min(tok2words(needle_at_tokens), len(words))
    # Snap to the next sentence end so we don't split a sentence.
    while pos < len(words) and not words[pos - 1].endswith("."):
        pos += 1
    return " ".join(words[:pos] + NEEDLE.split() + words[pos:])


# ---------------------------------------------------------------------------

def list_rerank_endpoints() -> list[str]:
    resp = es.inference.get(task_type="rerank", inference_id="_all")
    return [e["inference_id"] for e in resp.get("endpoints", [])]


def rerank(docs: list[str]) -> dict:
    t0 = time.time()
    try:
        resp = es.inference.rerank(inference_id=INFERENCE_ID, query=QUERY, input=docs)
    except ApiError as e:
        return {"error": e.status_code, "body": json.dumps(e.body)[:800], "latency": time.time() - t0}
    except Exception as e:  # transport / timeout
        return {"error": type(e).__name__, "body": str(e)[:800], "latency": time.time() - t0}
    dt = time.time() - t0
    scores = {x["index"]: x["relevance_score"] for x in resp["rerank"]}
    return {"scores": scores, "latency": dt}


def header(name: str, docs: list[str], note: str = ""):
    words = sum(len(d.split()) for d in docs)
    print(f"\n=== {name} ===  {len(docs)} docs, {words:,} words (~{int(words*TOKENS_PER_WORD):,} tokens est.)")
    if note:
        print("    " + note)


def show(res: dict, tags: list[str]):
    if "error" in res:
        print(f"    ERROR {res['error']} after {res['latency']:.1f}s: {res['body']}")
        return None
    print(f"    latency={res['latency']:.1f}s")
    for i, tag in enumerate(tags):
        print(f"    [{i:3d}] {tag:<36} score={res['scores'].get(i, float('nan')):+.4f}")
    return res["scores"]


# ---------------------------------------------------------------------------

def test_sanity():
    docs = [
        "Puddles form on sidewalks after light rain.",
        NEEDLE,
        "The choir rehearsed in the community hall on Tuesday evening.",
    ]
    header("0 Sanity", docs, "Middle doc is the needle; expect it to win clearly.")
    show(rerank(docs), ["unrelated", "NEEDLE alone", "unrelated"])


def test_boundary():
    total = 40_000
    offsets = [1_000, 4_000, 7_000, 7_800, 8_600, 10_000, 14_000, 24_000, 38_000]
    control = doc_with_needle_at(total, None, seed=11)
    print(f"\n### A Boundary sweep: one {total:,}-token doc, needle at increasing offsets, vs identical control")
    print("    Δ = needle score − control score. Δ ≈ 0 means the needle was cut off.")
    results = []
    for off in offsets:
        docs = [doc_with_needle_at(total, off, seed=11), control]
        res = rerank(docs)
        if "error" in res:
            print(f"    offset {off:>6,}: ERROR {res['error']}: {res['body'][:200]}")
            results.append((off, None))
            continue
        s_needle, s_ctrl = res["scores"][0], res["scores"][1]
        print(f"    offset {off:>6,}: needle={s_needle:+.4f} control={s_ctrl:+.4f}  Δ={s_needle - s_ctrl:+.4f}")
        results.append((off, s_needle - s_ctrl))
    return results


def test_request_cap():
    print("\n### B Request-size cap: N x 10K-token filler docs, needle doc FIRST")
    for n in (8, 12, 16, 20, 24, 32):
        docs = [doc_with_needle_at(10_000, 500, seed=100)] + [
            doc_with_needle_at(10_000, None, seed=100 + i) for i in range(1, n)
        ]
        header(f"B n={n}", docs)
        res = rerank(docs)
        if "error" in res:
            print(f"    ERROR {res['error']} after {res['latency']:.1f}s: {res['body'][:400]}")
            print("    Stopping the sweep at first error.")
            return n
        s = res["scores"]
        top = max(s, key=s.get)
        print(f"    latency={res['latency']:.1f}s  needle(doc 0)={s[0]:+.4f}  top index={top}  "
              f"max filler={max(v for k, v in s.items() if k != 0):+.4f}")
    return None


def test_doc_count_cap():
    print("\n### C Document-count cap: N short docs, needle doc LAST")
    for n in (32, 64, 65, 100, 200):
        docs = [" ".join(filler_words(45, seed=1000 + i)) for i in range(n - 1)] + [NEEDLE]
        header(f"C n={n}", docs)
        res = rerank(docs)
        if "error" in res:
            print(f"    ERROR {res['error']} after {res['latency']:.1f}s: {res['body'][:400]}")
            continue
        s = res["scores"]
        top = max(s, key=s.get)
        print(f"    latency={res['latency']:.1f}s  returned={len(s)}  needle(idx {n-1})={s.get(n-1, float('nan')):+.4f}  "
              f"top index={top}  max filler={max(v for k, v in s.items() if k != n-1):+.4f}")


def test_position_bias():
    print("\n### D Position bias: 12 distinct 2K-token filler docs, forward then reversed")
    seeds = list(range(500, 512))
    docs = [doc_with_needle_at(2_000, None, seed=s) for s in seeds]
    fwd = rerank(docs)
    rev = rerank(list(reversed(docs)))
    if "error" in fwd or "error" in rev:
        print("    ERROR", fwd.get("body") or rev.get("body"))
        return
    print(f"    {'seed':>6} {'fwd pos':>8} {'fwd score':>10} {'rev pos':>8} {'rev score':>10}")
    for i, s in enumerate(seeds):
        j = len(seeds) - 1 - i
        print(f"    {s:>6} {i:>8} {fwd['scores'][i]:>+10.4f} {j:>8} {rev['scores'][j]:>+10.4f}")
    print("    If a seed's score follows its position (changes a lot between fwd/rev) -> position bias.")
    print("    If it follows the seed (roughly stable) -> the model is scoring content.")


def main():
    global INFERENCE_ID
    info = es.info()
    print(f"Connected: {info['name']}  version={info['version']['number']}")
    eps = list_rerank_endpoints()
    print("Rerank endpoints on this project:", eps)
    if not INFERENCE_ID:
        cands = [e for e in eps if "3-5" in e or "3.5" in e or "v35" in e]
        if len(cands) == 1:
            INFERENCE_ID = cands[0]
        else:
            sys.exit(f"Set INFERENCE_ID to one of: {eps}")
    print(f"Using INFERENCE_ID={INFERENCE_ID}")

    test_sanity()
    test_boundary()
    test_request_cap()
    test_doc_count_cap()
    test_position_bias()


if __name__ == "__main__":
    main()