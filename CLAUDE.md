# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a demo/research project showing how Elasticsearch's `chunk_rescorer` on the `text_similarity_reranker` retriever improves relevance when using `jina-reranker-v3.5` via the Elastic Inference Service (EIS) on a Serverless project.

**Core thesis:** jina-reranker-v3.5 hard-caps each document at ~8,192 tokens (~6,000 words). `chunk_rescorer` splits documents and sends only the top-matching passage(s), allowing relevant content past the cut to be found and reducing latency/cost significantly.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management with Python 3.12.

```bash
# Install dependencies
uv sync

# Run the EIS probe script
uv run python src/chunk_rescorer/testeis.py

# Generate the synthetic manual corpus (one-time; needs an authenticated Gemini CLI; output is committed)
uv run python src/chunk_rescorer/generate_manuals.py --dry-run   # plan only
uv run python src/chunk_rescorer/generate_manuals.py             # ~175K output tokens, ~90 min
uv run python src/chunk_rescorer/generate_manuals.py --postprocess  # re-clean manuals on disk, rebuild manifest

# Run the demo notebook
uv run jupyter lab demo.ipynb

# Bring a generated image to an exact size (cover 1920x1080, section 900x300, or WxH)
assets/images/resize.sh cover  <in.png> assets/images/cover.png
assets/images/resize.sh section <in.png> assets/images/section3.png
```

The `.venv` is managed by uv automatically. Do not use `pip install` directly.

## Environment Variables

The notebook's Section 1 runs Terraform and writes `.env` (gitignored) with:
```bash
ELASTIC_CLOUD_ID=<cloud_id>            # terraform output elastic_cloud_id
ELASTIC_CLOUD_API_KEY=<cloud_api_key>  # the Cloud API key from terraform.tfvars; accepted by the project's ES endpoint
```
`testeis.py` has its own `ES_CLOUD_ID` / `ES_API_KEY` constants at the top of the file; fill them in before running it. `INFERENCE_ID` defaults to `.jina-reranker-v3.5`.

## Infrastructure (Terraform)

Provisions an Elastic Cloud Serverless Search project:

```bash
cd terraform
cp terraform.tfvars.sample terraform.tfvars
# Edit terraform.tfvars with your elastic_cloud_api_key
terraform init
terraform apply
terraform destroy  # teardown
```

Terraform manages only the Serverless project (`ec_elasticsearch_project`); it outputs the Cloud ID and echoes the Cloud API key. The notebook runs `terraform apply` in Section 1 and `destroy` in the last section via `%%bash` cells. All indices and queries are created by the notebook so readers see them happen.

## Code Architecture

```
src/chunk_rescorer/
  testeis.py           # Standalone probe: tests EIS truncation/cap/bias behavior
  generate_manuals.py  # One-time synthetic corpus generator (Gemini CLI); writes assets/data/manuals/
  check_queries.py     # Asserts each query's entry is in its target manual only; reports query overlap
  probe.py             # Notebook helper: filler docs with a marker sentence at a token offset (Section 3)
  corpus.py            # Notebook helper: loads manuals + manifest into index-ready docs, per-manual table, loads queries.json (Sections 4, 6, 8, 9)
  display.py           # Notebook helper: comparison readout with verdicts (5, 7), N-column results table (6, 8, 9), rejection readout (7)
  __init__.py          # Empty

assets/data/manuals/      # Generated corpus: <MODEL>.md per product + manifest.json (committed)
assets/data/queries.json  # Nine symptom queries (one per long manual, with the distinguishing entry), three spec queries
assets/article.md         # The LinkedIn article ("What Your Reranker Never Sees"); mirrors the notebook sections
assets/images/            # cover.png (1920x1080), arch.png (native size), section3-9.png (900x300); resize.sh
assets/prompts/           # gpt-image-2.5 prompts for every image, plus the presentation prompt
assets/plan.md            # Experiment design, established behaviour, deliverable status (see its §8 for departures)
README.md                 # Repo front page
index.html                # Ten-slide HTML presentation (self-contained; open in a browser; ?check runs a layout overflow test)
demo.ipynb                # Main demo notebook, 10 sections, complete and executed
terraform/                # EC Serverless project provisioning
```

**`testeis.py`** is a self-contained script that probes four behaviors of the EIS reranker:
- **Test 0 Sanity:** Basic rerank works, score range established
- **Test A Boundary:** Per-document token truncation point (cliff between 8,600–10,000 estimated tokens)
- **Test B Request cap:** Total payload limit (~200K tokens; ~16 docs × 10K tokens)
- **Test C Doc-count cap:** Tests the "64 documents per call" claim (no cap found up to 200)
- **Test D Position bias:** Checks if scores track list position vs. document content

## Notebook Structure

Ten sections, each a markdown cell (`## Section N - Title`, bulleted narrative, then **What the results show, and why**) followed by one code cell. The two configurations are called **Rerank** (whole documents) and **Chunk Rerank** (`chunk_rescorer`) everywhere; never B/C.

1. Elastic Serverless Provisioning (`%%bash` Terraform)
2. Connect to the Serverless project (`CONFIG` dict: `index`, `semantic_index`, `inference_id`, `rank_window_size`, `max_chunk_size`, `chunk_size`, paths)
3. The cap, measured (pairwise `es.inference.rerank`, control first; yes/no table)
4. Load and index the corpus (`manuals` index)
5. Two retriever configurations (one query through both, readout with verdicts)
6. Run the queries (nine symptom queries: Rerank 4 of 9, Chunk Rerank 9 of 9; ~1.8 s vs ~120 ms)
7. The request limit (window 12: Rerank rejected 400, Chunk Rerank ranks the catalog)
8. Does semantic_text avoid this? (`manuals_semantic` index: semantic-only 8 of 9, Rerank 4 of 9 with identical margins to Section 6, Chunk Rerank 9 of 9)
9. When the answer is already in view (spec queries: Rerank 3 of 3, Chunk Rerank 2 of 3, loses the camera query)
10. Takeaways and teardown (findings, recommended configuration, two ways to solve it, follow-ups; `%%bash` destroy)

Live results are stored in the notebook outputs. Ranks are stable across runs; margins move by a few hundredths.

## Notebook Cell Convention

Notebook code cells hold only the code that carries the concept and narrative of the section: the query, the marker or document being tested, the `es.*` call with its arguments, and the verdict. Plumbing (text generation, caching, file loading, table and chart rendering, bookkeeping) goes in a module under `src/chunk_rescorer/` and is imported into the cell. The package is installed editable by `uv sync`, so `from chunk_rescorer.<module> import ...` works in the kernel. Aim for a cell a reader can take in at a glance (roughly 30 lines or fewer).

## Key Technical Facts

- **Elasticsearch client only:** All ES interaction goes through the `elasticsearch` Python client (9.x). No `requests`, no raw HTTP.
- **Token/word ratio:** 1.326 tokens/word for English prose (Qwen3 tokenizer, consistent with Jina API)
- **Default chunking is dangerous:** Default `chunking_settings` for both `jinaai` and EIS services uses 7,000-word chunks (~9,300 tokens) — exceeding the per-document cap. Always specify explicit `chunking_settings`.
- **Never set `top_n` in task settings:** The coordinator expects one result per chunk.
- **Scores are list-dependent:** The same document varies 0.07–0.26 across requests depending on neighbors. Only within-request rank is meaningful; avoid `min_score`.
- **Chunk selection is lexical:** `chunk_rescorer` scores chunks with BM25 in an in-memory index on the shard, no inference call. A semantic first stage can find the document while the rescorer misses the passage when the query's words are not in it.
- **`semantic_text` does not exempt the reranker:** a cross-encoder reads text, not vectors. `text_similarity_reranker` on a `semantic_text` field fetches the whole original text and truncates exactly as on a `text` field (Section 8, measured). The Elastic docs do not state this either way.
- **Request limit:** Jina rejects requests whose estimated total exceeds 200,000 tokens (its estimate runs ~25% high). Twelve whole manuals (~228K estimated) are refused before inference.
- **Evidence rule:** support technical claims with Elastic documentation or a live test, not by reading Elasticsearch source code.

## Retriever Configuration Patterns

```python
INFERENCE_ID = ".jina-reranker-v3.5"

def first_stage(q, category):  # the three manuals of the category the customer is browsing
    return {"standard": {"query": {"bool": {
        "filter": [{"term": {"category": category}}],
        "must": [{"match": {"content": q}}]}}}}

def rerank(q, category, window=3):  # Rerank: whole documents to the reranker
    return {"text_similarity_reranker": {
        "retriever": first_stage(q, category),
        "field": "content", "inference_id": INFERENCE_ID, "inference_text": q,
        "rank_window_size": window,
    }}

def chunk_rerank(q, category, window=3, size=1, max_chunk_size=300):  # Chunk Rerank: best chunk only
    r = rerank(q, category, window)
    r["text_similarity_reranker"]["chunk_rescorer"] = {
        "size": size,
        "chunking_settings": {"strategy": "sentence", "max_chunk_size": max_chunk_size, "sentence_overlap": 1},
    }
    return r
```

## Dataset

Twelve synthetic hardware-store product manuals for an ecommerce troubleshooting use case: a customer types a symptom and expects the manual that fixes it. Three clusters of three same-brand models (pressure washers RG-2600P/RG-3200P/RG-4000P, garage door openers KG-800C/KG-1200B/KG-1500W, generators RG-4500E/RG-7500E/RG-10000D) plus three short manuals (drill, thermostat, hose reel) under the cap. Within a cluster, siblings share the head's safety section verbatim and adapt its other sections, so their first ~8K tokens read alike; each model's Troubleshooting section (starting ~75% in, past the 8,192-token cap) carries exactly one distinguishing fault from `assets/data/queries.json`. Generated once by `src/chunk_rescorer/generate_manuals.py` through the Gemini CLI (free) and committed under `assets/data/manuals/` with `manifest.json`; the notebook never calls an LLM API. Two indices: `manuals` (`content` as `text`) and, for Section 8, `manuals_semantic` (`content` as `semantic_text`, default endpoint). The first-stage query filters by `category`, so the rerank window is the three manuals of one category (W=3); Section 7 drops the filter for a window of 12. Without `chunk_rescorer` the reranker sees three near-identical heads; with it, the fix chunk. No nDCG, no shuffled-order averaging, no parameter sweeps. See `assets/plan.md` §4 and §8.

## Images and Prompts

- Every image is generated from a prompt under `assets/prompts/` with gpt-image-2.5, delivered as PNG, dark theme, **no logos or brand marks** (generators draw them badly).
- The generator does not honour requested sizes. Bring the LinkedIn cover to 1920×1080 and section diagrams to 900×300 with `assets/images/resize.sh` (`cover` / `section` presets). It fills and centre-crops when aspect ratios are close, pads otherwise, and flattens transparent inputs onto the dark background. The README architecture image is used at its native size and is **not** resized.
- Check the delivered image for text errors against the prompt's permitted-string list, and for a transparent or white background, before accepting it.
