# Demo Plan: `chunk_rescorer` + jina-reranker-v3.5 on Elastic Serverless

**Goal:** A reproducible demo showing where `chunk_rescorer` on the `text_similarity_reranker` retriever measurably improves relevance with jina-reranker-v3.5 on EIS, why it works, and where it doesn't.

**Scenario:** A hardware retailer publishes the owner's manuals for the products it sells and exposes them through a support search on its ecommerce site. A customer types a symptom ("pressure washer pulses and loses pressure", "generator runs a few seconds then dies") and expects the manual, and ideally the passage, that fixes it. The fix is almost always in the troubleshooting table at the back of the manual.

**Scope:** Infrastructure, data, experiment design, and deliverable structure. The article (`assets/article.md`), README, images and prompts are tracked in §8.

**Status (2026-09-13):** The notebook is complete and executed end to end against a live Serverless project. Sections 8 to 10 differ from the original plan; see §8 for what changed and why.

---

## 1. Thesis

Through EIS (and the Jina API), jina-reranker-v3.5 hard-caps every document at ~8K tokens and discards the rest. Anything past that point — roughly 6,000 words or 25 pages — is invisible to the reranker. A whole rank window is also sent in a single request, which fails outright when the raw payload exceeds Jina's 200K-token estimate.

`chunk_rescorer` splits each document, picks the best-matching passage(s) lexically, and sends only those. The relevant passage survives regardless of where it sits in the document, the request stays small, and latency drops with it.

**Trade-off:** v3.5 is listwise and benefits from context; chunking removes intra-document context. For documents that already fit under the cap, chunking should be neutral or slightly negative. The demo shows both regimes.

| Regime | Without chunk_rescorer | With chunk_rescorer |
|---|---|---|
| Docs under the cap | Best (or equal) | Equal or slightly worse |
| Docs over the cap, answer in the visible head | Fine | Equal |
| Docs over the cap, answer past the cut | **Fails** — answer never seen | **Works** |
| Window × raw doc size > 200K est. tokens | **400 error** | Works |

---

## 2. Established behavior

Established from live probes of `.jina-reranker-v3.5` on a Serverless 9.6.0 project and from the Elastic documentation. Rule for this project: support claims with Elastic docs or live tests, not by reading Elasticsearch source code.

**Elasticsearch: `chunk_rescorer` on a `text` field**
- Shard side: the field value is chunked with the configured chunker; each chunk is scored against `inference_text` in an in-memory Lucene index (lexical, no inference call); the top `size` chunks become the document's feature data.
- Coordinator: all chunks from all docs in the window are flattened into one rerank request. Each chunk is scored as its own document. A document's score is the **max** over its chunks.
- Default `chunking_settings` (when omitted) for both the `jinaai` and EIS services: sentence-boundary, **7,000 words**, no overlap. At ~1.33 tokens/word that is ~9,300 tokens — over the per-document cap, so the default chunk is itself truncated. **Explicit `chunking_settings` are mandatory.**
- No client-side truncation. Scores are normalized `max(s,0) + min(exp(s),1)`.
- **`semantic_text` does not exempt the reranker (measured, notebook Section 8).** With the manuals indexed as `semantic_text`, a `semantic` query alone ranks by best chunk and finds the buried entry (8 of 9). `text_similarity_reranker` on that field still fetches the whole original text and reproduces the plain-text Rerank result exactly (4 of 9, identical margins); `chunk_rescorer` restores 9 of 9. The Elastic docs do not state what text the retriever sends for a `semantic_text` field; this is established empirically only.

**EIS `.jina-reranker-v3.5`**
- Preconfigured on Serverless alongside `.jina-reranker-v3`, `.jina-reranker-v2-base-multilingual`, `.jina-reranker-m0`, `.rerank-v1-elasticsearch`.
- Per-document cut between 8,600 and 10,000 estimated tokens (8,192 real, by Jina API billing). Past the cut, needle and control docs score identically to four decimals: content is dropped, not sampled.
- Request limit: Jina's "estimated total tokens exceed the maximum limit of 200,000", returned as an ES 400 `status_exception`. Trips at ~16 × 10K-token docs (~160K real tokens); the estimator runs ~25% high. Practical ceiling: ~12 docs of 10K tokens or ~5–6 of 20K per request.
- No document-count cap: 200 short docs in one call rank correctly.
- Latency ~1.1s per 10K-token document unchunked (7s for 8 docs, 14s for 12).
- Scores are list-dependent: the same doc varies 0.07–0.26 across requests depending on neighbors, and shifts ±0.03–0.05 with list order. No directional position bias observed. Only within-request rank is meaningful; `min_score` is unreliable with this model.
- Recency effect: a needle just before the cut lifts the score more (Δ 0.36) than one at the start (Δ 0.11).
- Measured tokens/word on English prose: 1.326 (Qwen3 tokenizer).

---

## 3. Environment

### 3.1 Elastic Serverless project (Terraform)
- `terraform/elastic_serverless.tf`: an `ec_elasticsearch_project` (Serverless, `optimized_for = "general_purpose"`, region variable defaulting to `gcp-us-central1`) via the `elastic/ec` provider `~> 0.13`. That is the only provider; the unused `elasticstack` provider was removed on 2026-09-13.
- Outputs: `elastic_cloud_id` and `elastic_cloud_api_key` (the Cloud API key passed in as a variable, marked sensitive). There is no Elasticsearch-scoped API key resource; the notebook authenticates with the Cloud API key. Verified: the Cloud API key is accepted by the project's Elasticsearch endpoint (notebook Section 2).
- The notebook, not a Makefile, runs Terraform: Section 1 does `init` and `apply` in a `%%bash` cell and writes `.env` with `ELASTIC_CLOUD_ID` and `ELASTIC_CLOUD_API_KEY`; the last section runs `destroy` and removes `.env`. `terraform.tfvars` holds the Cloud API key and is gitignored along with `.env`, state, and the lock file.
- Nothing inside Elasticsearch is Terraform-managed; indices and queries are created in the notebook so readers see it happen.

### 3.2 Inference endpoint
Use the preconfigured `.jina-reranker-v3.5`. No key, no setup, and it's what Serverless readers have.

Fallback for deployments without EIS:
```python
es.inference.put(
    task_type="rerank",
    inference_id="jina-rerank-v35",
    inference_config={
        "service": "jinaai",
        "service_settings": {"api_key": JINA_API_KEY, "model_id": "jina-reranker-v3.5"},
        "task_settings": {"return_documents": False},
    },
)
```
Never set `top_n` in task settings; the coordinator expects one result per chunk.

### 3.3 Tooling
- **All Elasticsearch interaction goes through the official Python client (`elasticsearch` 9.x).** No raw REST calls, no `requests`, no Dev Console snippets in the notebook. Connection is `Elasticsearch(cloud_id=..., api_key=...)`.
- Client methods used: `es.info`, `es.inference.get/put/rerank`, `es.indices.create/delete/exists/refresh`, `elasticsearch.helpers.bulk`, `es.search(retriever=...)`. Errors are handled via `elasticsearch.ApiError` (status code and body).
- One Jupyter notebook orchestrates everything, including Terraform apply and destroy via `%%bash` cells. Credentials are read from `.env` (written by Section 1) with `python-dotenv`. Structure in §7.
- EIS returns no token usage; token figures in the notebook are the manifest's estimates at 1.326 tokens/word, and cost is shown as payload size plus measured Elasticsearch `took`.
- Corpus generation is a separate one-time script (`src/chunk_rescorer/generate_manuals.py`) that drives the Gemini CLI headlessly (`gemini -p`), so it costs nothing beyond an authenticated CLI. Its output under `assets/data/manuals/` is committed, as is the query set `assets/data/queries.json`. Nothing in the notebook depends on any LLM API.

---

## 4. Dataset and experiment design

### 4.1 Corpus: three clusters of three manuals, plus three short ones

**Scope decision.** The point is a controlled comparison: the same customer question, the same three candidate manuals, with and without `chunk_rescorer`. That needs a handful of long documents, not a catalog. Twelve manuals demonstrate the mechanism, the cost and latency gap, the request-limit failure, and the under-the-cap regime. What they cannot show is aggregate relevance lift on a real catalog; the article states the mechanism and leaves that as a follow-up.

**Why synthetic.** Real manuals are copyrighted, so they can't be committed and every reader would download them at runtime from links that rot. They extract noisily from PDF and are often trilingual. Synthetic manuals modeled on real structure are committed to the repo, so the demo is self-contained and reproducible, and every fault a query asks about is one we wrote. The trade is credibility: the article says the manuals are modeled on real ones and argues that the cliff is a property of the reranker's token cap, not of the text. Section 3's direct `es.inference.rerank` probe makes that argument independent of the corpus.

**Why clusters.** A hardware store sells several models per category, and a customer's symptom query names the category ("pressure washer pump drips..."). With one manual per category, the first 8K tokens of the right manual already say "pressure washer" and the reranker ranks it first without ever seeing the fix; `chunk_rescorer` would have nothing to fix. Three same-brand models per category, sharing the same safety boilerplate and near-identical specifications and operation sections, make the troubleshooting entry the only thing that separates them. That is also the realistic case: the fix that applies to one model and not its siblings is exactly what support search has to find.

- **Generation.** `src/chunk_rescorer/generate_manuals.py` writes each manual section by section through the Gemini CLI (free, already authenticated locally). The first model of a cluster is the head. Siblings reuse the head's safety section word for word and adapt its other sections, keeping structure and shared wording. Each model's Troubleshooting section includes exactly one distinguishing fault taken from `assets/data/queries.json` and is told to omit its siblings' faults. A post-processing pass (rerunnable with `--postprocess`) strips bold, removes entries that leaked into other sections, guarantees the model's entry is present, and rebuilds the manifest from the files on disk. Run once; commit the output under `assets/data/manuals/`.
- **Clusters** (brand, models, exact Qwen3 token counts of the generated manuals; troubleshooting start in parentheses):
  - Pressure washers, Ridgeline Outdoor Power: RG-2600P 20.7K (16.5K), RG-3200P (head) 24.1K (19.7K), RG-4000P 16.8K (12.4K). Cluster total 61.6K tokens.
  - Garage door openers, Kestrel Home Systems: KG-800C 26.9K (20.7K), KG-1200B (head) 35.4K (26.3K), KG-1500W 31.9K (26.1K). Cluster total 94.2K.
  - Portable generators, Ridgeline Outdoor Power: RG-4500E 21.3K (16.5K), RG-7500E (head) 19.5K (14.5K), RG-10000D 22.8K (18.1K). Cluster total 63.5K.
  - Siblings share 59% to 84% of their first 8,192 tokens line for line with their head (safety verbatim; specifications and operation adapted).
- **Three short manuals**, all under the cap: Northfield cordless drill 5.0K tokens, Kestrel thermostat 6.3K, Northfield hose reel 3.0K. They show the under-the-cap regime in the corpus table (Section 4); no query targets them.
- Troubleshooting begins at roughly 75% of each manual, so in every long manual it sits past the 8,192-token cap (earliest at 12.4K tokens, the RG-4000P).
- Per cluster, the three manuals total 62K to 94K real tokens, under the ~160K request limit, so the unchunked baseline runs within a cluster. The whole corpus in one window is ~228K estimated tokens and does not fit, which is how the 400 is shown (§4.4).
- Store `content`, `category`, `brand`, `model`, `name`, `token_count`, `troubleshooting_start_tokens` per document. All token figures are the manifest's estimates (1.326 tokens/word); the notebook does not tokenize.
- **Two indices.** `manuals` (`content` as `text`) carries Sections 4 to 7 and 9. `manuals_semantic` (`content` as `semantic_text`, default endpoint, which resolves to `.jina-embeddings-v5-text-small` on this project) carries Section 8. The first-stage query filters on `category` and matches the symptom text, so the rank window is the three manuals of the category the customer is looking at (W=3). Every rank reported is the reranker's ordering of those three.

### 4.2 Queries (`assets/data/queries.json`)
- **Nine symptom queries**, one per long manual, each answered by exactly one model's troubleshooting entry. Unchunked, the reranker sees three near-identical manual heads and the correct manual lands at rank one about a third of the time. With `chunk_rescorer`, the chunk carrying the fix is what gets sent, and the correct manual ranks first.
- **Three spec queries** ("which pressure washer is rated 4,000 PSI") answered near the front of a manual, inside the visible range. Used in Section 9: Rerank ranks all three correctly; Chunk Rerank loses the camera query, which combines a feature only one manual has with one all three share.
- Customer-worded variants were written and tested but cut from the notebook and removed from the data file (see §8).
- `src/chunk_rescorer/check_queries.py` asserts each entry is present in its target's Troubleshooting section and that its distinguishing terms appear in no other manual.

### 4.3 Retriever configurations

Two configurations carry the demo, named **Rerank** (whole documents) and **Chunk Rerank** (`chunk_rescorer`) throughout the notebook, article and code; the earlier B/C letters are retired. Each is a Python function returning the `retriever` dict, executed with `es.search(index=CONFIG["index"], retriever=..., size=3)`. `first_stage` accepts `category=None` for the whole catalog, and both functions take `window`.

```python
INFERENCE_ID = ".jina-reranker-v3.5"

def first_stage(q, category):
    return {"standard": {"query": {"bool": {
        "filter": [{"term": {"category": category}}],
        "must": [{"match": {"content": q}}]}}}}

# Rerank — the three manuals of the category, whole documents
def rerank(q, category, window=3):
    return {"text_similarity_reranker": {
        "retriever": first_stage(q, category),
        "field": "content", "inference_id": INFERENCE_ID, "inference_text": q,
        "rank_window_size": window,
    }}

# Chunk Rerank — same, but send only the best-matching chunk of each manual
def chunk_rerank(q, category, window=3, size=1, max_chunk_size=300):
    r = rerank(q, category, window)
    r["text_similarity_reranker"]["chunk_rescorer"] = {
        "size": size,
        "chunking_settings": {"strategy": "sentence", "max_chunk_size": max_chunk_size, "sentence_overlap": 1},
    }
    return r
```

Section 8 swaps the first stage for a `semantic` query on the `semantic_text` field and points the same two retrievers at `manuals_semantic`. Never set `top_n` in task settings; the coordinator expects one result per chunk. No parameter sweeps.

### 4.4 Measurements
- Per symptom query: rank of the correct manual under Rerank and Chunk Rerank, its margin over the best other manual, and Elasticsearch `took`. One table, nine rows, with win counts and median time per configuration. Elasticsearch does not expose which chunk `chunk_rescorer` selected, and EIS returns no token usage, so neither is reported.
- Request limit: run Rerank once with the whole corpus as the window (no category filter, `rank_window_size=12`) and show Jina's message; then Chunk Rerank the same way (ranks correctly, 0.69 margin).
- `semantic_text`: the nine symptom queries as semantic-only, Rerank and Chunk Rerank on `manuals_semantic` (8 / 4 / 9 of 9).
- No nDCG, no shuffled-order averaging. With three candidates per query the per-query table is the result. List-order sensitivity (§2) is why scores are reported but only ranks are compared.

### 4.5 Caveat shown in the notebook
One section (Section 9), one cell:
- **The answer is already in view.** The three spec queries under Rerank and Chunk Rerank. Rerank ranks all three correctly from the manual heads. Chunk Rerank gets the two that hinge on one word only one manual contains (4,000 PSI, propane) and loses "garage door opener with built-in camera and battery backup": all three openers have battery-backup passages, chunk selection is per manual and lexical, and no chunk can say that the rest of its manual never mentions a camera. Chunking finds a single distinguishing passage anywhere; it loses the document-level context a listwise model uses when the answer is near the front.

Two further caveats were built, run and then cut at the author's direction as not adding value: default `chunking_settings` (6 of 9 at whole-manual latency) and customer-worded queries (lexical selection misses the fix; `size: 3` helped at the margin). The default-chunking warning lives in Section 5's narrative and Section 10's recommended configuration; the lexical limit is one line in Section 10's follow-ups.

---

## 5. Remaining checks
- None open. The Cloud API key authenticates against the project's Elasticsearch endpoint (verified in notebook Section 2).

---

## 6. Risks and fallbacks
- **Sibling manuals differ too much in their heads.** If Rerank ranks the correct manual first on most symptom queries, the visible heads carry model-specific signal. Check Rerank's three scores per query; they should be close. Fix by copying more of the head's sections verbatim in the generator and regenerating the cluster.
- **Rerank gets lucky.** One chance in three per query, nine queries. Report the table as is; add queries rather than reruns if the picture is muddy.
- **Lexical chunk selection misses the fix.** Customers say "won't turn on"; manuals say "fails to energize". Measured (1 of 3 customer-worded queries) but not shown in the notebook; stated as a limit in Section 10. Synonyms or query expansion at the first stage, or semantic chunk scoring, are the follow-ups.
- **A cluster exceeds the request limit under Rerank.** Cluster totals are 62K to 94K real tokens against a ~160K limit. If a regeneration lands larger, lower that cluster's target and regenerate with `--only ... --force`.
- **Credibility of a made-up corpus.** State plainly that the manuals are synthetic and modeled on real structure, and that the cliff is measured directly against the reranker in Section 3 before the corpus is used.
- **`chunk_rescorer` not enabled on Serverless.** Fall back to Elastic Cloud Hosted 9.x with EIS; the story is unchanged.
- **Cost.** Negligible: about thirty rerank calls for the whole notebook.

---

## 7. Deliverable structure

### 7.1 Terraform (`terraform/`)
Current state, as written:
- `elastic_serverless.tf`: the `ec` provider and the `ec_elasticsearch_project` resource named `demo_project`.
- `variables.tf`: `elastic_cloud_api_key` (sensitive), `region` (default `gcp-us-central1`).
- `outputs.tf`: `elastic_cloud_id`, `elastic_cloud_api_key` (sensitive).
- `terraform.tfvars.sample`: template for the one required variable.

Review notes to resolve:
- **Elasticsearch authentication.** Resolved: the Cloud API key authenticates `es.info()` in Section 2; no scoped Elasticsearch API key is needed.
- **Silent failures.** Section 1's `%%bash` cell sends `terraform init` and `apply` output to `/dev/null`, so a failed apply prints nothing but a missing "Done." Show stderr, or at least `terraform apply` output on failure.
- **Project name is fixed** (`demo_project`). Fine for a demo; a variable would let two readers in one org run it side by side.
- **Unused provider.** Resolved: `elasticstack` removed.
- README exists (`README.md`): summary, presentation link, architecture image, features, prerequisites, installation, usage. Section 1's markdown cell is still only a heading; a cost and teardown note there remains open.

### 7.2 Notebook (`demo.ipynb`)
One notebook, top to bottom, complete and executed. Each block is a markdown cell titled `## Section N - Title`, a bulleted narrative, a "What the results show, and why" block, then one code cell. Sections 1 and 10 are `%%bash` cells that run Terraform; everything in between is Python against the `elasticsearch` client.

| # | Title | Code |
|---|---|---|
| 1 | Elastic Serverless Provisioning | `%%bash`: `terraform init` / `apply`, write `.env` |
| 2 | Connect to the Serverless project | `dotenv`, `CONFIG` dict (index, semantic_index, inference_id, rank_window_size, max_chunk_size, chunk_size, paths), client, `es.info()` |
| 3 | The cap, measured | `probe.probe_doc`; five two-document `es.inference.rerank` calls, control first; yes/no table |
| 4 | Load and index the corpus | `corpus.load_manuals` + `describe`; mapping; `helpers.bulk` keyed by model |
| 5 | Two retriever configurations | `first_stage`, `rerank`, `chunk_rerank`; the weep-hole query through both; `display.show_comparison` |
| 6 | Run the queries | nine symptom queries through both; `display.show_query_table` (4 of 9 vs 9 of 9) |
| 7 | The request limit | Rerank with no filter, window 12: `BadRequestError`, `display.show_rejection`; Chunk Rerank ranks the catalog |
| 8 | Does semantic_text avoid this? | `manuals_semantic` index; semantic-only, Rerank and Chunk Rerank (8 / 4 / 9 of 9); three-column table |
| 9 | When the answer is already in view | the three spec queries through both (3 of 3 vs 2 of 3) |
| 10 | Takeaways and teardown | findings, recommended configuration, two ways to solve it, follow-ups; `%%bash`: `terraform destroy`, remove `.env` |

Conventions:
- Python client only: every Elasticsearch call is an `es.*` method; the notebook never shows raw HTTP or Console syntax.
- Cells hold concept code only (the query, the document under test, the `es.*` call, the verdict); plumbing such as text generation, caching, loading and rendering lives in modules under `src/chunk_rescorer/` and is imported. Section 3 uses `probe.py`; Sections 4 to 9 use `corpus.py` and `display.py`.
- Every code cell is idempotent: create-if-missing; delete-and-recreate for test indices.
- All knobs (chunk size, window, index names, inference id) in one `CONFIG` dict in Section 2.

---

## 8. Deliverables beyond the notebook, and departures from this plan

**Deliverables (all under `assets/` unless noted)**
- `article.md`: the LinkedIn article, title "What Your Reranker Never Sees". Sections mirror the notebook, with result blocks pasted from the executed cells and one image per section.
- `README.md` (repo root): summary, slide deck link, architecture image, features, prerequisites, install and usage.
- `index.html` (repo root): a self-contained ten-slide HTML deck for an Elastic Architect presenting to customer engineers and their management. Dark theme, fixed 1600×900 stage scaled to the window, arrow-key and button navigation, uses the cover, architecture and section images. Generated from `prompts/prompt_preso.txt`; layout verified headless with a built-in `?check` overflow test.
- `images/`: `cover.png` (LinkedIn article cover, 1920×1080), `arch.png` (README architecture diagram, kept at the generator's native size), `section3.png` to `section9.png` (article section diagrams, 900×300). All PNG, dark theme, no logos.
- `images/resize.sh`: brings generated images to exact sizes (`cover`, `arch`, `section` presets or `WxH`); fills and centre-crops when aspect ratios are close, pads with the image's own background otherwise; flattens transparent inputs onto the dark theme colour.
- `prompts/`: image-generation prompts (`prompt_cover.txt`, `prompt_arch.txt`, `prompt_section3.txt` to `prompt_section9.txt`, `prompt_preso.txt`). Images are generated with gpt-image-2.5, which does not honour requested sizes, hence `resize.sh`.

**Departures from the plan as first written**
- Configuration names: **Rerank** and **Chunk Rerank** replace B and C everywhere.
- Section 8 is new: the `semantic_text` test, added after the author challenged the claim that the cap applies to `semantic_text` fields; settled empirically (§2).
- The caveats section was reduced to the spec-query caveat (§4.5); the default-chunking and customer-wording cells were cut, and `query_mismatch` and `short_manual_query` were removed from `queries.json`.
- The notebook never tokenizes; all token figures are manifest estimates. No `transformers` dependency.
- `requests` and `ipython` were dropped from `pyproject.toml`; the project depends on `elasticsearch`, `jupyterlab` and `python-dotenv` only.
- Comparison with other platforms: the article and Section 10 describe the alternative (chunk at ingest, chunk is the record) generically, without naming vendors. Google's Vertex AI Ranking API was checked against its docs (per-record limit 512 or 1,024 tokens, truncation, no passage selection) and is the reference case.

**Still open**
- Section 1's markdown cell is a bare heading; the cost and teardown note is unwritten.
- Section 1's `%%bash` cell still sends Terraform output to `/dev/null`.
