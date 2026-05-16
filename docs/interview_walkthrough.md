# AutoAnalyst — Interview-Level Walkthrough

## How to Present This Project

Lead with the problem, not the technology. The hook is:

> "Pharma marketing teams run MMM studies that cost $200K+ and take months. The output is a static PDF that sits in a folder. I built a copilot that makes that data queryable in plain English, generates benchmark comparisons, and simulates budget scenarios — in under 10 seconds per question."

That framing lands better than "I built a RAG app with Gradio."

---

## Part 1 — Full Walkthrough

### 1.1 The Problem

Marketing Mix Modeling is a statistical technique that quantifies how much revenue each marketing channel (TV, paid search, Salesforce reps, etc.) drove. Pharma companies run these models annually. The outputs are rich — spend, ROI, adstock parameters, revenue decomposition per channel — but they are locked in JSON/Excel files and require a data scientist to query.

The goal: let a commercial director or brand manager ask "Is our paid search ROI good compared to industry?" and get an answer in seconds, with a chart, without touching a spreadsheet.

### 1.2 Architecture Decision — Why Planner/Executor/Synthesizer

The naive approach is one-shot RAG: retrieve relevant chunks, stuff them in a prompt, get an answer. That breaks for complex questions because:

- A question like "cut TV 20% and reinvest in paid search" requires a simulation tool, not just retrieval
- A question about benchmarks needs a lookup against external data, not what's in the vector store
- Charts cannot be generated inside a prompt

The Planner/Executor/Synthesizer pattern solves this by separating three concerns:

| Stage | Responsibility | LLM? |
|-------|---------------|------|
| Planner | Decides *which* tools to call and *in what order* | Yes |
| Executor | *Runs* those tools | No |
| Synthesizer | *Interprets* the results and writes the answer | Yes |

The executor being LLM-free is intentional. Tool execution is deterministic Python — putting an LLM in that loop would add latency, cost, and non-determinism for no benefit.

### 1.3 Vector Store Design

**Why ChromaDB**: local, persistent, zero infrastructure. For a prototype with 50 documents and a single-server deployment, SQLite-backed ChromaDB is the right call. It would be swapped for Pinecone or Weaviate at scale.

**Why all-MiniLM-L6-v2**: 384-dimensional embeddings, runs on CPU in ~20ms, adequate semantic quality for structured MMM data. Overkill models like `text-embedding-3-large` would be wasted here since the documents are highly structured and the vocabulary is narrow (channel names, brand names, financial terms).

**Document structure**: each JSON file is one MMM model output (one brand, one year). The embedder (`src/embedder.py`) chunks it into one document per channel so that retrieval is channel-level granular, not model-level. Metadata per chunk: `brand`, `sub_vertical`, `year`, `channel`, `category`, `type`.

**Query filtering**: ChromaDB supports metadata filters (`$eq`, `$in`, `$and`). The query parser extracts structured filters from the question (brand name, year, channel, sub-vertical) and applies them as a pre-filter before semantic search. This dramatically improves precision — without it, "Keytruda TV spend" might return vaccine TV data.

### 1.4 Query Parser — Two-Stage Design

```
question → regex_parser() → if empty → llm_parser() → build_where_clause()
```

**Regex first**: fast, zero cost, zero latency. Matches brand names, years, channel names, and domain terms (oncology/vaccines/hcp/consumer) by exact pattern.

**LLM fallback**: for ambiguous phrasing like "last year's blockbuster oncology drug" where regex finds nothing. The LLM is given the full list of known brands and channels from the vector store and asked to return structured JSON. This is a cheap LLM call (small prompt, small output) so it adds ~300ms.

**Design principle**: never make a network call when a local computation can do the job.

### 1.5 Benchmark Fetcher

A static lookup table of ROI ranges per channel per sub-vertical, sourced from published pharma MMM benchmarks. The data structure is:

```
channel → sub_vertical → { roi_low, roi_mid, roi_high, percentile_75 }
```

Given a brand's actual ROI, it computes:
- Which performance tier (outperformer / average / underperformer)
- Gap to median
- Gap to 75th percentile
- Plain English insight string for the synthesizer

This is deliberately not a database query or an external API call. MMM benchmark data changes annually at most. A static dict is the right representation.

### 1.6 Scenario Simulator

Uses a power curve model: `new_revenue = current_revenue × (new_spend / current_spend)^α`

The exponent `α` controls diminishing returns:
- `α = 1.0` → perfectly linear (doubling spend doubles revenue)
- `α = 0.5` → moderate saturation
- `α = 0.3` → strong saturation (e.g., Salesforce calls — reps can only make so many calls)

These exponents are calibrated per channel based on typical pharma MMM model outputs. `paid_search` gets `α = 0.8` because intent-based search scales more linearly than broadcast TV.

Both single-channel and portfolio-level simulation are supported. The portfolio version runs each channel through the same formula and sums the before/after revenue to produce a blended ROI.

### 1.7 Chart Generator

Six chart types, all via matplotlib with a consistent dark theme. All return base64-encoded PNG — no file I/O in the hot path. The Gradio UI writes the base64 to a temp file because Gradio's `gr.Image` component requires a file path.

The synthesizer extracts chart base64 **before** building the LLM context. A base64 PNG can be 200KB+. Putting it in a prompt would waste thousands of tokens and add nothing — the LLM cannot interpret a PNG as text. So: chart goes to the UI directly, the LLM is told only "Chart generated successfully."

### 1.8 Dual Interface

**Gradio (`app.py`)**: UI for interactive use. Yields streaming status updates during the pipeline run so the user sees "Running pipeline…" immediately. Example questions are pre-loaded as buttons. The clear button resets all outputs.

**FastAPI (`main.py`)**: REST API for programmatic access. `POST /analyze` runs the full pipeline. `POST /query` is a raw retrieval endpoint — useful for debugging what the vector store returns for a given question without running the full LLM pipeline. The collection is loaded once at startup via the `lifespan` context manager.

### 1.9 LiteLLM as the LLM Abstraction

LiteLLM provides a unified `completion()` interface over 100+ LLM providers. Swapping from Gemini to GPT-4o or Claude is a one-line config change. This is the right call for a system that might need to run in different cloud environments with different approved vendor lists.

---

## Part 2 — Interview Questions and Answers

### System Design

**Q: Walk me through what happens when a user submits "Which channels are underperforming for Keytruda?"**

The question hits `analyze()` in `app.py`. It calls `pipeline.run()` which first calls `build_tool_registry()` to wire up all tool closures against the live ChromaDB collection.

The planner sends the question to the LLM with a system prompt listing all 9 tools. The LLM reasons: this is a performance question requiring retrieval and benchmarks, so it returns a plan with `mmm_retriever` → `benchmark_fetcher` → `benchmark_comparison_chart`.

The executor runs each step in order. `mmm_retriever` extracts "keytruda" via the query parser, builds a ChromaDB `$eq` filter on `brand`, runs semantic search, and returns formatted chunks. `benchmark_fetcher` receives the channel list from the plan args (populated by the LLM from its knowledge of the context) and returns ROI comparisons. `benchmark_comparison_chart` receives those results and returns a base64 PNG.

The synthesizer receives all results (minus the chart blob), sends them to the LLM, and gets back a structured JSON with summary, per-channel insights, and a 3-5 paragraph narrative. The chart base64 is re-attached before returning.

Gradio receives the `CopilotAnswer` dict, renders the markdown fields, and displays the chart image.

---

**Q: Why two LLM calls instead of one?**

One call would mean asking the LLM to simultaneously decide what to do, interpret the data, and write the answer — with no access to real tool outputs. That forces hallucination.

Two calls separates *planning* (what tools are needed?) from *synthesis* (what do the results mean?). The synthesizer has actual numbers from real tools — it cannot make up figures because the system prompt explicitly prohibits it and the context is grounded.

A single-call approach with function calling / tool use (like OpenAI's tool_calls) could collapse this into one round trip. That is a valid architectural alternative, but it requires the LLM to be mid-stream when tool calls happen, which complicates streaming and error handling.

---

**Q: How does the system handle a question it can't answer?**

At the query parser level: if neither regex nor LLM can extract filters, the retrieval runs with no `where` clause — broader but valid.

At the planner level: if the JSON is malformed, it falls back to retrieval only and returns whatever the vector store has.

At the executor level: each tool call is wrapped in try/except. Errors are logged as strings and passed to the synthesizer, which can acknowledge "benchmark data was not found for this channel" in its narrative.

At the synthesizer level: if JSON parsing fails, it returns the raw LLM text as the narrative with empty insights — the user still gets something, not a 500 error.

---

**Q: How would this scale to 100 concurrent users?**

Right now it would not scale cleanly. The FastAPI `analyze` endpoint calls `run_pipeline()` synchronously inside an `async def`, which blocks the event loop for 2–5 seconds per request. Under concurrent load this creates a queue.

The fix is to run the pipeline in a thread pool: `await asyncio.get_event_loop().run_in_executor(None, run_pipeline, question, collection)`. For deeper async, `litellm.acompletion()` supports native async, so the planner and synthesizer could be `await`ed directly.

At 100+ concurrent users, the vector store becomes a bottleneck. ChromaDB is not designed for high-concurrency reads. The path is Pinecone or Weaviate with connection pooling.

---

**Q: How do you prevent the LLM from making up numbers?**

Three layers. First, the synthesizer system prompt has an explicit rule: "Use actual numbers from the tool results — no made-up figures." Second, all quantitative results come from deterministic Python tools (the executor), not from LLM reasoning. The LLM only sees those results and writes prose around them. Third, the `ChannelInsight` Pydantic model validates that `current_roi` is a float — if the LLM tries to return a string or omit the field, Pydantic will catch it.

That said, hallucination is not fully eliminated. The LLM can still misattribute a number from one brand to another when the context has multiple brands. The mitigation is clear source labelling in the context string.

---

**Q: Why use metadata filters in ChromaDB rather than relying purely on semantic search?**

Semantic search alone would retrieve the 8 most semantically similar chunks, which may span multiple brands and years. If someone asks about Keytruda 2023 and gets back Lenvima 2024 data, the answer is confidently wrong.

Metadata filters apply structured constraints *before* the semantic ranking. This is a hybrid retrieval pattern — structured filter + semantic re-ranking — which consistently outperforms pure vector search for domain-specific structured data.

---

**Q: What is adstock and why does it matter to the scenario simulator?**

Adstock is the carryover effect of advertising — spending on TV this week influences sales next week. It is modelled as an exponential decay. In the MMM JSON files, each channel has an `adstock_decay_rate` field.

The current scenario simulator does not model adstock — it uses a simple power curve applied to the reporting period spend/revenue. This is a simplification. A more accurate simulator would apply the adstock transformation to the projected spend series before estimating revenue impact. This is noted as a future improvement.

---

**Q: How would you add authentication to this?**

Minimum viable: an `X-API-Key` header middleware in FastAPI that validates against a key stored in an environment variable. Rejects requests with a 401 if the header is missing or wrong.

Production level: OAuth2 with JWT bearer tokens, scoped by role (read-only analyst vs. admin who can reload the vector store). FastAPI has first-class support for OAuth2 via `fastapi.security`.

For the Gradio UI, HTTP Basic Auth is supported natively: `demo.launch(auth=("user", "pass"))`.

---

### Technical Deep-Dives

**Q: Why Pydantic for the pipeline models?**

Two reasons. First, it gives explicit schema validation at the boundary between LLM output and Python code — if the LLM returns `"current_roi": "high"` instead of a float, Pydantic raises immediately rather than silently propagating a type error downstream. Second, it generates JSON schemas that can be fed back into the LLM prompt as the expected output format, reducing hallucination.

---

**Q: What is the time complexity of `load_known_entities()`?**

It calls `collection.get()` which returns all documents in the collection. This is O(n) in the number of documents. It is called on every query — currently inside `mmm_retriever` and also inside `build_tool_registry`. For 50 documents this is negligible. At 50,000 documents it becomes a bottleneck. The fix is to cache the result at collection init time since the known entities don't change between requests.

---

**Q: Why does the chart base64 get extracted in the synthesizer rather than the executor?**

The executor's job is to run tools and collect results — it has no knowledge of what the downstream consumer (the synthesizer) needs. Putting extraction logic there would couple the executor to the synthesizer's concerns.

The synthesizer knows it is building an LLM prompt, so it has the context to decide: "this result is a binary blob that the LLM cannot process — extract it separately." The separation keeps each stage focused on one concern.

---

**Q: What happens if the planner hallucinates a tool name that doesn't exist?**

The executor looks up the tool name in `tool_registry`. If it's not found, it logs a warning and appends an error entry: `"ERROR: tool 'imaginary_tool' not registered"`. Execution continues to the next subtask. The synthesizer sees the error string in context and can acknowledge it in the narrative.

This is a graceful degradation pattern. The pipeline does not crash; it returns a partial answer.

---

**Q: How would you evaluate the quality of answers?**

There is no evaluation framework currently. Adding one would involve:
1. A golden dataset: 20–30 questions with known correct answers (manually verified against the raw JSON files)
2. Metrics: factual accuracy (do the numbers match the source data?), completeness (are all requested channels addressed?), plan correctness (did the planner choose the right tools?)
3. Regression suite: run the golden dataset on every PR and flag regressions

Tools: LangSmith or a custom pytest suite with LLM-as-judge for the narrative quality component.

---

## Part 3 — Improvements Not in the Current Project

These are all additive — none requires rewriting existing code.

---

### 3.1 Async Pipeline

**What**: make `run_planner()` and `run_synthesizer()` use `litellm.acompletion()` and make `run()` an `async def`.

**Why it matters**: the synchronous LLM calls block FastAPI's event loop. Under 10 concurrent users, requests queue behind each other. Async removes that bottleneck.

**Steps**:
1. Replace `completion(...)` with `await litellm.acompletion(...)` in both LLM functions
2. Mark `run_planner`, `run_synthesizer`, and `run` as `async def`
3. In `main.py`, `analyze` already is `async def` — remove the `run_in_executor` workaround
4. In `app.py`, Gradio's generator function must call `asyncio.run(run_pipeline(...))` or use `gr.ChatInterface` with async support

---

### 3.2 Response Caching

**What**: cache the full `CopilotAnswer` dict keyed by a hash of the question. Return cached results instantly without hitting the LLM.

**Why it matters**: MMM data does not change in real time. The same question asked twice should not trigger two LLM pipeline runs.

**Steps**:
1. Add `cachetools` (in-memory LRU) or `redis-py` (distributed)
2. In `pipeline.run()`, compute `cache_key = hashlib.sha256(question.encode()).hexdigest()`
3. Check cache before `run_planner()`. On hit, return immediately
4. On miss, run the pipeline and store the result with a 1-hour TTL
5. Add a `POST /cache/clear` admin endpoint to invalidate on data refresh

---

### 3.3 Multi-Turn Conversation

**What**: allow follow-up questions that reference the previous answer — "Now show me the same for vaccines" or "What if I increase that budget by 10%?".

**Why it matters**: single-turn Q&A forces the user to re-state context in every question. Conversation history makes the system feel like an analyst, not a search engine.

**Steps**:
1. Add a session store: `dict[session_id, list[{ role, content }]]` (in-memory or Redis)
2. Generate a `session_id` UUID on first question, return it in the API response
3. On subsequent requests, include `session_id` to retrieve prior turns
4. Prepend conversation history to the planner's user message
5. In the Gradio UI, switch from `gr.Blocks` with a textbox to `gr.ChatInterface` which manages history natively
6. Add a `max_history_turns` limit (e.g., 5) to prevent context overflow

---

### 3.4 Streaming LLM Responses

**What**: stream the synthesizer's narrative token-by-token to the Gradio UI so the user sees words appearing rather than waiting for the full response.

**Why it matters**: the synthesizer writes 3–5 paragraphs. At current LLM speeds that is 5–10 seconds of silence. Streaming reduces perceived latency dramatically.

**Steps**:
1. In `run_synthesizer()`, call `litellm.completion(..., stream=True)` which returns a generator
2. Yield each `chunk.choices[0].delta.content` as it arrives
3. Change `run()` to a generator function that yields partial answers
4. In `app.py`, change `analyze()` to yield streaming markdown updates to `narrative_out` as chunks arrive
5. The summary and insights can still be rendered after the stream completes (they require the full structured JSON)

---

### 3.5 Authentication and Rate Limiting

**What**: API key middleware protecting all `/analyze` and `/query` endpoints, plus per-key rate limiting.

**Why it matters**: the current endpoints are completely open. Any caller can trigger unlimited LLM pipeline runs.

**Steps**:
1. Add `slowapi` to `pyproject.toml`
2. Add a `Limiter` instance to `main.py` with `@limiter.limit("10/minute")` on `/analyze`
3. Add an `APIKeyMiddleware` that reads `X-API-Key` from request headers and validates against `os.getenv("API_KEY")`
4. Return `HTTP 401` on missing key, `HTTP 429` on rate limit exceeded
5. For the Gradio UI: `demo.launch(auth=("admin", os.getenv("GRADIO_PASSWORD")))`

---

### 3.6 OpenTelemetry Observability

**What**: distributed tracing with spans per pipeline stage, plus Prometheus metrics for request count, latency percentiles, and LLM token usage.

**Why it matters**: without observability you cannot answer "why was that query slow?" or "how much did last week's traffic cost in LLM tokens?"

**Steps**:
1. Add `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-exporter-otlp` to dependencies
2. Instrument `main.py` with `FastAPIInstrumentor`
3. In `pipeline.py`, add manual spans: `tracer.start_as_current_span("planner")`, `tracer.start_as_current_span("executor")`, `tracer.start_as_current_span("synthesizer")`
4. Add LLM token usage as span attributes: `span.set_attribute("llm.input_tokens", response.usage.prompt_tokens)`
5. Export to Grafana Tempo (open source) or Datadog
6. Add a `prometheus_client` metrics endpoint at `/metrics`: request count, latency histograms, error rate

---

### 3.7 Automated Evaluation Pipeline

**What**: a pytest-based evaluation harness that runs a golden question set against the live pipeline and asserts correctness.

**Why it matters**: without tests, refactoring the planner prompt or swapping LLM models is flying blind. Regressions are discovered by users.

**Steps**:
1. Create `tests/golden_questions.json` — 20 questions with expected fields: expected tool sequence, expected brand/channel mentions, numeric ranges for key ROI figures
2. Create `tests/test_pipeline_eval.py` — parameterised pytest that runs each golden question
3. Assert: planner chose the correct lead tool, synthesizer `sources` contains the expected brand, `insights[0].current_roi` is within ±5% of the value in the source JSON
4. Mock LiteLLM with `pytest-mock` for fast unit tests; use real LLM for nightly integration tests
5. Add `pytest --eval` to the CI workflow with a pass threshold (e.g., 90% of golden questions must pass)

---

### 3.8 Dynamic Benchmark Data Loading

**What**: move the hardcoded `BENCHMARKS` dict in `benchmark_fetcher.py` to a versioned YAML or JSON config file that can be updated without a code deploy.

**Why it matters**: benchmark data changes annually. Today, updating it requires a code change, a PR, and a deploy.

**Steps**:
1. Extract the `BENCHMARKS` dict to `config/benchmarks.yaml`
2. Add a `load_benchmarks(path: str) -> dict` function with a module-level singleton
3. Add a `POST /admin/benchmarks/reload` endpoint that re-reads the YAML and refreshes the singleton
4. Add a file watcher (via `watchdog`) that auto-reloads on file change in development
5. In production, store the YAML in S3 or GCS and pull it on container startup via an init script

---

### 3.9 Tool Input Validation

**What**: define a Pydantic model for each tool's input args and validate before calling the tool function.

**Why it matters**: the planner LLM can hallucinate malformed args — a missing required key raises a `KeyError` that surfaces as a generic 500 error. Validation gives a clear error message and prevents partial execution.

**Steps**:
1. Define Pydantic input models: `MMMRetrieverArgs`, `BenchmarkFetcherArgs`, `ScenarioSimulatorArgs`, etc.
2. In `run_executor()`, before calling `tool_fn`, parse `subtask.tool_args` against the model: `validated = MMMRetrieverArgs(**subtask.tool_args)`
3. On `ValidationError`, append a structured error to the log: `"ERROR: missing required field 'current_roi' for benchmark_fetcher"`
4. The synthesizer's error message to the user becomes specific and actionable

---

### 3.10 Adstock-Aware Scenario Simulation

**What**: incorporate the adstock decay rates from the MMM JSON files into the scenario simulator instead of using a static power curve.

**Why it matters**: the current simulator ignores that TV spend this period affects revenue in future periods. This makes TV scenario results look too pessimistic (a TV cut looks cheaper than it is) and paid search results too optimistic.

**Steps**:
1. Extend `ScenarioSimulatorArgs` to accept an optional `adstock_decay_rate` field
2. Apply the Koyck lag model: `effective_spend_t = spend_t + decay_rate × effective_spend_{t-1}` over a 12-week window
3. Compute projected revenue against the adstock-adjusted spend series rather than the raw spend change
4. Update the `mmm_retriever` to extract and pass `adstock_decay_rate` per channel to the scenario tool args
5. Add an `adstock_impact` field to the simulator output explaining the lag effect in plain English

---

### 3.11 CI/CD Pipeline

**What**: GitHub Actions workflow that runs linting, type checking, and tests on every PR, and builds + pushes the Docker image on merge to main.

**Steps**:
1. Create `.github/workflows/ci.yml` with jobs: `lint` (`uv run ruff check .`), `typecheck` (`uv run mypy src/`), `test` (`uv run pytest tests/`)
2. Create `.github/workflows/docker.yml` that builds the image on merge to main and pushes to GitHub Container Registry
3. Add `[tool.ruff]` and `[tool.mypy]` sections to `pyproject.toml`
4. Add branch protection: require all CI jobs to pass before merge
5. Add Dependabot config to auto-update Python dependencies weekly

---

## Summary Table

| Improvement | Effort | Impact | Priority |
|-------------|--------|--------|---------|
| Async pipeline | 1 day | High — fixes blocking under load | 1 |
| Auth + rate limiting | 2 days | High — security blocker | 2 |
| CI/CD pipeline | 1 day | High — prevents regressions | 3 |
| Tool input validation | 1 day | Medium — better error messages | 4 |
| Response caching | 1 day | Medium — cost and latency | 5 |
| Automated evaluation | 1 week | High — enables safe iteration | 6 |
| Streaming responses | 2 days | Medium — UX improvement | 7 |
| Multi-turn conversation | 3 days | High — product differentiator | 8 |
| OpenTelemetry | 2 days | Medium — production visibility | 9 |
| Dynamic benchmarks | 2 days | Low — ops convenience | 10 |
| Adstock simulation | 3 days | High — analytical accuracy | 11 |
