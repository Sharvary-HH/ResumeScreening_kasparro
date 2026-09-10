# AI Resume Screening & Ranking System

Screens a folder of resumes against an AI/Python intern brief and returns a ranked
shortlist with a score breakdown and written justification for every candidate. An LLM
does the two things it is genuinely good at — reading messy documents into structured
facts, and judging the depth of a project — while every hard decision (who is eligible,
what the score adds up to, who ranks where) is plain deterministic Python. The output is
`results.json`, a self-contained `results.html`, and a table printed straight to the
terminal.

**[View the live results report →](https://sharvary-hh.github.io/ResumeScreening_kasparro/output/results.html)**
— the real output of a 50-resume run, no setup required.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and fill in the two credentials.

**Resumes are not committed.** `resumes/` ships empty — the candidate files are
personal data (names, emails, phone numbers) and do not belong in a repository. Drop
your own `.pdf` / `.docx` files there to run a fresh batch. The committed
`output/results.json` and `.cache/` are from a real 50-resume run, so the results
below can be inspected without them.

**Python 3.10 note.** This targets Python 3.10. Pydantic v2 evaluates annotations at
runtime there, so `str | None` raises `TypeError` — every model in `src/models.py` uses
`Optional[...]` / `List[...]` from `typing` instead.

### LLM credentials

Any OpenAI-compatible chat-completions endpoint works. The default is Google's OpenAI
compatibility layer:

```
LLM_API_KEY=<your key>
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-flash-lite-latest
```

Get a key at [aistudio.google.com](https://aistudio.google.com/apikey) → **Get API key**.
`.env.example` also lists ready-made settings for Hugging Face Inference Providers and
OpenAI — switching provider is a two-line edit with no code change.

**You may not need a key at all.** `.cache/llm/` is committed to this repo, so
`python main.py --offline` replays all 50 resumes from cache with zero API calls. See
[Reviewer portability](#reviewer-portability).

### GitHub token

Either grab one from an already-authenticated GitHub CLI:

```bash
gh auth token
```

…or create one at github.com → Settings → Developer settings → Personal access tokens →
Fine-grained → **Public repositories (read-only)**. No scopes beyond public read are
needed.

**Without a token GitHub allows only 60 requests per hour.** This run makes up to three
calls per candidate with a GitHub profile, so an unauthenticated run will rate-limit part
way through and those candidates come back `rate_limited` with 0 GitHub points. With a
token the limit is 5,000/hour.

---

## Run

```bash
python main.py --input ./resumes --output ./output/results.json
```

Quick smoke run — verifies the pipeline end to end in well under a minute before you
commit to the full batch:

```bash
python main.py --limit 5
```

Replay everything from cache with no API key and no network:

```bash
python main.py --offline
```

Skip GitHub enrichment:

```bash
python main.py --no-github
```

Run the tests:

```bash
pytest -q
```

**Expected runtime.** A cold run of all 50 resumes took **177 seconds** (~3 min) at the
default 4 workers. A cached re-run of the same 50 takes **1.1 seconds**.

**Where output lands.** `output/results.json` and `output/results.html`, both written on
every run.

**Three ways to read the results:**

| | |
|---|---|
| Terminal | printed automatically at the end of every run |
| HTML | [live on GitHub Pages](https://sharvary-hh.github.io/ResumeScreening_kasparro/output/results.html), or `open output/results.html` (macOS) / `xdg-open` (Linux) |
| JSON | `python -m json.tool output/results.json \| less` |

### All flags

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `./resumes` | Folder of resumes (`.pdf`, `.docx`, `.txt`) |
| `--output` | `./output/results.json` | Where to write the JSON |
| `--limit N` | all | Process only the first N — for fast iteration |
| `--top N` | 10 | Rows in the terminal ranking table |
| `--offline` | off | Serve from cache only; never call the network |
| `--no-github` | off | Skip GitHub enrichment |
| `--no-cache` | off | Ignore the LLM and GitHub caches |
| `--no-html` | off | Skip writing the HTML report |
| `--verbose` | off | Print a line per resume while running |

---

## Actual run output

All 50 resumes, GitHub enrichment on:

```
TOP 12 CANDIDATES
RANK  CANDIDATE               SCORE  AI/40  PY/30  CLOUD/15  GH/10  ENG/5
-------------------------------------------------------------------------
   1  Prathamesh Patil           93     36     28        14     10      5
   2  Yash Maini                 89     35     26        14     10      4
   3  Jyandeep Baishya           85     35     25        12      9      4
   4  ABHINAV MISHRA             85     32     24        14     10      5
   5  Vivek Chimnani             84     32     28        15      4      5
   6  Nevil Jasmin Mehta         84     32     26        12     10      4
   7  Sumaiya Sultana Shaik      83     32     26        12      9      4
   8  PRAJWAL A S                82     36     26        11      5      4
   9  Hari Shanker Sharma        81     35     26        12      4      4
  10  V Sree Raghu Vardhan       81     32     26        12      7      4
  11  Aditi Kala                 81     32     24        12     10      3
  12  Gourab Das                 80     32     28        14      1      5

BATCH SUMMARY
  total resumes          50
  successfully parsed    50
  eligible               40
  rejected               10
  failed unreadable      0
  duplicates skipped     0
  github enriched        33
  github failed          2
  llm failures           0
  run seconds            176.8

10 rejected -- 5 both, 3 no AI evidence, 2 no Python evidence
```

All 50 parsed, none unreadable, no LLM failures. Of the 10 rejections, 5 had neither
Python nor AI evidence, 3 had Python but no AI project, and 2 had AI work on a non-Python
stack. The 2 GitHub failures are dead profile links (404), reported as `not_found` with 0
points rather than being allowed to fail the run.

---

## How scoring works

100 points, assembled from five categories:

| Category | Max | Assigned by | What earns it |
|---|---|---|---|
| `ai_project_depth` | 40 | LLM | Real AI systems: agents, RAG, retrieval, tool calling, state, orchestration, evaluation. Depth of implementation, not name-dropping. |
| `python_backend` | 30 | LLM | Python depth and backend engineering — FastAPI, async, PostgreSQL, Redis, API design. Evidence in projects beats a keyword list. |
| `cloud_fullstack` | 15 | LLM | GCP/AWS/Azure, Docker, CI/CD, real deployment. React/Next.js count as supporting signals in an end-to-end system. |
| `github` | 10 | **Python** | Deterministic: recency of public pushes (0–5) plus relevant maintained repos (0–5). |
| `engineering_depth` | 5 | LLM | Testing, architecture, caching, queues, observability, concurrency, failure handling. |

**Penalties** are a list of `{reason, points}`, each worth 5–15 points, applied when:

- an "AI project" is a thin wrapper around a single LLM or API call, with no workflow,
  data processing, retrieval, state, backend logic, or evaluation; or
- projects are tutorial-style, with no implementation detail or evidence of ownership.

The final arithmetic happens in Python, never in the model:

```
raw   = ai_project_depth + python_backend + cloud_fullstack + engineering_depth + github
total = max(0, min(100, raw - sum(penalty points)))
```

Ties break on AI depth, then Python, then name — so the ordering is stable and a re-run
produces byte-identical output.

---

## Design Decisions

### Filtering is deterministic Python; the LLM never votes on it

`src/screening/eligibility.py` takes a `CandidateProfile` and returns an
`EligibilityResult`. It imports nothing from `src/llm/`, touches no network, and is
therefore exhaustively unit-testable — the eligibility suite runs in milliseconds and
covers every branch.

The extraction prompt is written to match: it asks for neutral facts and explicitly
instructs the model *not* to infer, judge, or evaluate. The model is never asked whether
a candidate is qualified. That is a business rule, and business rules belong in code you
can read, test, and change without re-running a prompt.

A candidate is eligible when there is **Python evidence AND AI evidence**. Evidence is
gathered from a flattened corpus of skills plus every project's and role's tech stack and
description — so Python buried in a project's tech stack counts exactly as much as Python
in the skills list. Matching uses word boundaries, because a substring search for `ai`
hits "email" and "available".

Extra frontend or JVM skills never cause a rejection. The filter screens *for* what is
required, never *against* what else is present; there is a regression test asserting that
Python + RAG + React + Next.js stays eligible.

### Strong vs weak AI keywords

AI keywords are tiered in config. `AI_KEYWORDS_STRONG` is LLM/agentic work — RAG,
LangGraph, tool calling, embeddings, vector stores. `AI_KEYWORDS_WEAK` is classical ML —
TensorFlow, scikit-learn, OpenCV, XGBoost.

Weak-only candidates **pass the gate** but get `"AI evidence is classical ML, not
LLM/agentic"` attached to their concerns. The reasoning: a hard reject on classical ML
would throw away genuinely capable engineers over a keyword, but the role does ask for
agentic work — so the 40-point AI category, which explicitly rewards agentic depth,
demotes them naturally instead. Filtering decides *who is considered*; ranking decides
*who is best*. Conflating the two is how you lose good candidates to a keyword list.

### Scoring: the model judges, Python calculates

The scorer returns sub-scores with cited evidence, plus `project_summary`, `strengths`,
and `concerns`. Every sub-score field carries `Field(ge=..., le=...)` bounds, so an
out-of-range response fails Pydantic validation and re-enters the retry path rather than
silently corrupting a total. Totals, clamping, penalty subtraction, and ranking are pure
Python in `src/screening/ranking.py`, tested against the 0 and 100 boundaries.

### LLM usage: one adapter, structured output, graceful failure

Everything provider-specific lives behind `chat_json(messages, schema_model)` in
`src/llm/client.py`. It:

- sends `temperature=0` for reproducibility;
- sends `response_format={"type": "json_object"}`, and **retries once without it** if the
  provider 400s — support varies across providers;
- embeds `schema_model.model_json_schema()` in the system prompt as well, rather than
  trusting `response_format` alone;
- strips `<think>` blocks and markdown fences, reads `content` only (never concatenating a
  separate `reasoning_content` field), and falls back to extracting the outermost `{...}`;
- treats `finish_reason == "length"` as a failure and retries with a larger budget, rather
  than parsing a truncated object;
- on `ValidationError`, **retries with the validation error text fed back to the model**,
  which works far better than simply asking again;
- honours a server-supplied `Retry-After` on 429 in preference to its own backoff schedule
  — free tiers are strict and usually tell you exactly how long to wait;
- **trips a circuit breaker** on 401/402/403. A bad key or an exhausted quota is one
  run-level problem, not 50 identical candidate problems, so the first one short-circuits
  the rest of the batch and is reported once at the top of the summary;
- **never raises into the pipeline** — it returns `None` and appends a diagnosable reason
  to the caller's error list.

Failures are contained per candidate. A resume that cannot be extracted is recorded in
`failed_files` with its reason. A candidate who is eligible but cannot be scored stays in
the output as eligible with `rank: null` and an error — we never claim someone is
ineligible when the truth is that we failed to read them.

Nulls are absorbed rather than argued with. The extraction prompt tells the model to use
`null` for absent fields, so the schema coerces `null` → `""` / `[]` on those fields
instead of burning a retry rejecting output we asked for.

### Why the LLM is not the source of truth for URLs

After extraction, `github_url` is **overridden deterministically in Python** from the URLs
pulled out of the file itself. URL recovery is a parsing problem with a right answer, not
a judgment call.

### The link-annotation finding

This is the single most consequential detail in the corpus, and it is invisible unless you
look for it.

These resumes hyperlink the *word* "GitHub" rather than printing the URL. The target lives
in the PDF's link annotations, not in the text layer. Measured across all 50 files:

| Source | Count |
|---|---|
| GitHub URL present in visible text | 19 / 50 |
| Present in PDF link annotations | 40 / 50 |
| **Present ONLY in link annotations** | **24 / 50** |
| No GitHub link anywhere | 7 / 50 |

A parser that regexes the extracted text finds 19 profiles. This one collects
`page.get_links()` URIs as well and unions both sources, finding 42. **Text-only
extraction would silently discard more than half the GitHub signal** — and it would fail
quietly, looking exactly like a corpus where candidates simply had no GitHub.

One resume also wraps a URL across a line break, which yields a truncated and wrong
username from regex alone. The link annotation carries the correct URL; a
line-rejoining fallback handles the text-only case, and `merge_urls()` discards a
truncated form when the complete one is also present.

### GitHub scoring is deterministic, not an LLM judgment

Fixed thresholds, so the same profile always yields the same points and the boundaries are
directly unit-testable:

**Activity (0–5)** — days since the most recent public push event:

| ≤30 | ≤90 | ≤180 | ≤365 | >365 | no events |
|---|---|---|---|---|---|
| 5 | 4 | 3 | 2 | 1 | 0 |

**Repositories (0–5)** — non-fork public repos pushed within 365 days whose primary
language is Python, or whose name/description/topics match an AI keyword:

| ≥5 | 3–4 | 2 | 1 | 0 relevant, repos exist | no repos |
|---|---|---|---|---|---|
| 5 | 4 | 3 | 2 | 1 | 0 |

Every failure path returns `total=0` with a descriptive status — `not_found`,
`rate_limited`, `error`, `missing` — and the batch continues. GitHub is enrichment; it is
never allowed to break a run. Results are cached per username, so no user is fetched twice.

### Why there is no OCR and no column detection

Both were considered and deliberately rejected, on measurement rather than assumption.

**No OCR.** All 50 PDFs were checked for embedded text: minimum 2,546 characters, median
3,849, maximum 12,032. There are zero scanned or image-only files. An OCR stage would add
a heavy dependency, minutes of runtime, and transcription errors, to solve a problem this
corpus does not have.

**No layout reconstruction.** Six resumes use a sidebar or two-column layout
(`candidate_04`, `_11`, `_15`, `_23`, `_30`, `_41`). Reading order under plain
`page.get_text()` was checked on each: sections stay intact and parse correctly. Column
detection and bounding-box sorting would be significant complexity for no measured gain.

What the corpus *did* need was link-annotation extraction — which is where that effort
went instead.

One further measured detail: `candidate_07.pdf` has corrupt embedded fonts and makes MuPDF
write zlib and FreeType errors to C-level stderr. Its text extracts perfectly (4,761
characters), so those warnings are suppressed rather than treated as a parse failure.
There is a test pinning that behaviour.

### Reviewer portability

A take-home that dies on a missing credential does not get read. Four things guard against
that:

1. **`.cache/llm/` and `.cache/github/` are committed** (the resumes themselves are not —
   see Setup). LLM cache keys are content
   hashes of model + messages, so running the same 50 resumes with the same model is a
   100% cache hit — the whole batch completes with zero API calls and no key. The GitHub
   cache means a reviewer without a token gets real enrichment scores instead of 33
   `rate_limited` zeros. It stores only the handful of fields that feed the score, which
   is the difference between 264 KB and 4.5 MB of raw API payloads. Verified from a clean
   clone with no `.env`: all 50 candidates, 0 LLM failures, in 8 seconds.
2. **`--offline`** serves from cache only and never touches the network. A cache miss
   records `llm_unavailable` on that candidate and the batch continues.
3. **A missing key fails with a sentence, not a traceback:** `LLM_API_KEY not set. Run
   with --offline to replay cached results, or see README setup.`
4. **`output/results.json` and `output/results.html` are committed**, so the real output
   can be inspected without running anything at all — and the HTML report is
   [published on GitHub Pages](https://sharvary-hh.github.io/ResumeScreening_kasparro/output/results.html)
   so it can be read in a browser without even cloning.

### The HTML report

Python stdlib only — no Jinja2, no build step, no CDN links, no external fonts. One
self-contained file that opens from `file://` with no network. It reads `results.json`
rather than pipeline internals, so it can be regenerated standalone:

```bash
python -m src.report output/results.json
```

Every interpolated value goes through `html.escape()`. Resume text is untrusted input and
a candidate named `<script>` must not execute.

---

## If I Had More Time

- **An OCR fallback**, triggered by the existing quality gate rather than run
  unconditionally — so a scanned resume degrades instead of failing, without paying OCR
  costs on a corpus that does not need it.
- **Layout reconstruction for genuinely broken multi-column PDFs.** The gate that decides
  when to bother is the interesting part: compare naive reading order against
  bounding-box-sorted order and only reconstruct when they meaningfully disagree.
- **Embedding-based section retrieval**, to feed the scorer only the passages relevant to
  each category instead of the whole profile. On long resumes this would cut tokens and
  sharpen the evidence the model cites.
- **A small FastAPI wrapper** — `POST /screen` to enqueue a batch, `GET /results/{id}` to
  fetch it — so the pipeline is callable as a service rather than only as a CLI.
- **A golden-set regression harness.** Hand-label 10 resumes with expected eligibility and
  a score band, then assert against them in CI. Prompt changes currently have no automated
  safety net; this is the gap I would close first.

---

## Project structure

```
├── resumes/                    drop resumes here (empty; not committed)
├── src/
│   ├── config.py               every threshold, weight, and keyword set
│   ├── models.py               all Pydantic schemas
│   ├── parser/
│   │   ├── __init__.py         parse_resume() dispatcher + SHA-256 dedupe
│   │   ├── pdf_parser.py       PyMuPDF text + link annotations
│   │   ├── docx_parser.py      paragraphs, tables, and rels hyperlinks
│   │   ├── urls.py             URL normalization and recovery
│   │   └── quality.py          text quality gate
│   ├── llm/
│   │   ├── client.py           provider adapter, cache, retry, breaker
│   │   ├── extractor.py        resume text -> CandidateProfile
│   │   └── scorer.py           CandidateProfile -> LLMScores
│   ├── screening/
│   │   ├── eligibility.py      deterministic filter (imports no LLM code)
│   │   └── ranking.py          arithmetic and ordering
│   ├── github/client.py        enrichment + deterministic 0-10 scoring
│   ├── report.py               results.json -> self-contained results.html
│   └── pipeline.py             orchestration and batch summary
├── tests/                      81 tests, no network required
├── output/                     results.json + results.html
├── .cache/llm/                 committed, enables --offline
└── main.py
```

`src/screening/eligibility.py` importing nothing from `src/llm/` is the architectural
point of the whole project, not an accident of layout.
