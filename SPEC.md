# chatlectify — Build Specification

> Single-source-of-truth spec. Hand verbatim to Claude Code. Every ambiguity resolved; every sanity check explicit.

---

## 0. Project

- **Name:** `chatlectify`
- **Package (PyPI / GitHub repo):** `chatlectify`
- **CLI entry:** `chatlectify`
- **Tagline:** Compile your AI chat history into a writing-style skill.
- **Goal:** Ingest AI chat export → emit Claude `SKILL.md` + portable system prompt + benchmark report proving style fidelity.
- **Constraints:** local-first, BYO API key, Python 3.11+, <600 LOC (excl. tests), single binary install.
- **Non-goals:** web UI, multi-user, hosted service, OAuth proxying, live session scraping.

### Accuracy expectation (documented in README)
Surface features (vocab, punctuation, sentence length, formatting): high fidelity (75–90% match). Deep voice / rhetorical style: limited (50–65%). Same ceiling as all current prompt-based style-transfer tools. Benchmark module exposes exact numbers per run.

---

## 1. File Structure

```
chatlectify/
├── pyproject.toml
├── README.md
├── LICENSE                        # MIT
├── SPEC.md                        # this file
├── src/chatlectify/
│   ├── __init__.py
│   ├── cli.py                     # typer entry
│   ├── ingest.py                  # format detect + parse
│   ├── clean.py                   # filter / dedupe / sanitize
│   ├── features.py                # stylometric extraction
│   ├── gates.py                   # sanity + variance gates
│   ├── synth.py                   # LLM → SKILL.md
│   ├── benchmark.py               # A/B tests
│   ├── emit.py                    # output writers
│   ├── schemas.py                 # pydantic models
│   └── assets/
│       ├── verbs_top500.txt
│       ├── hedges.txt
│       ├── antiexemplars.txt
│       ├── dict_words.txt
│       └── baseline_prompts.txt
├── tests/
│   ├── fixtures/
│   │   ├── claude_export.json     # ~20 msgs minimal
│   │   ├── chatgpt_export.json
│   │   └── expected_features.json
│   ├── test_ingest.py
│   ├── test_clean.py
│   ├── test_features.py
│   ├── test_gates.py
│   ├── test_synth.py              # mocks LLM
│   └── test_benchmark.py          # mocks LLM
├── scripts/
│   └── bundle_assets.py           # generates assets from public sources
└── .github/workflows/ci.yml       # pytest + ruff
```

---

## 2. Dependencies

```toml
[project]
name = "chatlectify"
version = "0.1.0"
description = "Compile your AI chat history into a writing-style skill."
license = {text = "MIT"}
requires-python = ">=3.11"
readme = "README.md"
dependencies = [
    "typer>=0.12",
    "pydantic>=2.6",
    "scikit-learn>=1.4",
    "numpy>=1.26",
    "anthropic>=0.39",
    "openai>=1.40",
    "tiktoken>=0.7",
    "regex>=2024.5",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]
[project.scripts]
chatlectify = "chatlectify.cli:app"
[project.urls]
Homepage = "https://chatlectify.dev"
Repository = "https://github.com/chatlectify/chatlectify"
```

Rationale: no heavyweight NLP (no spaCy/NLTK). Regex + sklearn covers everything. tiktoken for length accuracy.

---

## 3. CLI Contract

```
chatlectify ingest INPUT_PATH [--format auto|claude|chatgpt|gemini] [--out normalized.json]
chatlectify features INPUT [--out features.json]
chatlectify build INPUT --out-dir ./skill/ [--provider anthropic|openai] [--model MODEL] [--force]
chatlectify benchmark --skill ./skill/SKILL.md --user-samples user.json [--provider X] [--n 100]
chatlectify all INPUT --out-dir ./skill/ [--provider X] [--skip-benchmark]
```

**Env vars:** `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. No hardcoding. CLI exits 2 with clear message if missing.

---

## 4. Schemas (`schemas.py`, pydantic)

```python
class Message(BaseModel):
    msg_id: str
    conv_id: str
    timestamp: datetime | None
    role: Literal["human", "assistant"]
    text: str
    word_count: int

class StyleFeatures(BaseModel):
    # lexical
    ttr: float
    avg_word_len: float
    top_unigrams: list[tuple[str, int]]     # top 100
    top_bigrams:  list[tuple[str, int]]     # top 50
    contraction_rate: float                 # per 100 words
    # syntactic
    avg_sent_len: float
    sent_len_std: float
    punct_hist: dict[str, float]            # rate per 1000 chars
    cap_start_ratio: float
    # structural
    bullet_rate: float
    header_rate: float
    fence_rate: float
    avg_line_breaks: float
    # markers
    top_sent_starters: list[tuple[str, int]]
    hedge_rate: float
    emoji_rate: float
    typo_rate: float
    # meta
    imperative_rate: float
    question_rate: float
    avg_msg_words: float
    avg_msg_words_std: float
    msg_count: int

class GateReport(BaseModel):
    passed: bool
    msg_count: int
    char_count: int
    paste_contamination_pct: float
    warnings: list[str]
    errors: list[str]

class BenchmarkReport(BaseModel):
    ngram_auc_baseline: float
    ngram_auc_skill: float
    ngram_auc_delta: float
    feature_dist_baseline: float
    feature_dist_skill: float
    feature_dist_improvement: float
    pass_criteria: dict[str, bool]
    overall_pass: bool
```

---

## 5. Module Specs

### 5.1 `ingest.py`

**Format auto-detect:**
```
if top-level key 'chat_messages' present → claude
elif top-level key 'mapping' or list-of-dicts with mapping nodes → chatgpt
elif input.endswith('.html') → gemini (regex scrape)
else raise ValueError with detected keys
```

**Claude schema:** list of `{uuid, name, chat_messages: [{uuid, text, sender, created_at, ...}]}`. Emit `Message` for `sender in ("human", "user")`.

**ChatGPT schema:** tree via `mapping[node_id].message.author.role`. DFS, collect `role == "user"` leaves.

**Gemini (Takeout HTML):** extract user queries via bs4 or regex targeting user blocks. Fallback: lines prefixed "You:".

**Sanity checks:**
- Raise if parsed message count == 0
- Raise if any msg.text is None / empty
- Warn if detected format mismatches file extension
- Timestamps: tolerate missing (set to None, don't crash)

### 5.2 `clean.py`

**Pipeline (order matters):**
1. Strip fenced code: `re.sub(r'```.*?```', '', text, flags=re.DOTALL)`
2. Strip indented code: lines with ≥4 leading spaces AND no prose punctuation
3. Strip URLs: `re.sub(r'https?://\S+', '', text)`
4. Strip emails: standard regex
5. Collapse whitespace
6. Discard if `word_count < 5` post-clean
7. Dedupe: `md5(text[:200].lower().encode()).hexdigest()` as key

**Paste contamination detection (per msg):**
```python
is_paste = any([
    word_count > global_p95_wc,
    line_count > 20,
    alnum_ratio < 0.6,
    unique_char_ratio < 0.15,
])
```
Flag, don't drop. Emit separate `pastes.json`. Main pipeline uses non-paste messages only.

**Sanity checks:**
- Assert `len(cleaned) <= len(raw)`
- Log counts: `raw=N, post_clean=M, deduped=K, pastes_flagged=P`
- Fail if `M/N < 0.3` (over-aggressive cleaning bug)

### 5.3 `features.py`

**Implementation notes:**
- Sentence split: regex `(?<=[.!?])\s+(?=[A-Z])` + single-newline fallback
- Word split: `regex.findall(r'\b\w+\b', text.lower())`
- Punct hist denominator: total chars (not words)
- Top-N tokens: keep stopwords — stopword usage IS style
- Contractions: list `["n't","'re","'s","'ve","'ll","'d","'m"]`; substring hits
- Hedge list: bundled ~30 items, case-insensitive word-boundary match
- Emoji: regex unicode `\p{Emoji}`
- Typos: compare against bundled SCOWL 50k; rate = `unknown_words / total_words`; skip tokens <3 chars or with digits
- Imperative: first token in bundled top-500 verbs AND sentence not starting with capital pronoun

**Sanity checks:**
- Assert all rates in `[0, 1]` or documented unit
- Assert `top_unigrams` non-empty if `msg_count > 0`
- Fail loudly on NaN (bad division)
- Unit test: known fixture → assert expected values within ±2%

### 5.4 `gates.py`

```python
def run_gates(msgs, features, paste_pct) -> GateReport:
    errors, warnings = [], []
    char_count = sum(len(m.text) for m in msgs)

    # hard gates
    if features.msg_count < 200: errors.append("min_msgs=200 failed")
    if char_count < 20_000:      errors.append("min_chars=20000 failed")
    if paste_pct > 0.5:          errors.append(f"paste_contamination {paste_pct:.1%} > 50%")

    # soft gates
    if 0.3 < paste_pct <= 0.5:   warnings.append(f"paste_contamination {paste_pct:.1%}")
    if features.msg_count < 500: warnings.append("low sample; fidelity may degrade")
    if features.avg_msg_words < 3: warnings.append("mostly very short msgs")
    if features.ttr > 0.8:       warnings.append("unusually high TTR")

    return GateReport(passed=len(errors)==0, ...)
```

**Exit behavior:** `build` aborts if `passed=False` unless `--force`.

### 5.5 `synth.py`

**Inputs:** `StyleFeatures`, 40 stratified exemplars (10 short + 20 mid + 10 long), 5 anti-exemplars from bundled `antiexemplars.txt` (generic AI prose).

**Prompt (single LLM call):**
```
You are building a writing-style skill file. Output ONLY a SKILL.md with YAML frontmatter.

<user_style_features>
{features_json_truncated}
</user_style_features>

<user_exemplars>
{40 numbered messages}
</user_exemplars>

<anti_exemplars>
{5 generic-AI-voice paragraphs}
</anti_exemplars>

Produce SKILL.md with:
1. YAML frontmatter: name (snake_case), description (1 sentence, <200 chars)
2. ## Style Rules — 10-15 imperative rules derived from features (concrete, measurable)
3. ## Exemplars — 5 best user messages verbatim
4. ## Anti-patterns — bullet list of "do NOT" derived from anti-exemplars contrast
5. ## Quantified Targets — table: metric | target value

Output ONLY the SKILL.md content. No preamble. No code fences around the whole document.
```

**Sanity checks:**
- Parse output → verify frontmatter is valid YAML
- Verify `name` matches `^[a-z][a-z0-9_-]{1,63}$`
- Verify all 5 sections present (regex for `## `)
- If any fails: retry once with error appended; if still fails, fall back to deterministic template
- Cost gate: abort if input tokens >50k

### 5.6 `benchmark.py`

**Split:** 80/20 on cleaned messages → `U_train`, `U_test` (stratified by length bucket).

**Generate comparison sets:** for N=100 prompts (extract topic from first sentence of each `U_test` msg):
- `baseline_out[i]`: LLM with generic system prompt (`"You are a helpful assistant. Answer in 1-3 sentences."`)
- `skill_out[i]`: LLM with generated SKILL.md body as system prompt
- Both: same model, `temp=0.7`, `max_tokens=150`, seeded

**Test A — n-gram classifier AUC:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def ngram_auc(user_texts, ai_texts) -> float:
    X = user_texts + ai_texts
    y = [1]*len(user_texts) + [0]*len(ai_texts)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=5000)
    Xv = vec.fit_transform(X)
    return float(cross_val_score(
        LogisticRegression(max_iter=1000, random_state=42),
        Xv, y, cv=5, scoring='roc_auc'
    ).mean())
```

**Test B — feature distance:** cosine between feature vectors of `U_test`, `baseline_out`, `skill_out` (reuse `features.py`).

**Pass criteria:**
```
pass_ngram   = auc_skill < 0.75 AND (auc_baseline - auc_skill) > 0.10
pass_feat    = d_skill < d_baseline AND (d_baseline - d_skill) / d_baseline > 0.20
overall_pass = pass_ngram AND pass_feat
```

**Sanity checks:**
- Abort if `len(U_test) < 30`
- Abort if any `baseline_out[i]` empty (LLM call failed)
- Log 2 baseline + 2 skill samples for manual eyeball
- Seed `random_state=42` everywhere
- Validate AUC ∈ [0,1]; flag bug if baseline AUC is 0.5 ± 0.02 (should be distinguishable)

### 5.7 `emit.py`

Writes to `out_dir`:
- `SKILL.md` — direct from synth
- `system_prompt.txt` — SKILL.md minus frontmatter, prepended with `"Write in this style:\n\n"`
- `style_metrics.json` — `StyleFeatures` dump
- `gate_report.json` — `GateReport` dump
- `benchmark_report.json` — if benchmark ran
- `exemplars.json` — top 40 stratified messages
- `pastes.json` — flagged paste contamination
- `pipeline_report.json` — counts + pass/fail per stage

---

## 6. Bundled Assets (`src/chatlectify/assets/`)

Generated by `scripts/bundle_assets.py` or sourced from permissive public lists (cite in README):

- `verbs_top500.txt` — top-500 English verbs (for imperative detection)
- `hedges.txt` — ~30 hedge words
- `antiexemplars.txt` — 20 generic AI paragraphs (2–3 sentences each)
- `dict_words.txt` — SCOWL 50k (typo detection)
- `baseline_prompts.txt` — 100 generic topics for benchmark

---

## 7. Sanity-Check Philosophy

Every module **must**:
1. Validate inputs at entry (pydantic handles most)
2. Log counts at each stage boundary
3. Use `assert` for invariants
4. Fail loudly on NaN/Inf (explicit `np.isfinite` checks)
5. Seed all RNG with `42`
6. Emit final `pipeline_report.json` summarizing counts + pass/fail at each gate

**Rule:** no silent `try/except` swallowing. Either handle with specific recovery or propagate.

---

## 8. Build Order

Execute in order. Each step ends with passing tests before proceeding. Commit after each step with message `step N: <module>`.

1. `pyproject.toml` + `src/chatlectify/__init__.py` + `schemas.py`
2. `ingest.py` + `tests/test_ingest.py` (fixture: `claude_export.json` with 10 msgs)
3. `clean.py` + `tests/test_clean.py` (dedupe, paste detection, code strip)
4. `features.py` + `tests/test_features.py` (exact value checks on fixture)
5. `gates.py` + `tests/test_gates.py` (pass/fail paths)
6. `cli.py` with `ingest`, `features` subcommands; manual smoke test
7. `synth.py` (mocked LLM for tests; real in CLI)
8. `benchmark.py` (mocked LLM for unit tests; real integration in `all`)
9. `emit.py` + wire `build` and `all` commands
10. `README.md` — install, usage, troubleshooting, accuracy expectations (cite §0)
11. CI: `.github/workflows/ci.yml` running ruff + pytest
12. `scripts/bundle_assets.py` — populates `src/chatlectify/assets/`

**Per step:** run `pytest -x tests/test_<module>.py && ruff check src/`. Do not proceed on failure.

---

## 9. Acceptance Criteria

- [ ] `chatlectify all tests/fixtures/claude_export.json --out-dir ./out/` exits 0
- [ ] `./out/SKILL.md` parses as valid YAML frontmatter + markdown
- [ ] `./out/benchmark_report.json` contains every `BenchmarkReport` field
- [ ] On dogfood test (≥500-msg real export): `overall_pass=True` OR clear failure reason in report
- [ ] `pytest` 100% green
- [ ] `ruff check` clean
- [ ] README documents: install, all 5 commands, gate thresholds, accuracy ceiling
- [ ] LOC (excl. tests/fixtures/assets): <600 per `cloc src/`

---

## 10. Explicit Non-Requirements (do NOT build)

- GUI / web frontend
- Cloud storage / user accounts
- OAuth to Claude.ai / ChatGPT
- Scraping live sessions
- Real-time streaming
- Multi-language (English-only v1)
- Fine-tuning (prompt-engineering only)
- Auto-update / telemetry

---

## 11. Open Parameters (CC picks sensible defaults, documents in README)

- Default model: `claude-sonnet-4-6` or `gpt-4o-mini`
- Benchmark sample N: 100 (configurable `--n`)
- Exemplar count in SKILL.md: 5
- Paste detection threshold: per-user p95
- Stratification buckets: short (<10 words), mid (10–50), long (>50)

---

## 12. Entry Prompt for Claude Code

```
Read SPEC.md end-to-end. Build the project in the order specified in §8. After each numbered step, run the tests listed and show output before proceeding. Do not ask me questions — resolve ambiguities by choosing the option that minimizes LOC and dependencies. If a spec item is genuinely impossible, halt and report with a proposed alternative. Commit after each step with message "step N: <module>".
```
