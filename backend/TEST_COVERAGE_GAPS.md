# Backend Test Coverage Gaps

Measured with `pytest --cov=app` (99% overall, 439 passed). This is a checklist
of functions and branches that have **no test coverage**, grouped by module.
Items marked **ERROR PATH** are exactly the failure modes that matter in
production: Qdrant down, Redis down, LLM timeout/retry exhaustion, and malformed
SQLite rows (the project uses SQLite via aiosqlite, not MySQL — same concept:
rows that don't match the expected schema).

Legend: `[x]` checked = covered by `tests/`; `[ ]` unchecked = gap to cover.
Line numbers refer to the current `backend/app/*.py`.

---

## app/llm.py — 97% (covered by tests/test_llm.py)

Retry/timeout engine fully exercised: happy path, backoff retry, exhaustion,
non-retryable immediate failure, `_is_retryable` classification, and the stream
variants (retry-before-first-chunk, no mid-stream retry, usage capture).

- [x] **`generate_answer` happy path** (lines 57-71): success return + token
      usage propagation.
- [x] **`generate_answer` retry on retryable error** (lines 74-84): a
      `APITimeoutError`/`APIConnectionError`/`RateLimitError`/5xx triggers
      backoff and re-call; `asyncio.sleep` sequence asserted (no real sleep).
      **ERROR PATH — LLM timeout.**
- [x] **`generate_answer` exhaustion** (line 75): after `LLM_MAX_RETRIES`
      retries still failing → raises `LLMUnavailableError`.
      **ERROR PATH — LLM retry exhaustion.**
- [x] **`generate_answer` non-retryable error** (line 74): a non-retryable
      exception (e.g. `APIStatusError` 4xx / arbitrary exception) raises
      `LLMUnavailableError` immediately, no retry.
- [x] **`_is_retryable`** (lines 42-46): `APIStatusError` 500/429 = retryable;
      4xx non-429 = not; arbitrary exception = not.
- [x] **`stream_answer` happy path** (lines 102-129): chunks yielded, usage
      captured into `usage_holder` as `LLMResult`.
- [x] **`stream_answer` retry-before-first-chunk** (lines 130-134): failure
      before any content → retry with backoff.
      **ERROR PATH — LLM stream timeout.**
- [x] **`stream_answer` mid-stream failure** (line 132, `started=True`): raises
      `LLMUnavailableError` immediately, never retries mid-stream (dedup
      guarantee).
- [x] **`stream_answer` `usage_holder=None`** (lines 121-128): no usage recorded.
- [ ] **Dead-code raises** (lines 85, 143): post-loop `raise LLMUnavailableError`
      — unreachable while the loop always raises at `attempt >= LLM_MAX_RETRIES`;
      not worth covering unless the loop structure changes.

---

## app/reranker.py — 100% (covered by tests/test_reranker.py)

All startup/failure/load paths exercised with faked optimum/transformers/
sentence_transformers imports and a tmp ONNX cache dir.

- [x] **ONNX fast path `predict`** (lines 84-90): 2-D logits → column 0;
      `logits.ndim != 2` branch.
- [x] **torch fallback `predict`** (line 91): `self._onnx is None` path.
- [x] **ONNX load failure fallback** (lines 35-51): `optimum` import/export
      raises → falls back to torch. **ERROR PATH — model load failure.**
- [x] **`_load_onnx` cached-load branch** (lines 61-62): `model.onnx` already
      present → load without lock.
- [x] **`_load_onnx` export-under-lock branch** (lines 64-78): first run exports
      and saves; `fcntl` lock acquire/release (lines 66-80).
- [x] **`_load_onnx` double-checked lock** (line 71): second worker finds the
      file after acquiring the lock. **ERROR PATH — concurrent gunicorn
      workers racing the export.**

---

## app/encoders.py — 100% (covered by tests/test_encoders.py)

All startup/encode branches exercised with faked fastembed /
sentence_transformers imports (no model download or real inference).

- [x] **fastembed init success** (lines 24-28): `_model` set, ONNX path used;
      `cuda=False` on cpu vs `cuda=True` on a non-cpu device.
- [x] **fastembed load failure → torch fallback** (lines 29-35).
      **ERROR PATH — model unavailable at startup.**
- [x] **`encode` both branches** (lines 39-41): torch fallback
      (`normalize_embeddings=True` asserted) vs fastembed generator path
      (`batch_size=1`).

---

## app/diversity.py — 100% (covered by tests/test_diversity.py)

- [x] **`diversify` short-circuit** (line 34): `len(results) <= n`, including
      truncation under `n`.
- [x] **`diversify` MMR loop** (lines 38-59): greedy selection; `_jaccard`
      similarity with the `sim_thresh` floor (above vs below floor);
      `lam` weighting (`lam=1.0` pure relevance vs `lam=0.0` pure diversity);
      `max_sim` over multiple chosen indices; `>` keeps first on ties.
- [x] **`_tokens` empty/None title** (line 17).
- [x] **`_jaccard` empty-set branch** (lines 21-22).

---

## app/query_fix.py — 100% (covered by tests/test_query_fix.py)

symspellpy faked in sys.modules; vocab artifacts written to tmp paths.

- [x] **`QueryFixer._build` with vocab** (lines 93-111): SymSpell built
      (`max_dictionary_edit_distance`, `prefix_length=7`), curated entities get
      `_CURATED_COUNT` via `create_dictionary_entry` even when already in the
      vocab.
- [x] **`_build` empty vocab → disabled** (lines 100-102): reached via a direct
      `_build` call with an empty entity list (`__init__` always injects the
      curated entity list, so this is unreachable through the constructor).
- [x] **`_load_vocab` file absent** (line 82) and **corrupt gzip/JSON → {}**
      (lines 88-90), plus the non-conforming-row filter (line 87).
      **ERROR PATH — malformed vocab artifact.**
- [x] **`_allowed_distance`** (line 115): short token → max edit 1.
- [x] **`fix` full pipeline** (lines 117-147): no-op when disabled (`_sym is
      None`, line 119) and on empty text; known/short/digit passthrough (line
      126); no-suggestion passthrough (line 130); suggestion equal to input
      and zero-distance (line 132); `sug.count < min_count` rejection (line
      134); capitalization restore (lines 137-138); fix list building.
- [x] **`init_fixer` disabled branch** (lines 162-164), **`init_fixer` enabled
      build** (line 165), and **`fix_query` no-op when fixer None** (line 171).

---

## app/click_boost.py — 100% (covered by tests/test_click_boost.py)

- [x] **disabled / empty results short-circuit** (line 17): both skip the
      `click_signals` call entirely.
- [x] **no click signals** (lines 20-21): pass-through, no mutation, no re-sort.
- [x] **boost loop** (lines 24-30): `CLICK_BOOST_MIN_ARTICLE_CLICKS` +
      `CLICK_BOOST_MIN_SHARE` gating (below-either → untouched), the
      `max(1, int(total * share))` floor, missing-`id` results (default 0
      clicks), score multiply by `CLICK_BOOST_MULT`.
- [x] **re-sort after change** (lines 31-32): boosted list sorted score-desc;
      no-change case keeps original order (`changed` stays False).
      **ERROR PATH — Redis down** (`click_signals` degraded → `None` →
      pass-through).

---

## app/health.py — 100% (covered by tests/test_health.py)

Redis `from_url`/`ping` mocked (no live Redis), the module-global
`_redis_client` reset between tests, and Qdrant faked with a `collection_exists`
coroutine.

- [x] **`/health`, `/live`** (lines 16, 21).
- [x] **`_qdrant_ok` client absent** (lines 27-28) and **Qdrant call failing /
      timing out** (lines 32-33) — `RuntimeError` and `asyncio.wait_for`
      raising `TimeoutError` both → False. **ERROR PATH — Qdrant down.**
- [x] **`_models_ok`** (line 37): all present → True; any key missing → False.
- [x] **`_llm_ok`** (line 41): key set vs empty.
- [x] **`_redis_status`** (lines 49-59): no REDIS_URL → `(True, "memory")`;
      ping ok → `(True, "redis")` (plus client reuse, single `from_url`);
      ping fail / timeout → `(True, "degraded")`. **ERROR PATH — Redis down.**
- [x] **`_readiness_report` + `/ready`/`/readyz`** (lines 62-93): report shape
      asserted with mocked + real checks wired together; ready → 200,
      not-ready → 503 on both endpoints.

---

## app/redis_cache.py — 100% (covered by tests/test_cache.py)

Redis faked in-process (`_RecordingRedis` happy path, `_FakeRedis` for the
degraded path).

- [x] **`get` Redis hit + JSON decode** (lines 42-43): a stored JSON string is
      `json.loads`-ed back into a dict; a Redis miss (`get` → None) falls
      through to the in-memory cache.
- [x] **`get`/`set` degraded fallback to in-memory** (lines 42-45, 51-54):
      Redis raising → warn-once → reads/writes the in-process `TTLCache`.
      **ERROR PATH — Redis down → silent in-process fallback.**
- [x] **`_client` lazy init + reuse** (lines 27-32): `from_url` called once,
      kwargs (`decode_responses`, timeouts) asserted, cached for later calls.
- [x] **`set` success writes JSON + TTL** (lines 50-51): default ttl vs per-call
      override.
- [x] **`close` with active client** (lines 57-58) and **`close` no-op when no
      client** (line 57 guard).

---

## app/cost_budget.py — 100% (covered by tests/test_cost_budget.py)

- [x] **`close` resets `_redis`** (lines 48-50): `aclose` called, global set to
      `None`; `close` no-op when no client.
- [x] **`spend_today` Redis failure → 0.0** (lines 61-63): fail-open on read,
      warn once, return `0.0`. **ERROR PATH — Redis down.**
- [x] **`spend_today` counter read** (line 60): non-empty counter parsed to
      float; `get` → None → `0.0`.
- [x] **`record_cost` failure** (lines 87-88): skip recording, warn once, no
      raise. **ERROR PATH — Redis down.**
- [x] **`assert_within_budget` over-cap** (lines 75-76): confirmed spend ≥ cap
      → `BudgetExceeded`; under-cap allowed; Redis-down never blocks.
- [x] **`_client` lazy init + `_base_url`** (lines 29-43): `from_url` once,
      DB index swapped to `ANALYTICS_REDIS_DB`, connection reused.

---

## app/analytics.py — 100% (covered by tests/test_analytics.py)

- [x] **`_client` first-call init** (lines 33-40): `from_url` once, DB index
      swapped to `ANALYTICS_REDIS_DB`, connection reused.
- [x] **`_degraded` warn-once** (lines 43-47): only the first failure logs.
- [x] **`close`** (lines 52-54): `aclose` called, global reset to `None`;
      no-op when no client.
- [x] **`record_click` with `article_id`** (line 102): per-query per-article
      `analytics:query_click:{q}` sorted-set tally (repeated clicks stack);
      without `article_id` the key is never created.
- [x] **`click_signals`** (lines 111-129): no raw → None; below
      `CLICK_BOOST_MIN_CLICKS` → None; success dict build (zero-count members
      filtered, `total` = sum of per-article counts). **ERROR PATH —
      Redis down → degraded → None.**
- [x] **`_i`/`_f` malformed value branches** (lines 135-136, 142-143):
      non-numeric / un-parseable values → `0` / `0.0`. **ERROR PATH —
      malformed Redis counters.**

---

## app/main.py — 100% (covered by tests/test_main_pipeline.py + test_main_http.py)

- [x] **`lifespan` startup + teardown** (lines 64-96): model/qdrant/llm init,
      chat+auth store connect + wiring, `init_fixer` args, retention task
      cancel + gather, and cache/analytics/cost-budget closes all asserted.
      **ERROR PATH — `ChatStore.connect` raising propagates out of startup.**
- [x] **`_effective_intent` month-scoped branch** (line 242): a month query
      rewrites to the bare topic ("top pharma deals of month january 2025" →
      ("pharma deals", 2025-01-01, 2025-01-31)); user-dates-win and
      no-intent passthrough regression-covered.
- [x] **`_retrieval_queries` year-in-review two-leg branch** (lines 270-272):
      Flashback rewrite + bare-topic second leg; month-scoped single-leg
      (line 275); plain; flashback==topic dedup.
- [x] **`_embed_sparse`** (line 298) — returns the first lazy-generator element
      (runs inside `asyncio.to_thread`; the thread-hop is exercised by
      `hybrid_search`).
- [x] **`hybrid_search`** (lines 315-345): vector-cache miss → encode + sparse
      embed + `cache.set`; cache hit skips encoding; `inference_lock`
      acquired exactly once on miss; RRF prefetch shape (`dense`/`sparse`,
      limit=4×top_k), `FusionQuery(RRF)`, `with_payload` default vs `True`;
      payload→SourceArticle mapping.
- [x] **`body_rescue`** (lines 427-447): empty articles; score ≥ threshold
      short-circuit; stopword-only query; empty bodies; rerank + `max()`
      rescoring + re-sort. **ERROR PATH — reranker predict raising
      propagates.**
- [x] **`_attach_bodies`** (lines 482-492): empty ids skip retrieve; single
      Qdrant retrieve for all ids; missing/`None` payload → empty body.
      **ERROR PATH — retrieve raising propagates.**
- [x] **`_retrieval_leg` query expansion** (lines 500-502): expanded query
      flows to `hybrid_search` with `max(top_k, RERANK_CANDIDATES)`; skipped
      for flashback queries and when `ENABLE_QUERY_EXPANSION` is off.
- [x] **`search` endpoint full path** (lines 554-584): cache hit (validated
      `SourceSummary` models, `record_search` cached=True) vs miss (facet
      filter → `retrieve_and_rerank` → click-boost + diversity wiring,
      `cache.set`, `record_search` cached=False); boost/diversity disabled
      variant; built qfilter passed through. **ERROR PATH — Qdrant/Redis down
      → 500 via TestClient** (`raise_server_exceptions=False`).
- [x] **`source_context` author/industry/dealtype branches** (lines 593-598):
      all facets → `Authors:`/`Industry:`/`Dealtype:` suffixes; none → bare
      `n/a`; body truncated to `body_limit`; no-summary.
- [x] **`_facet_values`** (lines 618-626) and **`facets` cache hit/miss**
      (lines 633-642): sorted string-only values; empty/None results; cache
      hit skips the Qdrant call. **ERROR PATH — facet API raising → 500.**
- [x] **`analytics_click`** (line 656) and **`get_analytics_summary`**
      (line 666): click beacon forwards query/position/id; summary returns
      `analytics_data()`.

---

## app/chat.py — 100% (covered by tests/test_chat.py)

- [x] **`ChatStore.connect` schema migration** (lines 147-152): legacy DBs
      missing `prompt_tokens`/`completion_tokens`/`cost` and separately missing
      `latency_ms` get the columns added with `0` defaults. **ERROR PATH —
      malformed / legacy SQLite schema.**
- [x] **`ChatStore.close`** (lines 155-158): close an open store and the
      idempotent close-when-already-closed no-op.
- [x] **`rename_session` / `delete_session` when session missing** (lines 269,
      282) → 404.
- [x] **`global_stats` exception handler** (lines 387-389): a failing query
      degrades to `{"error": "chat analytics unavailable"}`, never raises.
- [x] **`json_loads` malformed JSON** (lines 399-400): bad JSON, `None`, and
      non-string input all → `[]`. **ERROR PATH — malformed stored rows.**
- [x] **`_row_to_message`** (lines 403-414): malformed/legacy row field
      coercion — bad sources JSON and `NULL`/string token/cost fields fall back
      to `0` defaults.
- [x] **`_smalltalk_reply` non-smalltalk fallthrough** (line 436): empty /
      blank queries and >12-word messages are not small talk.
- [x] **dataviz helpers edge branches**: `_as_float` non-numeric string, bool,
      and `None` (line 491); `_missing_cell` token set incl. "not stated"/"—"
      (line 499); `_valid_value_column` all-missing vs numeric vs non-numeric
      (line 523); `_first_numeric_column` empty rows / empty first row
      (line 528); `_has_label_content` no-label-cols vs all-empty labels
      (line 541); `parse_dataviz` rejection paths — non-dict data, non-string
      columns, non-list rows (lines 561, 565, 567); `_sanitize_dataviz` empty
      text and no-fence passthrough (line 593).
- [x] **`_dataviz_nudge` view pinning** (line 661): a named view appends the
      "exact type of data block" instruction; a generic chart ask does not.
- [x] **`_parse_dataviz_with_view` invalid inputs** (lines 712, 715-716, 718):
      no fence, invalid JSON, and non-dict data → `None`; dict data gets the
      view applied.
- [x] **`_apply_requested_view` rewrite** (line 734): a block that fails to
      re-parse is left verbatim.
- [x] **`_is_ranking_refusal`** (line 776): refusal signatures ("cannot be
      generated", "do not contain specific amounts") → True; empty text and
      genuine ranked answers → False.
- [x] **`_answer_ranked` ranking-nudge retry** (lines 799-808): a refusal is
      re-asked once with `_RANKING_NUDGE` and tokens summed; the
      `LLMUnavailableError` guard keeps the first answer; non-refusals make a
      single call.
- [x] **`_prepare_turn` follow-up inheritance** (lines 856-859, 866, 876, 879):
      vague-follow-up with a year range keeps the previous topic and pins the
      new dates; `body_rescue` runs when `ENABLE_BODY_RESCUE`; no-sources
      short-circuit; weak fallback answer + note. **ERROR PATH — LLM/Qdrant/
      Redis down during retrieval.**
- [x] **`_run_turn` cost recording + finalize** (lines 917-920): budget check →
      `_answer_ranked` → `record_cost(result.cost())` → finalized answer
      (unrequested dataviz blocks stripped).
- [x] **`send_message` `BudgetExceeded` → 429** (lines 1091-1095) and
      **`LLMUnavailableError` → 503** (lines 1096-1100). **ERROR PATH — LLM
      retry exhaustion / daily budget.**
- [x] **`_require_store` uninitialized → 503** (line 1012) and **`_validate_question`
      too-long → 400** (line 1021).
- [x] **`send_message_stream` nudge retry branches** (lines 1169-1176,
      1181-1188): a dataviz/ranking nudge that succeeds swaps the answer and
      sums tokens; a failed nudge (`LLMUnavailableError`) keeps the streamed
      answer. **`error` SSE handlers** (lines 1213, 1216-1218): mid-stream
      `LLMUnavailableError` → "LLM temporarily unavailable" event; unexpected
      exceptions → "Something went wrong" event, never a 500. **ERROR PATH —
      LLMUnavailableError / BudgetExceeded mid-stream.**
- [x] **`retention_loop`** (lines 1225-1233): a failing purge is swallowed and
      the loop keeps ticking; the next tick purges expired conversations.

---

## app/auth.py — 100% (covered by tests/test_auth.py)

- [x] **`verify_password` `ValueError` branch** (lines 155-156): a malformed
      stored hash (bad salt / empty string) is swallowed as a plain `False`.
      **ERROR PATH — malformed stored password hash.**
- [x] **`create_user` non-unique rollback** (lines 264-266): a non-integrity
      INSERT failure rolls back the connection before re-raising, so it never
      holds an open write transaction. **ERROR PATH — SQLite write failure.**
      (The `IntegrityError`/duplicate path is covered by
      `test_concurrent_create_duplicate_race_no_poison`.)
- [x] **`update_user` empty no-op** (line 293): no fields → no SQL issued; the
      name/role/is_active branches persist each field (lines 284-291).
- [x] **`delete_user`** (lines 299-301): user removed and tokens cascade.
- [x] **`issue_token` error rollback** (lines 321-323): an INSERT failure rolls
      back before re-raising. **ERROR PATH — SQLite write failure.**
- [x] **`_require_auth_store` uninitialized → 503** (line 350).
- [x] **`_client_ip` X-Forwarded-For branch** (line 372): first hop wins;
      socket-peer fallback; `"unknown"` when no peer.
- [x] **`get_user` endpoint** (line 548): successful fetch of a user by id plus
      the 404 path.
- [x] **`patch_user` last-admin guard** (line 566): demoting or deactivating the
      last active admin → 400; the guard releases once a second admin exists.
      **ERROR PATH — self-lockout protection.**
- [x] **`delete_user` last-admin guard** (lines 580-585): deleting the last
      active admin → 400; allowed once a second admin exists. **ERROR PATH —
      self-lockout protection.**
- [x] **`bootstrap_admin` write-lock retry loop** (lines 620-624): a persistent
      write lock is retried 5 times with a 1s sleep between attempts, then
      gives up gracefully instead of failing startup. **ERROR PATH — concurrent
      worker bootstrap.**

---

## Remaining small gaps

- [ ] **app/query_expand.py:243** — expansion fallthrough branch (97% covered).
- [ ] **app/query_intent.py:245, 286, 364** — month/year-range edge cases
      (99% covered).
- [ ] **app/index_text.py:37, 43-44, 85, 87, 91, 94-95** — `split_names`
      fallback, `clean` edge cases, `normalize_date` malformed values.
      **ERROR PATH — malformed index rows.**
- [ ] **app/answer_fallback.py** — 100% covered.
- [ ] **app/config.py** — 100% covered.
- [ ] **app/rerank_boost.py** — 100% covered.

---

## Priority order (error paths first)

1. **app/index_text.py + app/query_intent.py + app/query_expand.py edge
   branches** — malformed index rows and month/year-range edge cases (the only
   modules below 100%: index_text 88%, llm 97% dead-code raises, query_expand
   97%, query_intent 99%).

`app/llm.py` (was 32%) is now 97% via `tests/test_llm.py`, `app/reranker.py`
(was 17%) is now 100% via `tests/test_reranker.py`, `app/encoders.py`
(was 25%) is now 100% via `tests/test_encoders.py`, `app/diversity.py`
(was 15%) is now 100% via `tests/test_diversity.py`, `app/query_fix.py`
(was 30%) is now 100% via `tests/test_query_fix.py`, `app/click_boost.py`
(was 15%) is now 100% via `tests/test_click_boost.py`, `app/health.py`
(was 37%) is now 100% via `tests/test_health.py`, `app/redis_cache.py`
(was 85%) is now 100% via `tests/test_cache.py`, `app/cost_budget.py`
(was 92%) is now 100% via `tests/test_cost_budget.py`, `app/analytics.py`
(was 74%) is now 100% via `tests/test_analytics.py`, `app/main.py`
(was 72%) is now 100% via `tests/test_main_pipeline.py` +
`tests/test_main_http.py`, `app/chat.py` (was 84%) is now 100% via
`tests/test_chat.py` (schema migration, error SSEs, budget/LLM HTTP paths,
dataviz edge branches, nudge retries, and retention loop), and `app/auth.py`
(was 91%) is now 100% via `tests/test_auth.py` (malformed-hash handling,
SQLite rollback paths, last-admin lockout guards, `_client_ip`, and the
bootstrap write-lock retry loop).