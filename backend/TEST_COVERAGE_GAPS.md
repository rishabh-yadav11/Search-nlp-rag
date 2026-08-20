# Backend Test Coverage Gaps

Measured with `pytest --cov=app` (81% overall, 250 passed). This is a checklist
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

## app/encoders.py — 25%

- [ ] **fastembed init success** (lines 24-28): `_model` set, ONNX path used.
- [ ] **fastembed load failure → torch fallback** (lines 29-35).
      **ERROR PATH — model unavailable at startup.**
- [ ] **`encode` both branches** (lines 39-41): torch fallback vs fastembed
      generator path.

---

## app/diversity.py — 15%

- [ ] **`diversify` short-circuit** (line 34): `len(results) <= n`.
- [ ] **`diversify` MMR loop** (lines 38-59): greedy selection, `_jaccard`
      similarity, `sim_thresh` floor, `lam` weighting.
- [ ] **`_tokens` empty/None title** (line 17).
- [ ] **`_jaccard` empty-set branch** (lines 21-22).

---

## app/query_fix.py — 30%

- [ ] **`QueryFixer._build` with vocab** (lines 93-111): SymSpell built, curated
      entities get `_CURATED_COUNT`.
- [ ] **`_build` empty vocab → disabled** (lines 100-102).
- [ ] **`_load_vocab` file absent** (line 82) and **corrupt gzip/JSON → {}**
      (lines 88-90). **ERROR PATH — malformed vocab artifact.**
- [ ] **`_allowed_distance`** (line 115): short token → max edit 1.
- [ ] **`fix` full pipeline** (lines 117-147): no-op when disabled (`_sym is
      None`, line 119); no-suggestion passthrough (line 130); suggestion equal
      to input (line 132); `sug.count < min_count` rejection (line 134);
      capitalization restore (lines 137-138); fix list building.
- [ ] **`init_fixer` disabled branch** (lines 162-164) and **`fix_query`
      no-op when fixer None** (line 171).

---

## app/click_boost.py — 15%

- [ ] **disabled / empty results short-circuit** (line 17).
- [ ] **no click signals** (lines 20-21).
- [ ] **boost loop** (lines 24-30): `CLICK_BOOST_MIN_ARTICLE_CLICKS` +
      `CLICK_BOOST_MIN_SHARE` gating; score multiply.
- [ ] **re-sort after change** (lines 31-32). **ERROR PATH — Redis down**
      (`click_signals` degraded returns None → pass-through).

---

## app/health.py — 37%

- [ ] **`/health`, `/live`** (lines 16, 21).
- [ ] **`_qdrant_ok` client absent** (lines 27-28) and **Qdrant call failing /
      timing out** (lines 32-33). **ERROR PATH — Qdrant down.**
- [ ] **`_models_ok`** (line 37).
- [ ] **`_llm_ok`** (line 41).
- [ ] **`_redis_status`** (lines 49-59): no REDIS_URL → memory; ping ok → redis;
      ping fail → degraded. **ERROR PATH — Redis down.**
- [ ] **`_readiness_report` + `/ready`/`/readyz`** (lines 62-93): ready vs not
      ready → 200 vs 503.

---

## app/redis_cache.py — 85%

- [ ] **`get` Redis hit + JSON decode** (lines 42-43).
- [ ] **`get`/`set` degraded fallback to in-memory** (lines 42-45, 51-54).
      **ERROR PATH — Redis down → silent in-process fallback.**
- [ ] **`close` with active client** (lines 57-58).

---

## app/cost_budget.py — 92%

- [ ] **`close` resets `_redis`** (lines 48-50).
- [ ] `spend_today` Redis failure → 0.0 (line 61-63), `record_cost` failure
      (lines 87-88), `assert_within_budget` over-cap (line 75-76) are the
      branches to probe — partly covered; confirm the **ERROR PATH — Redis
      down** degrade is asserted (fail-open on read, skip record on write).

---

## app/analytics.py — 74%

- [ ] **`_client` first-call init** (lines 33-40).
- [ ] **`_degraded` warn-once** (lines 43-47) + `close` (lines 52-54).
- [ ] **`record_click` with `article_id`** (line 102) — the click-boost key.
- [ ] **`click_signals`** (lines 111-129): no raw → None; below
      `CLICK_BOOST_MIN_CLICKS` → None; success dict build. **ERROR PATH —
      Redis down → degraded → None.**
- [ ] **`_i`/`_f` malformed value branches** (lines 135-136, 142-143).
      **ERROR PATH — malformed Redis counters.**

---

## app/main.py — 72%

- [ ] **`lifespan` startup + teardown** (lines 64-96): model/qdrant/llm init,
      chat+auth store connect, `init_fixer`, `retention_loop` cancel, cache /
      analytics / cost-budget close. **ERROR PATH — any startup dependency
      failing.**
- [ ] **`_effective_intent` month-scoped branch** (line 242): `range_query_topic`
      non-empty rewrite.
- [ ] **`_retrieval_queries` year-in-review two-leg branch** (lines 270-272).
- [ ] **`_embed_sparse`** (line 298) — run inside `asyncio.to_thread`.
- [ ] **`hybrid_search` vector-cache miss → encode + Qdrant query** (lines
      315-345) including the `inference_lock` serialization.
- [ ] **`body_rescue`** (lines 427-447): empty articles; score above threshold;
      no content tokens; articles without bodies; rerank + max().  **ERROR
      PATH — reranker unavailable.**
- [ ] **`_attach_bodies`** (lines 482-492): empty ids; Qdrant retrieve; payload
      shape handling. **ERROR PATH — Qdrant down / malformed payload.**
- [ ] **`_retrieval_leg` query expansion** (lines 500-502).
- [ ] **`search` endpoint full path** (lines 554-584): cache hit vs miss;
      `record_search` calls; click-boost and diversity wiring (lines 573-577).
      **ERROR PATH — Qdrant/Redis down surfacing as 500s here.**
- [ ] **`source_context` author/industry/dealtype branches** (lines 593-598).
- [ ] **`_facet_values`** (lines 618-626) and **`facets` cache hit/miss** (lines
      633-642). **ERROR PATH — Qdrant facet API down.**
- [ ] **`analytics_click`** (line 656) and **`get_analytics_summary`** (line 666).

---

## app/chat.py — 84%

- [ ] **`ChatStore.connect` schema migration** (lines 147-152): legacy DBs
      missing `prompt_tokens`/`latency_ms` columns. **ERROR PATH — malformed /
      legacy SQLite schema.**
- [ ] **`ChatStore.close`** (lines 155-158).
- [ ] **`rename_session` when session missing** (line 269, 282).
- [ ] **`json_loads` malformed JSON** (lines 399-400). **ERROR PATH — malformed
      stored rows.**
- [ ] **`_row_to_message`** (lines 403-414) — malformed/legacy row field
      coercion (`or 0` fallbacks).
- [ ] **`_smalltalk_reply` non-smalltalk fallthrough** (line 436) and empty
      query.
- [ ] **dataviz helpers edge branches**: `_as_float` non-numeric string (line
      491); `_missing_cell` token set (line 499); `_valid_value_column` (line
      528); `_has_label_content` (line 541); `parse_dataviz` rejection paths
      (lines 561, 565, 567); `_sanitize_dataviz` malformed fence (line 593).
- [ ] **`_dataviz_nudge` view pinning** (line 661).
- [ ] **`_parse_dataviz_with_view` non-dict/invalid** (lines 712, 715-716, 718).
- [ ] **`_apply_requested_view` rewrite** (line 734).
- [ ] **`_is_ranking_refusal`** (line 776).
- [ ] **`_answer_ranked` ranking-nudge retry** (lines 799-808) and the
      `LLMUnavailableError` guard.
- [ ] **`_prepare_turn` follow-up inheritance** (lines 856-859, 866, 876, 879):
      vague-follow-up with year range; `body_rescue` branch; no-sources; weak
      fallback. **ERROR PATH — LLM/Qdrant/Redis down during retrieval.**
- [ ] **`_run_turn` cost recording + finalize** (lines 917-920).
- [ ] **`send_message` `BudgetExceeded` → 429** (lines 1091-1095) and
      **`LLMUnavailableError` → 503** (lines 1096-1100). **ERROR PATH — LLM
      retry exhaustion / daily budget.**
- [ ] **`send_message_stream` nudge retry failure branches** (lines 1169-1176,
      1181-1188) and the **`error` SSE handlers** (lines 1213, 1216-1218).
      **ERROR PATH — LLMUnavailableError / BudgetExceeded mid-stream.**
- [ ] **`retention_loop`** (lines 1225-1233): purge + exception swallow.

---

## app/auth.py — 91%

- [ ] **`verify_password` `ValueError` branch** (lines 155-156). **ERROR PATH —
      malformed stored password hash.**
- [ ] **`create_user` non-unique rollback** (lines 264-266). **ERROR PATH —
      SQLite write-lock / constraint failure.**
- [ ] **`update_user` empty no-op** (line 293) and **`delete_user`** (lines
      299-301).
- [ ] **`issue_token` error rollback** (lines 321-323). **ERROR PATH — SQLite
      write failure.**
- [ ] **`_require_auth_store` uninitialized → 503** (line 350).
- [ ] **`_client_ip` X-Forwarded-For branch** (line 372).
- [ ] **`get_user` endpoint** (line 548).
- [ ] **`patch_user` last-admin guard** (lines 565-566) and **`delete_user`
      last-admin guard** (lines 582-585). **ERROR PATH — self-lockout
      protection.**
- [ ] **`bootstrap_admin` write-lock retry loop** (lines 620-624). **ERROR
      PATH — concurrent worker bootstrap.**

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

1. **app/main.py `hybrid_search` / `_attach_bodies` / `search` / `facets`** —
   Qdrant down and Redis down surfaces as 500s.
2. **app/health.py** — the readiness probes that detect Qdrant/Redis/LLM
   outages.
3. **app/redis_cache.py + app/analytics.py + app/cost_budget.py** — Redis down
   degrade paths (warn-once, in-memory fallback, fail-open budget).
4. **app/chat.py stream error SSEs + BudgetExceeded/LLMUnavailable HTTP paths.**
5. **app/encoders.py / app/query_fix.py** — startup model-load failure fallbacks.
6. **app/chat.py + app/auth.py SQLite layers** — malformed/legacy row handling
   and write-lock/concurrency paths.

`app/llm.py` (was 32%) is now 97% via `tests/test_llm.py`, and `app/reranker.py`
(was 17%) is now 100% via `tests/test_reranker.py`.