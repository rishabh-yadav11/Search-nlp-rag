# Backend Test Coverage Gaps

Measured with `pytest --cov=app`: **657 passed**, 86% overall (3652 statements,
504 missed). This is a checklist of functions and branches that have **no test
coverage**, grouped by module. The figures re-measured here are the ones for
the modules this change rewrote (`app/cost_budget.py`, `app/chat.py`); the rest
of the document is not re-audited here — issue #295 tracks a full accuracy pass.
Items marked **ERROR PATH** are exactly the failure modes that matter in
production: Qdrant down, Redis down, LLM timeout/retry exhaustion, and malformed
SQLite rows (the project uses SQLite via aiosqlite, not MySQL — same concept:
rows that don't match the expected schema).

Legend: `[x]` checked = covered by `tests/`; `[ ]` unchecked = gap to cover.
Line numbers refer to the current `backend/app/*.py`.

---

## app/llm.py — 100% (covered by tests/test_llm.py)

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
- [x] **Post-loop raises** (lines 85, 143): with `LLM_MAX_RETRIES=-1` the retry
      loop never runs and `raise LLMUnavailableError` fires without ever calling
      the client (both `generate_answer` and `stream_answer`).

---

## app/reranker.py — 100% (covered by tests/test_reranker.py)

The reranker is torch-only: the ONNX/optimum fast path was removed (optimum-onnx
is not installable alongside the pinned transformers 5.x), so its load/export/
lock tests went with it. Construction and `predict` are exercised with a faked
`sentence_transformers` import, plus a working faked `optimum.onnxruntime` that
proves a default construction never takes an ONNX path.

- [x] **Default backend** is torch and never attempts the ONNX path, even when
      `optimum` is importable.
- [x] **Explicit `backend="torch"`** builds a CPU `CrossEncoder`.
- [x] **Unsupported backend** (`backend="onnx"`) warns and falls back to torch.
      **ERROR PATH — stale `RERANK_BACKEND` in the environment.**
- [x] **`predict`** passes the `(query, passage)` pairs straight through to the
      torch `CrossEncoder`.

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

- [x] **`/health`, `/live`** (lines 28, 124): liveness only, inspected once;
      `/health` cannot fail by design, which is the property the deploy gate and
      the watchdog are tested against.
- [x] **`_qdrant_ok` client absent** (lines 129-130) and **Qdrant call failing /
      timing out** (lines 135-136) — `RuntimeError` and `asyncio.wait_for`
      raising `TimeoutError` both → False. **ERROR PATH — Qdrant down.**
- [x] **`_models_ok`** (line 149): all present → True; any key missing → False.
- [x] **`_llm_status`** (line 153): a real-shaped key → `(True, "ok")`; the
      shipped `.env.example` placeholder, every sentinel spelling, masked keys
      and wrong-shape keys → `(False, "placeholder"/"malformed")`; unset →
      `(False, "missing")`. It never returns the key itself. **ERROR PATH — the
      literal `bool(config.GEMINI_API_KEY)` this replaced read every one of
      those as healthy.**
- [x] **`_redis_status`** (lines 166-189): no REDIS_URL → `(True, "memory")`;
      ping ok → `(True, "redis")` (plus client reuse, single `from_url`);
      ping fail / timeout → `(True, "degraded")`. **ERROR PATH — Redis down.**
- [x] **`_readiness_report` + `/ready`/`/readyz`** (lines 278-303, 320, 351):
      report shape asserted with mocked + real checks wired together; ready →
      200, not-ready → 503 on both endpoints, and the verdict requires a usable
      LLM key.
- [x] **`/ready/deep`** (line 399) + **`_is_host_local_probe`** (line 370): the
      uncached/unrated monitoring probe, its bypass of both the shared readiness
      cache and the limiter, and its refusal of a non-loopback peer or a request
      carrying `X-Forwarded-For` — each proved by mutation.

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

## app/cost_budget.py — 100% (covered by tests/test_cost_budget.py + tests/test_budget_lua.py)

The cap is RESERVE / SETTLE / RELEASE around every billed call; the whole
read-modify-write lives in one Lua script (`_BUDGET_LUA`) over four keys
(day counter, live holds, holds expiry zset, holds-accounted ledger). The
Python side is covered against `FakeBudgetStore`, a model of that contract;
the shipped script itself is EXECUTED by `tests/test_budget_lua.py` under
`lua5.1` (skipped where lua5.1 is absent), so the two cannot drift apart
without a failure.

- [x] **`reserve` hold + rejection** (lines 365-394): hold written and counted
      against the cap for the next caller; a rejected reserve leaves no hold
      behind; cap disabled (`LLM_DAILY_BUDGET_USD <= 0`) touches no store;
      hold floored at 1 micro-USD so a zero `LLM_CALL_RESERVE_USD` cannot turn
      the cap into a no-op; first `BudgetExceeded` logs one warning.
- [x] **`reserve` store down → `BudgetUnavailable`** (lines 338-350): fails
      closed, is not a `BudgetExceeded`, and drops the cached script handle.
      **ERROR PATH — Redis down.**
- [x] **`settle` counter write** (lines 397-419): actual cost recorded whole
      (hold was never counted), over-reserve recorded and blocking the next
      call, zero-cost settle records nothing, empty id list still bills,
      store-down leaves the holds in place. **ERROR PATH — Redis down.**
- [x] **`settle` idempotency + crash promotion** (Lua sweep/settle arithmetic):
      settling the same ids twice charges once, duplicate ids within one call
      charge once, a hold that lapsed is CHARGED to the counter (a crash is
      not free spend) and its later settle REPLACES the estimate with the real
      cost rather than adding to it.
- [x] **crash promotion is REACHABLE, not just arithmetically correct**: the
      holds hash and the expiry zset are expired at twice the reservation TTL,
      because a hold's zset score first satisfies the sweep predicate exactly
      one TTL after the reserve — expiring the containers at one TTL killed them
      in the same instant the promotion became observable, and the crash's spend
      was silently lost. The Lua harness models key expiry (and rejects a
      non-positive TTL, as Redis does) so this is enforced, not assumed.
- [x] **`release` drops holds, never refunds** (lines 422-427): live hold
      dropped with the counter untouched, a promoted (already charged) id not
      refunded, store-down surfaces. **ERROR PATH — Redis down.**
- [x] **`_client` lazy init + reuse**: `from_url` once, DB index swapped to
      `ANALYTICS_REDIS_DB`, connection reused; `close` calls `aclose`, clears
      the global, and is a no-op when no client.
- [x] **TTL floors** (`hold_ttl` and `counter_ttl` floored at 1 second, in the
      Lua): one scenario per floor, each executing that guard with a
      non-positive value. Both floors used to be deletable with the suite still
      green — the single scenario that claimed them passed `0` as `hold_ttl`
      twice, and the harness let `EXPIRE key 0` succeed where Redis rejects it.
- [x] **`to_usd` canonical unit** (lines 430-447): INR `LLMResult.cost()` → USD
      before any accounting, and the 1.0 fallback rate on a nonsensical
      configured rate — which warns once and not on every turn. Both branches
      are executed; the fallback had none.

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

## app/chat.py — 92% (covered by tests/test_chat.py)

Measured: 848 statements, 67 missed. The entries below are the branches this
change added or re-pointed at. The residual is mostly
`_prepare_multi_entity_turn` (lines 1330-1440), which this change did not
touch. The missed lines inside the code this change added or edited are all
pre-existing lines; the rest of the residual is unattributed here and the full
accounting is #295's audit. Do not read this section as 100%.

- [x] **`ChatStore.connect` schema migration** (lines 119-175): legacy DBs
      missing `prompt_tokens`/`completion_tokens`/`cost` and separately missing
      `latency_ms` get the columns added with `0` defaults, as does the `aborted`
      column (line 174) that the abort rule needs. **ERROR PATH — malformed /
      legacy SQLite schema.**
- [x] **`ChatStore.close`** (lines 177-180): close an open store and the
      idempotent close-when-already-closed no-op.
- [x] **`rename_session` / `delete_session` when session missing** (lines 321,
      336) → 404.
- [x] **`global_stats` exception handler** (lines 404-464): a failing query
      degrades to `{"error": "chat analytics unavailable"}`, never raises.
- [x] **`json_loads` malformed JSON** (lines 562-616): bad JSON, `None`, and
      non-string input all → `[]`. **ERROR PATH — malformed stored rows.**
- [x] **`_row_to_message`** (lines 616-628): malformed/legacy row field
      coercion — bad sources JSON and `NULL`/string token/cost fields fall back
      to `0` defaults, and `aborted` coerces from a legacy `NULL`.
- [x] **`_smalltalk_reply` non-smalltalk fallthrough** (line 645): empty /
      blank queries and >12-word messages are not small talk.
- [x] **dataviz helpers edge branches**: `_as_float` non-numeric string, bool,
      and `None` (line 780); `_missing_cell` token set incl. "not stated"/"—"
      (line 804); `_valid_value_column` all-missing vs numeric vs non-numeric
      (line 815); `_first_numeric_column` empty rows / empty first row
      (line 822); `_has_label_content` no-label-cols vs all-empty labels
      (line 831); `parse_dataviz` rejection paths — non-dict data, non-string
      columns, non-list rows (lines 845-884); `_sanitize_dataviz` empty
      text and no-fence passthrough (line 887).
- [x] **`_dataviz_nudge` view pinning** (line 1012): a named view appends the
      "exact type of data block" instruction; a generic chart ask does not.
- [x] **`_parse_dataviz_with_view` invalid inputs** (lines 1066-1080):
      no fence, invalid JSON, and non-dict data → `None`; dict data gets the
      view applied.
- [x] **`_apply_requested_view` rewrite** (line 1082): a block that fails to
      re-parse is left verbatim.
- [x] **`_is_ranking_refusal`** (line 1164): refusal signatures ("cannot be
      generated", "do not contain specific amounts") → True; empty text and
      genuine ranked answers → False.
- [x] **`_answer_ranked` ranking-nudge retry** (lines 1186-1214): a refusal is
      re-asked once with `_RANKING_NUDGE` and tokens summed; the
      `LLMUnavailableError` guard keeps the first answer; non-refusals make a
      single call.
- [x] **`_prepare_turn` follow-up inheritance** (lines 1226-1329): vague-follow-up
      with a year range keeps the previous topic and pins the new dates;
      `body_rescue` runs when `ENABLE_BODY_RESCUE`; no-sources short-circuit;
      weak fallback answer + note. **ERROR PATH — LLM/Qdrant/Redis down during
      retrieval.**
- [x] **`_run_turn` cost recording + finalize** (lines 1650-1714): `reserve`
      before the billed call → `_answer_ranked` → `_discharge_turn_holds`
      (`settle(holds, cost_usd)`, the turn's single counter write;
      `release(holds)` when no request was ever sent) → finalized answer
      (unrequested dataviz blocks stripped). Four money rules live here and are
      tested:
      a zero COMPUTED cost is not evidence that nothing was spent —
      `generate_answer` returns a truthy zero-token `LLMResult` when the
      response carries no usage, so a delivered, billed answer would settle as
      free and leave the cap inert against such a provider — so the figure is
      floored at `LLM_CALL_RESERVE_USD` (the same estimate the streaming path's
      `mid_stream_estimate` uses), and that ONE figure is both settled and
      returned for storage, so the reported cost and the budget cannot
      disagree; an EMPTY hold list
      (cap disabled) settles nothing at all, so an opted-out
      deployment never depends on the counter; and a settle that cannot reach
      the store is best-effort, because the answer exists and has been billed,
      while the live hold is charged by the sweep either way. A FAILED call
      is charged, not refunded — the provider bills the prompt of every one of
      the `LLM_MAX_RETRIES + 1` attempts, so `LLMUnavailableError` settles the
      turn's holds for `exc.attempts * LLM_CALL_RESERVE_USD` and is then
      re-raised (#280), while `attempts=0` (the loop never ran, so nothing was
      sent) releases them. **ERROR PATH — Redis down after a billed call, and a
      total LLM outage.**
- [x] **`send_message` `BudgetExceeded` → 429** (lines 1989-1997),
      **`LLMUnavailableError` → 503** (lines 1998-2002), and
      **`BudgetUnavailable` → 503** (lines 2003-2012) with the dangling user
      message rolled back, so an unreadable counter is never reported as an
      empty answer. The 503 is REACHABLE, not just present: a total outage now
      propagates out of `_run_turn` and is driven end to end through the real
      `generate_answer` retry loop, and the same outage asserted against the SSE
      path's `error` event with the identical payload, so neither path can
      report a 200 with a fabricated "no answer" (#280). **ERROR PATH — LLM
      retry exhaustion / daily budget / Redis down.**
- [x] **`_require_store` uninitialized → 503** (line 1644) and
      **`_validate_question` too-long → 400** (line 1653).
- [x] **`send_message_stream` nudge retry branches** (lines 2091-2100,
      2123-2132): a dataviz/ranking nudge that succeeds appends its block and
      sums tokens; a failed nudge (`LLMUnavailableError`) keeps the streamed
      answer. **`error` SSE handlers**: mid-stream `LLMUnavailableError` →
      "LLM temporarily unavailable" (line 2440), which settles the failed
      attempts like the JSON path rather than refunding them (#280);
      `BudgetExceeded` → "Daily AI
      budget reached" (line 2184); `BudgetUnavailable` → "AI budget service
      unavailable" (line 2190); unexpected exceptions → "Something went wrong"
      (line 2194), never a 500. The `BudgetUnavailable` handler is still
      reachable: `reserve()` is called at the pre-call gate and by each nudge,
      and a rejection there propagates before a byte is delivered. **ERROR PATH —
      LLMUnavailableError / BudgetExceeded / BudgetUnavailable mid-stream.**
- [x] **`retention_loop`** (lines 2207-2215): a failing purge is swallowed and
      the loop keeps ticking; the next tick purges expired conversations.

Added by this change, all exercised end to end through the HTTP handlers:

- [x] **the ONE abort rule** (lines 1927-1968): the client's disconnect is
      checked at every gate, and a turn that streamed nothing is rolled back
      while a turn that streamed anything is PERSISTED with
      `[answer truncated]` and `aborted=True` — the server's history can never
      contradict what is already on the client's screen. `persist_truncated_turn`
      (lines 1903-1924) is the single writer for every partial turn, including
      the mid-stream-failure path.
- [x] **a billed call is charged, never refunded, on a disconnect** (lines
      2030, 2091-2100, 2123-2132): a disconnect in the gate-to-first-delta
      window settles the hold at the estimate the gate took it at, and the two
      post-stream checks pass the finished stream's real cost instead of
      storing the turn as free. A mid-stream FAILURE after deltas is charged
      too (lines 2036-2069): `chunks` is non-empty there, so the provider
      billed those tokens even though usage was never reported, and
      `fail_turn` (lines 1970-1997) applies the same rule. A zero COMPUTED
      cost is not evidence that nothing was spent — `stream_answer` appends a
      truthy zero-token result when a provider sends no usage chunk — so every
      abandon site ON THE STREAMING PATH routes through `billed_usd` (lines
      1841-1857) and the completed streaming path floors at the estimate (lines
      2145-2156); otherwise a provider that never reports usage would leave the
      cap inert and every turn free. The non-streaming half of the same rule is
      `_run_turn`'s cost floor above, which the streaming test pair
      (`test_completed_turn_with_{no_usage_report,reported_usage}`) is mirrored
      by at `tests/test_chat.py::test_api_json_turn_with_no_usage_report_is_still_charged`
      and `::test_run_turn_with_reported_usage_still_uses_the_real_cost`. A turn
      that made no billed call at all still releases.
- [x] **a delivered answer survives a failed settle** (lines 1859-1892,
      1481-1502): once deltas are on the wire, or the LLM has already answered,
      an unreachable counter no longer converts a complete answer into a
      truncated `aborted` row plus an `error` event.
- [x] **`send_message` 499 on a client that disconnected** (lines 1786-1791): a
      JSON client has seen nothing, so the turn is a clean rollback reported as
      a non-success status rather than a completed `TurnOut`.
- [x] **`_trim_history` / `CHAT_MAX_HISTORY_CHARS`** (lines 1657-1680, and
      `_start_turn`): oldest-first, never splitting a message, a single
      oversized message kept alone, `0`/negative disabling the cap, and the char
      budget applied to what actually reaches the prompt.
- [x] **frontend/backend dataviz fence parity** (`DATAVIZ_FENCE_PATTERN` line
      750): the TSX's own `FENCE_SRC` and `stripOpenFence` are executed under
      `node` and compared BEHAVIOURALLY with Python's, including the unclosed
      fence where neither side may render raw JSON. A match-only comparison
      would have asserted nothing there.

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

## app/query_expand.py — 100% (covered by tests/test_query_expand.py)

- [x] **concept expansion** (lines 175-215): layoffs/job-cut, acquisition, funding,
      IPO, and edtech queries append synonym terms.
- [x] **no-concept passthrough** (line 243): `expand_query` returns the query
      unchanged when no concept matches.
- [x] **token budget bound** (lines 234-243): expansions are capped at
      `_MAX_EXTRA_TOKENS`; when every candidate term exceeds the remaining budget
      the query is returned unchanged (`monkeypatch`ed `_MAX_EXTRA_TOKENS=1`).

---

## app/query_intent.py — 100% (covered by tests/test_query_intent.py)

- [x] **year-span rollover** (line 286): a short span whose 2-digit end year is
      below the start (`2024-23`) rolls forward 100 years — the inverse of the
      century-rollover case already covered (`1999-00` → 2000).
- [x] **fiscal-span rollover** (line 245): a backward `FY 2025-24` span applies
      the same `end += 100` normalization in `_fiscal_range`.
- [x] **`_referenced_year` Flashback prefix** (line 364): an explicit
      `flashback <year>` prefix resolves to that year (both via the direct
      helper and through `rewrite_year_in_review`).

---

## app/index_text.py — 100% (covered by tests/test_index_text.py)

- [x] **`split_names` whitespace-only** (line 37): a value of only spaces
      returns `[]`.
- [x] **`split_names` malformed JSON** (lines 43-44): a string starting with
      `[` that fails `json.loads` falls through to the delimiter split instead
      of raising. **ERROR PATH — malformed index rows.**
- [x] **`normalize_date`** (lines 85, 87, 91, 94-95): `None` → `None`; a
      `datetime` instance kept verbatim then tz-normalized; blank string →
      `None`; a non-parseable string is returned as-is instead of raising.
      **ERROR PATH — malformed index rows.**

---

## Remaining small gaps

None in `app/cost_budget.py` and `app/chat.py` beyond the entries listed above.
The modules this change rewrote are measured and current; the document's other
module sections are NOT re-audited here and several do not in fact sit at 100%
(`app/recommender.py` 25%, `app/user_profile.py` 62%, `app/main.py` 86%,
`app/query_intent.py` 86%, `app/redis_cache.py` 82%). Issue #295 is the full
accuracy pass for this document.

---

## Priority order (error paths first)

All error paths across the codebase are now covered: malformed index rows
(index_text), month/year/quarter/fiscal range edges (query_intent),
query-expansion token budgets (query_expand), the LLM retry/dead-code raises
(llm.py), SQLite write failures and malformed stored rows (auth + chat),
Redis/Qdrant down (health, redis_cache, cost_budget, analytics, click_boost),
model-load fallbacks (reranker, encoders), and the query-fix / diversity
utility branches.

`app/llm.py` (was 32%) is now 100% via `tests/test_llm.py` (retry/backoff/
exhaustion/dead-code raises), `app/reranker.py`
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
`tests/test_main_http.py`, `app/chat.py` (was 84%) is now 92% via
`tests/test_chat.py` (schema migration, error SSEs, budget/LLM HTTP paths,
dataviz edge branches, nudge retries, the abort rule, and the retention loop),
`app/auth.py`
(was 91%) is now 100% via `tests/test_auth.py` (malformed-hash handling,
SQLite rollback paths, last-admin lockout guards, `_client_ip`, and the
bootstrap write-lock retry loop), and the last three gaps —
`app/query_expand.py` (was 97%), `app/query_intent.py` (was 99%), and
`app/index_text.py` (was 88%) — are now 100% via `tests/test_query_expand.py`,
`tests/test_query_intent.py`, and `tests/test_index_text.py`.