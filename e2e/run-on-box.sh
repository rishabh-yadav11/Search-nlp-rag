#!/usr/bin/env bash
# =====================================================================
# run-on-box.sh — E2E orchestrator for the VCCircle hybrid-search RAG app.
#
# Runs ON the production box via ssh (fully isolated scratch stack; the
# running deployment, pm2, nginx and prod data are never touched):
#
#   ssh vccircle-search 'bash -s' < e2e/run-on-box.sh                  # default: teardown
#   ssh vccircle-search 'bash -s -- --no-teardown' < e2e/run-on-box.sh  # keep the scratch stack
#
# The script expects to be MET on the box as a file whose sibling directory
# contains the whole e2e suite (package.json, playwright.config.ts, tests/).
# It copies that suite into the scratch clone so playwright runs against a
# git-HEAD-of-origin/main tree PLUS the suite (which is not yet committed).
#
# Order of operations (each asserts its own result before the next):
#   a. scratch clone of origin/main      (isolated, never prod)
#   b. throwaway qdrant on 16333/16334   (pinned digest from setup.sh) + volume
#   c. scratch index build               (scratch qdrant collection + data dir)
#   d. assert index usable (collection count ~ corpus; then /search >=1 result)
#   e. scratch backend (uvicorn :8099, no LLM key, redis pointed at an unused port)
#   f. scratch frontend (npm ci, next build, next start :3099)
#   g. Playwright chromium into the box user cache
#   h. npx playwright test  (exit code reflects the suite)
#   i. teardown trap: kill scratch daemons, rm scratch qdrant container+volume,
#      rm -rf the scratch dir (--no-teardown keeps it and prints its path)
# =====================================================================
set -euo pipefail

PROD_TREE="${PROD_TREE:-/home/ubuntu/search-nlp-rag}"
QDRANT_PORT_A=16333   # qdrant REST (6333 inside the container)
QDRANT_PORT_B=16334   # qdrant gRPC (6334 inside the container)
REDIS_PORT=16379      # scratch redis (6379 inside the container)
API_PORT=8099         # scratch backend
NEXT_PORT=3099        # scratch frontend (dev prod uses different ports; these are scratch-only)
QDRANT_COLLECTION="e2e_vccircle_articles"
E2E_ADMIN_EMAIL="e2e-admin@e2e.local"
E2E_ADMIN_PASSWORD="E2eAdmin42"     # backend policy: letter + digit, >=8
KEEP=0
for arg in "$@"; do [ "$arg" = "--no-teardown" ] && KEEP=1; done

# ---------------- helpers -------------------------------------------------
log()  { printf '[e2e %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[e2e %s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Per-stage wall clock. mark_stage "label" starts a timer for that stage; the
# NEXT mark_stage call prints the completed stage's elapsed time. The final
# report_stage() prints the last stage + total elapsed.
_STAGE_START=0
_STAGE_LABEL=""
mark_stage() { # $1 = label of the stage starting now
  local now="$1"
  now="$(date +%s)"
  if [ "$_STAGE_START" -gt 0 ] && [ -n "$_STAGE_LABEL" ]; then
    printf '[e2e %s] stage done in %ss: %s\n' "$(date +%H:%M:%S)" "$((now - _STAGE_START))" "$_STAGE_LABEL"
  fi
  _STAGE_START="$now"
  _STAGE_LABEL="$1"
}
RUN_START="$(date +%s)"
report_stages() { # final: last stage + total wall time
  local now
  now="$(date +%s)"
  [ -n "$_STAGE_LABEL" ] && printf '[e2e %s] stage done in %ss: %s\n' "$(date +%H:%M:%S)" "$((now - _STAGE_START))" "$_STAGE_LABEL"
  printf '[e2e %s] total run %ss\n' "$(date +%H:%M:%S)" "$((now - RUN_START))"
}

# Ensure the box's node/npm (nvm v24.19.0 per AGENTS.md) is before anything
# else on PATH, so npm ci / next / npx playwright all use it in non-interactive
# ssh. Deliberately NOT sourcing nvm.sh: its `nvm use` deep-dives can kill a
# `set -e` script before node is even checked. The version path is fixed and
# verified on the deploy box.
setup_node_path() {
  local node_bin="$HOME/.nvm/versions/node/v24.19.0/bin"
  [ -d "$node_bin" ] && export PATH="$node_bin:$PATH"
  command -v node >/dev/null 2>&1 || fail "node not found on box (nvm v24.19.0 missing?)"
  command -v npm  >/dev/null 2>&1 || fail "npm not found on box"
  command -v npx  >/dev/null 2>&1 || fail "npx not found on box"
  log "node: $(node --version)  npm: $(npm --version)  npx: $(npx --version)"
}

port_free() {
  # 1 if a probe connects to the port (busy => stale stack), else 0. The probe
  # runs in a child bash under `timeout` so a hung/failing /dev/tcp connect can
  # never trip the parent's `set -e` (observed on the box: the old
  # `if (exec 3<>...)` form terminated the script instead of evaluating the
  # condition) and a firewall that drops SYN counts as free after 2s.
  if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null; then
    echo "ERROR: scratch port $1 already in use — a stale stack may be running." >&2
    return 1
  fi
  return 0
}

kill_pid_file() {
  local pf="$1" pid
  [ -f "$pf" ] || return 0
  pid="$(cat "$pf" 2>/dev/null || true)"
  [ -n "$pid" ] || return 0
  # Negative pid == the process GROUP (setsid-launched daemons below), killing npm/node + uvicorn.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  rm -f "$pf"
}

# TCP-level wait (redis has no HTTP endpoint; wait_http would spin forever).
wait_tcp() { # $1 = port, $2 = label, $3 = max tries
  local p="$1" label="$2" tries="${3:-60}" i
  for i in $(seq 1 "$tries"); do
    if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$p" 2>/dev/null; then return 0; fi
    sleep 1
  done
  fail "$label never became reachable on port $p (tried $tries times)"
}

wait_http() { # $1 = url, $2 = label, $3 = max tries
  local url="$1" label="$2" tries="${3:-60}" i
  for i in $(seq 1 "$tries"); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  fail "$label never became reachable at $url (tried $tries times)"
}

teardown() {
  set +e
  if [ "$KEEP" -eq 1 ]; then
    echo "[e2e] --no-teardown: scratch stack kept at:" >&2
    echo "  $SCRATCH" >&2
    [ -n "${QDRANT_NAME:-}" ] && echo "  qdrant container=$QDRANT_NAME volume=$QDRANT_VOL" >&2
    [ -n "${REDIS_NAME:-}" ] && echo "  redis container=$REDIS_NAME" >&2
    echo "  stop with:  docker rm -f $QDRANT_NAME; docker volume rm $QDRANT_VOL; kill -- -\$(cat $SCRATCH/backend.pid) -\$(cat $SCRATCH/frontend.pid)" >&2
    return 0
  fi
  [ -n "${BACKEND_PID:-}" ] && kill_pid_file "$BACKEND_PID"
  [ -n "${FRONTEND_PID:-}" ] && kill_pid_file "$FRONTEND_PID"
  [ -n "${QDRANT_NAME:-}" ] && { docker rm -f "$QDRANT_NAME" >/dev/null 2>&1; log "removed qdrant $QDRANT_NAME"; }
  [ -n "${QDRANT_VOL:-}" ] && { docker volume rm "$QDRANT_VOL" >/dev/null 2>&1; log "removed qdrant volume $QDRANT_VOL"; }
  [ -n "${REDIS_NAME:-}" ] && { docker rm -f "$REDIS_NAME" >/dev/null 2>&1; log "removed redis $REDIS_NAME"; }
  [ -n "${SCRATCH:-}" ] && rm -rf "$SCRATCH"
}
trap teardown EXIT INT TERM

# ---------------- (a) scratch clone ---------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Guard: when this script is piped via `bash -s < ...` (stdin), BASH_SOURCE[0]
# is empty and SCRIPT_DIR resolves to the box's cwd — copying the whole home
# dir into scratch. Require the sibling test files instead.
for f in playwright.config.ts tests; do
  [ -e "$SCRIPT_DIR/$f" ] || fail "run this as a FILE on the box whose sibling directory holds the e2e suite (not via stdin). Staging: scp -r e2e vccircle-search:~/"
done
setup_node_path

[ -d "$PROD_TREE/.git" ] || fail "production checkout not found at $PROD_TREE"
port_free "$QDRANT_PORT_A" || exit 1
port_free "$QDRANT_PORT_B" || exit 1
port_free "$REDIS_PORT" || exit 1
port_free "$API_PORT" || exit 1
port_free "$NEXT_PORT" || exit 1

mark_stage "setup + scratch clone"
SCRATCH="${HOME}/scratch-e2e-$RANDOM-$$"
mkdir -p "$SCRATCH"
log "scratch root: $SCRATCH"

log "cloning origin/main (HEAD) into scratch (read-only on the prod tree)..."
git clone --quiet --no-tags "$PROD_TREE" "$SCRATCH/clone" || fail "git clone failed"
git -C "$SCRATCH/clone" fetch --quiet origin 2>/dev/null || true
git -C "$SCRATCH/clone" checkout --quiet -B e2e-run origin/main 2>/dev/null \
  || git -C "$SCRATCH/clone" checkout --quiet -B e2e-run main 2>/dev/null \
  || fail "cannot set scratch clone HEAD to origin/main"
log "scratch clone HEAD: $(git -C "$SCRATCH/clone" rev-parse --short HEAD)  branch: $(git -C "$SCRATCH/clone" branch --show-current)"
APP="$SCRATCH/clone"

# Carry this e2e suite into the scratch clone (origin/main has no e2e yet).
APPDIR="$APP/e2e"
mkdir -p "$APPDIR"
cp -R "$SCRIPT_DIR"/. "$APPDIR/"
rm -rf "$APPDIR/node_modules"
log "e2e suite copied into scratch: $(ls "$APPDIR" | tr '\n' ' ')"
mark_stage "qdrant + redis containers"

# ---------------- (b) throwaway qdrant on scratch ports --------------------
# The pinned digest is read straight out of setup.sh (never hand-copied here).
QDRANT_IMAGE="$(sed -nE 's/^QDRANT_IMAGE="\$\{QDRANT_IMAGE:-(.*)\}"$/\1/p' "$APP/setup.sh" | head -1 || true)"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.19.0@sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc}"
QDRANT_NAME="e2e-qdrant-$RANDOM-$$"
QDRANT_VOL="e2e-qdrant-$RANDOM-$$-vol"
log "starting scratch qdrant: $QDRANT_NAME (image $QDRANT_IMAGE)"
docker run -d --name "$QDRANT_NAME" \
  -p "127.0.0.1:$QDRANT_PORT_A:6333" \
  -p "127.0.0.1:$QDRANT_PORT_B:6334" \
  -v "$QDRANT_VOL:/qdrant/storage" \
  "$QDRANT_IMAGE" >/dev/null || fail "docker run qdrant failed"
wait_http "http://127.0.0.1:$QDRANT_PORT_A/healthz" "scratch qdrant" 120

# Scratch redis on a scratch port. The backend NEEDS redis for its rate
# limiter (a down redis 503s /search with "Rate limiter unavailable" — only
# the HybridCache degrades to memory) and for the HybridCache, so this is not
# optional. Ephemeral: no volume, no persisted data.
REDIS_IMAGE="$(sed -nE 's/^REDIS_IMAGE="\$\{REDIS_IMAGE:-(.*)\}"$/\1/p' "$APP/setup.sh" | head -1 || true)"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine@sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499}"
REDIS_NAME="e2e-redis-$RANDOM-$$"
log "starting scratch redis: $REDIS_NAME (image $REDIS_IMAGE)"
docker run -d --name "$REDIS_NAME" \
  -p "127.0.0.1:$REDIS_PORT:6379" \
  "$REDIS_IMAGE" >/dev/null || fail "docker run redis failed"
wait_tcp "$REDIS_PORT" "scratch redis" 60
mark_stage "corpus fetch + backend venv"

# ---------------- (c+1/2) corpus + scratch venv ----------------------------
# The corpus is git-ignored, so the prod tree is the only local source. We only
# COPY a trimmed prefix (read-only on prod); the scratch clone owns its data/.
# A full 67k-article index takes ~an hour of CPU embedding time, which overruns
# the one-session ssh budget; the E2E wiring (SSR, search, auth, chat, /for-you,
# analytics) is identical on a 5000-article subset and finishes in minutes.
TRIM="${E2E_TRIM_LINES:-5000}"
mkdir -p "$APP/backend/data"
head -n "$TRIM" "$PROD_TREE/backend/data/articles.jsonl" > "$APP/backend/data/articles.jsonl" \
  || fail "cannot read corpus at $PROD_TREE/backend/data/articles.jsonl"
CORPUS_LINES="$(wc -l < "$APP/backend/data/articles.jsonl")"
log "corpus: $CORPUS_LINES lines (trimmed to $TRIM for a fast scratch index)"

log "creating scratch backend venv + installing requirements (may take a while)..."
# Prefer 3.11 (CI/python target, README-documented) when the box has it; fall
# back to the default python3.
PY_BIN="$(command -v python3.11 || command -v python3)"
"$PY_BIN" -m venv "$APP/backend/venv" || fail "venv creation failed"
"$APP/backend/venv/bin/pip" install --quiet --disable-pip-version-check \
  -r "$APP/backend/requirements.txt" || fail "pip install -r backend/requirements.txt failed"
mark_stage "index build"

# ---------------- (c) scratch index build ----------------------------------
log "building scratch index into $QDRANT_COLLECTION ..."
( cd "$APP/backend" && \
  QDRANT_URL="http://127.0.0.1:$QDRANT_PORT_A" \
  QDRANT_COLLECTION="$QDRANT_COLLECTION" \
  "$APP/backend/venv/bin/python" scripts/build_index.py ) \
  || fail "index build failed (broken indexer / model download)"
log "index build complete"
mark_stage "backend start + search assert"

# ---------------- (d) assert the index is usable (fast fail) ---------------
COUNT="$(curl -fsS --max-time 10 "http://127.0.0.1:$QDRANT_PORT_A/collections/$QDRANT_COLLECTION" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["points_count"])')"
log "scratch index assert: points_count=$COUNT corpus_lines=$CORPUS_LINES"
[ "$COUNT" -ge 1 ] && [ "$COUNT" -le "$CORPUS_LINES" ] \
  || fail "index assert: collection has $COUNT points (expected 1..$CORPUS_LINES) — indexer broken"

# Derive a guaranteed-hit query from the corpus itself (title of the first
# article), so retrieval must return >=1 result under /search.
E2E_SEARCH_QUERY="$(
  python3 - "$APP/backend/data/articles.jsonl" <<'PY'
import json, re, sys
topic = "Ola Electric IPO"
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        title = (d.get("title") or "").strip()
        if title:
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9&\.\-\x27]*", title)
            topic = " ".join(words[:4]) if words else title[:60]
            break
print(topic)
PY
)"
E2E_SEARCH_QUERY="${E2E_SEARCH_QUERY:-Ola Electric IPO}"
log "guaranteed-hit query: $E2E_SEARCH_QUERY"

# ---------------- (e) scratch backend on :8099 ------------------------------
BACKEND_PID="$SCRATCH/backend.pid"   # unset until start; keep SCRATCH for teardown
BACKEND_LOG="$APP/backend-uvicorn.log"
E2E_API_BASE="http://127.0.0.1:$API_PORT"
(
  cd "$APP/backend" &&
  QDRANT_URL="http://127.0.0.1:$QDRANT_PORT_A" \
  QDRANT_COLLECTION="$QDRANT_COLLECTION" \
  REDIS_URL="redis://127.0.0.1:$REDIS_PORT/0" \
  ALLOWED_HOSTS="localhost,127.0.0.1" \
  CORS_ORIGINS="http://127.0.0.1:$NEXT_PORT,http://localhost:$NEXT_PORT" \
  AUTH_COOKIE_SECURE=false \
  GEMINI_API_KEY="" \
  AUTH_DB_PATH="$APP/backend/data/auth.db" \
  CHAT_DB_PATH="$APP/backend/data/chat.db" \
  AUTH_ADMIN_EMAIL="$E2E_ADMIN_EMAIL" AUTH_ADMIN_PASSWORD="$E2E_ADMIN_PASSWORD" \
  exec setsid "$APP/backend/venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$API_PORT" \
  >"$BACKEND_LOG" 2>&1 < /dev/null & echo $! > "$BACKEND_PID"
)
wait_http "http://127.0.0.1:$API_PORT/health" "scratch backend" 120
log "scratch backend healthy on $E2E_API_BASE (no LLM key, scratch redis on $REDIS_PORT)"

# Search assertion (index usable end-to-end, before the UI runs).
N_RESULTS="$(curl -fsS -G --max-time 60 "http://127.0.0.1:$API_PORT/search" \
  --data-urlencode "q=$E2E_SEARCH_QUERY" \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["results"]))')"
log "search assert: /search?q=$E2E_SEARCH_QUERY returned $N_RESULTS result(s)"
mark_stage "frontend (npm ci + build + start)"
[ "$N_RESULTS" -ge 1 ] \
  || fail "search assert: /search?q=$E2E_SEARCH_QUERY returned 0 results — retrieval broken"

# ---------------- (f) scratch frontend on :3099 -----------------------------
log "npm ci of frontend deps..."
( cd "$APP/frontend" && npm ci ) || fail "npm ci frontend failed"
log "next build..."
( cd "$APP/frontend" && NEXT_PUBLIC_API_BASE="$E2E_API_BASE" npm run build ) \
  || fail "next build failed"
FRONTEND_PID="$SCRATCH/frontend.pid"
FRONTEND_LOG="$APP/frontend.log"
(
  cd "$APP/frontend" && \
  NEXT_PUBLIC_API_BASE="$E2E_API_BASE" \
  exec setsid npm run start -- -p "$NEXT_PORT" \
  >"$FRONTEND_LOG" 2>&1 < /dev/null & echo $! > "$FRONTEND_PID"
)
wait_http "http://127.0.0.1:$NEXT_PORT/" "scratch frontend" 120
log "scratch frontend healthy on http://127.0.0.1:$NEXT_PORT (API base $E2E_API_BASE)"
mark_stage "chromium install"

# ---------------- (g) chromium into the box user cache ----------------------
# Target the box user cache explicitly, in case an orchestrator-wide override
# points elsewhere; the browser goes to ~/.cache/ms-playwright as intended.
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
# Install the project's test deps into $APPDIR/node_modules FIRST. Without them
# `npx playwright` falls back to a cache-downloaded CLI package and its config
# load fails with "Cannot find module '@playwright/test'".
(
  cd "$APPDIR" && npm ci --no-audit --no-fund
) || {
  echo "ERROR: npm ci in the e2e dir failed." >&2
  exit 1
}
(
  cd "$APPDIR" && npx playwright install chromium \
) || {
  echo "ERROR: playwright chromium install failed." >&2
  echo "  The box is probably missing Playwright's system libraries for Chromium." >&2
  echo "  Install libs+e browser with:  npx playwright install --with-deps chromium" >&2
  echo "  (or apt-get install the missing libs listed by the error above)." >&2
  echo "  The test run below will then fail with a readable Playwright error." >&2
}
log "chromium checked into $PLAYWRIGHT_BROWSERS_PATH"
mark_stage "playwright suite"

# ---------------- (h) run the suite ------------------------------------------
export E2E_API_BASE E2E_ADMIN_EMAIL E2E_ADMIN_PASSWORD E2E_SEARCH_QUERY
log "running: npx playwright test (in $APPDIR)"
# Capture playwright's output to a file so a buffering quirk or a mid-run
# connection drop can never swallow the result, then replay it to the operator.
set +e
( cd "$APPDIR" && timeout 1800 npx playwright test ) >"$SCRATCH/playwright.out" 2>&1
PW_RC=$?
set -e
if [ -s "$SCRATCH/playwright.out" ]; then
  echo "----- playwright output (rc=$PW_RC) -----" >&2
  cat "$SCRATCH/playwright.out" >&2
  echo "----- end playwright output -----" >&2
else
  echo "[e2e] ERROR: playwright exited rc=$PW_RC with NO output" >&2
fi
report_stages
if [ "$PW_RC" -ne 0 ]; then
  echo "[e2e] E2E TESTS FAILED (rc=$PW_RC)" >&2
  exit 1
fi
log "E2E SUITE PASSED"
# teardown trap runs here on normal exit (unless --no-teardown)
