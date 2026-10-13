


























set -euo pipefail

PROD_TREE="${PROD_TREE:-/home/ubuntu/search-nlp-rag}"
QDRANT_PORT_A=16333
QDRANT_PORT_B=16334
REDIS_PORT=16379
API_PORT=8099
NEXT_PORT=3099
QDRANT_COLLECTION="e2e_vccircle_articles"
E2E_ADMIN_EMAIL="e2e-admin@e2e.local"
E2E_ADMIN_PASSWORD="E2eAdmin42"
KEEP=0
for arg in "$@"; do [ "$arg" = "--no-teardown" ] && KEEP=1; done


log()  { printf '[e2e %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[e2e %s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }




_STAGE_START=0
_STAGE_LABEL=""
mark_stage() {
  local now="$1"
  now="$(date +%s)"
  if [ "$_STAGE_START" -gt 0 ] && [ -n "$_STAGE_LABEL" ]; then
    printf '[e2e %s] stage done in %ss: %s\n' "$(date +%H:%M:%S)" "$((now - _STAGE_START))" "$_STAGE_LABEL"
  fi
  _STAGE_START="$now"
  _STAGE_LABEL="$1"
}
RUN_START="$(date +%s)"
report_stages() {
  local now
  now="$(date +%s)"
  [ -n "$_STAGE_LABEL" ] && printf '[e2e %s] stage done in %ss: %s\n' "$(date +%H:%M:%S)" "$((now - _STAGE_START))" "$_STAGE_LABEL"
  printf '[e2e %s] total run %ss\n' "$(date +%H:%M:%S)" "$((now - RUN_START))"
}






setup_node_path() {
  local node_bin="$HOME/.nvm/versions/node/v24.19.0/bin"
  [ -d "$node_bin" ] && export PATH="$node_bin:$PATH"
  command -v node >/dev/null 2>&1 || fail "node not found on box (nvm v24.19.0 missing?)"
  command -v npm  >/dev/null 2>&1 || fail "npm not found on box"
  command -v npx  >/dev/null 2>&1 || fail "npx not found on box"
  log "node: $(node --version)  npm: $(npm --version)  npx: $(npx --version)"
}

port_free() {





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

  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  rm -f "$pf"
}


wait_tcp() {
  local p="$1" label="$2" tries="${3:-60}" i
  for i in $(seq 1 "$tries"); do
    if timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$p" 2>/dev/null; then return 0; fi
    sleep 1
  done
  fail "$label never became reachable on port $p (tried $tries times)"
}

wait_http() {
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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



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




REMOTE_URL="$(git -C "$PROD_TREE" remote get-url origin)" || fail "cannot read origin remote of $PROD_TREE"
git clone --quiet "$REMOTE_URL" "$SCRATCH/clone" || fail "git clone from origin failed"
git -C "$SCRATCH/clone" checkout --quiet -B e2e-run origin/main 2>/dev/null \
  || git -C "$SCRATCH/clone" checkout --quiet -B e2e-run main 2>/dev/null \
  || fail "cannot set scratch clone HEAD to origin/main"
log "scratch clone HEAD: $(git -C "$SCRATCH/clone" rev-parse --short HEAD)  branch: $(git -C "$SCRATCH/clone" branch --show-current)"
APP="$SCRATCH/clone"


APPDIR="$APP/e2e"
mkdir -p "$APPDIR"
cp -R "$SCRIPT_DIR"/. "$APPDIR/"
rm -rf "$APPDIR/node_modules"
log "e2e suite copied into scratch: $(ls "$APPDIR" | tr '\n' ' ')"
mark_stage "qdrant + redis containers"



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





REDIS_IMAGE="$(sed -nE 's/^REDIS_IMAGE="\$\{REDIS_IMAGE:-(.*)\}"$/\1/p' "$APP/setup.sh" | head -1 || true)"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine@sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499}"
REDIS_NAME="e2e-redis-$RANDOM-$$"
log "starting scratch redis: $REDIS_NAME (image $REDIS_IMAGE)"
docker run -d --name "$REDIS_NAME" \
  -p "127.0.0.1:$REDIS_PORT:6379" \
  "$REDIS_IMAGE" >/dev/null || fail "docker run redis failed"
wait_tcp "$REDIS_PORT" "scratch redis" 60
mark_stage "corpus fetch + backend venv"







TRIM="${E2E_TRIM_LINES:-5000}"
mkdir -p "$APP/backend/data"
head -n "$TRIM" "$PROD_TREE/backend/data/articles.jsonl" > "$APP/backend/data/articles.jsonl" \
  || fail "cannot read corpus at $PROD_TREE/backend/data/articles.jsonl"
CORPUS_LINES="$(wc -l < "$APP/backend/data/articles.jsonl")"
log "corpus: $CORPUS_LINES lines (trimmed to $TRIM for a fast scratch index)"

log "creating scratch backend venv + installing requirements (may take a while)..."


PY_BIN="$(command -v python3.11 || command -v python3)"
"$PY_BIN" -m venv "$APP/backend/venv" || fail "venv creation failed"
"$APP/backend/venv/bin/pip" install --quiet --disable-pip-version-check \
  -r "$APP/backend/requirements.txt" || fail "pip install -r backend/requirements.txt failed"
mark_stage "index build"


log "building scratch index into $QDRANT_COLLECTION ..."
( cd "$APP/backend" && \
  QDRANT_URL="http://127.0.0.1:$QDRANT_PORT_A" \
  QDRANT_COLLECTION="$QDRANT_COLLECTION" \
  "$APP/backend/venv/bin/python" scripts/build_index.py ) \
  || fail "index build failed (broken indexer / model download)"
log "index build complete"
mark_stage "backend start + search assert"


COUNT="$(curl -fsS --max-time 10 "http://127.0.0.1:$QDRANT_PORT_A/collections/$QDRANT_COLLECTION" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["points_count"])')"
log "scratch index assert: points_count=$COUNT corpus_lines=$CORPUS_LINES"
[ "$COUNT" -ge 1 ] && [ "$COUNT" -le "$CORPUS_LINES" ] \
  || fail "index assert: collection has $COUNT points (expected 1..$CORPUS_LINES) — indexer broken"



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


BACKEND_PID="$SCRATCH/backend.pid"
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


N_RESULTS="$(curl -fsS -G --max-time 60 "http://127.0.0.1:$API_PORT/search" \
  --data-urlencode "q=$E2E_SEARCH_QUERY" \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["results"]))')"
log "search assert: /search?q=$E2E_SEARCH_QUERY returned $N_RESULTS result(s)"
mark_stage "frontend (npm ci + build + start)"
[ "$N_RESULTS" -ge 1 ] \
  || fail "search assert: /search?q=$E2E_SEARCH_QUERY returned 0 results — retrieval broken"


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




export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"



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


export E2E_API_BASE E2E_ADMIN_EMAIL E2E_ADMIN_PASSWORD E2E_SEARCH_QUERY
log "running: npx playwright test (in $APPDIR)"


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

