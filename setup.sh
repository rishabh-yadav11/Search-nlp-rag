#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

QDRANT_PORT="${QDRANT_PORT:-6333}"
REDIS_PORT="${REDIS_PORT:-6379}"
API_PORT="${API_PORT:-8001}"
NEXT_PORT="${NEXT_PORT:-3000}"
PUBLIC_PORT="${PUBLIC_PORT:-8080}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"

VENV="$SCRIPT_DIR/backend/venv"
VENV_PY="$VENV/bin/python"
LOGS="$SCRIPT_DIR/logs"
ENV_FILE="$SCRIPT_DIR/backend/.env"
PID_DIR="$SCRIPT_DIR/.pid"

usage() {
    cat <<'EOF'
usage: setup.sh [stage ...]

stages (run in order):
  deps       ensure node >= 18 (apt, or ~/.local tarball fallback)
  backend    venv + pip deps + .env + Qdrant/Redis docker containers
  index      build the index from MySQL (fetch -> embed -> seed incremental state)
  frontend   npm ci + production build (Next.js)
  services   start gunicorn + next in the background
  cron       install the 15-minute incremental sync
  nginx      write + enable nginx config (public port -> app + API)

  all        deps backend index frontend services cron nginx

env overrides:
  QDRANT_PORT REDIS_PORT API_PORT NEXT_PORT PUBLIC_PORT GUNICORN_WORKERS
  PUBLIC_BASE_URL   e.g. http://your-host:8080 (baked into the Next.js build)
EOF
}

stage() { echo; echo "==> $*"; }

have() { command -v "$1" >/dev/null 2>&1; }

node_major() {
    node -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'
}

ensure_node() {
    if have node && [ "$(node_major)" -ge 18 ]; then
        echo "node $(node -v) already available"
        return
    fi
    if sudo -n true 2>/dev/null; then
        echo "installing nodejs/npm via apt..."
        sudo apt-get update -y -q && sudo apt-get install -y -q nodejs npm
    fi
    if have node && [ "$(node_major)" -ge 18 ]; then
        return
    fi
    echo "installing node LTS tarball to ~/.local..."
    ARCH="$(uname -m)"; [ "$ARCH" = "x86_64" ] && ARCH="x64"
    VER="$(curl -fsSL --max-time 20 https://nodejs.org/dist/index.json | python3 -c \
        "import sys,json; print(next(v['version'] for v in json.load(sys.stdin) if v.get('lts') and v['version'].startswith('v22.')))")"
    curl -fsSL -o /tmp/node.tar.xz "https://nodejs.org/dist/$VER/node-$VER-linux-$ARCH.tar.xz"
    tar -xJf /tmp/node.tar.xz -C /tmp
    mkdir -p ~/.local/node ~/.local/bin
    rm -rf ~/.local/node/* && cp -r "/tmp/node-$VER-linux-$ARCH/"* ~/.local/node/
    ln -sf ~/.local/node/bin/node ~/.local/bin/node
    ln -sf ~/.local/node/bin/npm ~/.local/bin/npm
    ln -sf ~/.local/node/bin/npx ~/.local/bin/npx
    echo "node $(node -v) installed"
}

docker_up() {
    if ! have docker || ! docker info >/dev/null 2>&1; then
        echo "ERROR: docker is required (install it, then re-run)." >&2
        exit 1
    fi
}

ensure_docker_container() {
    local name="$1" image="$2" ports="$3" volume="$4"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ]; then
            echo "container '$name' already running"
            return
        fi
        echo "starting existing '$name' container..."
        docker start "$name"
        return
    fi
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume "$image"
    echo "container '$name' started"
}

wait_http() {
    local url="$1" tries="${2:-30}"
    for _ in $(seq 1 "$tries"); do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: '$url' not ready after ${tries}s" >&2
    return 1
}

run_deps() {
    stage "deps"
    ensure_node
}

run_backend() {
    stage "backend"
    docker_up
    ensure_docker_container qdrant qdrant/qdrant \
        "$QDRANT_PORT:6333" "-v $SCRIPT_DIR/qdrant_data:/qdrant/storage"
    ensure_docker_container redis redis:7 "$REDIS_PORT:6379" ""
    wait_http "http://localhost:$QDRANT_PORT/healthz"

    if [ ! -x "$VENV_PY" ]; then
        echo "creating venv..."
        python3 -m venv "$VENV"
    fi
    "$VENV_PY" -m pip install -q --upgrade pip
    "$VENV_PY" -m pip install -q -r backend/requirements.txt

    if [ ! -f "$ENV_FILE" ]; then
        echo "creating backend/.env from example (fill in credentials!)"
        cp backend/.env.example "$ENV_FILE"
    fi
    if ! grep -q '^REDIS_URL=' "$ENV_FILE"; then
        echo "REDIS_URL=redis://localhost:$REDIS_PORT/0" >> "$ENV_FILE"
    fi
    echo "backend ready"
}

run_index() {
    stage "index"
    [ -x "$VENV_PY" ] || run_backend
    mkdir -p "$LOGS"
    echo "fetching articles from MySQL..."
    (cd backend && "$VENV_PY" scripts/fetch_data.py)
    echo "building index..."
    (cd backend && "$VENV_PY" scripts/build_index.py)
    if [ ! -f backend/data/index_state.json ]; then
        echo "seeding incremental state..."
        (cd backend && "$VENV_PY" scripts/update_index.py --init)
    else
        echo "incremental state already seeded; skipping --init"
    fi
}

run_frontend() {
    stage "frontend"
    ensure_node
    (cd frontend && npm ci)
    local build_env=()
    if [ -n "$PUBLIC_BASE_URL" ]; then
        build_env=(NEXT_PUBLIC_API_BASE="$PUBLIC_BASE_URL")
    fi
    (cd frontend && env "${build_env[@]}" npm run build)
}

start_service() {
    local name="$1" pidfile="$2" logfile="$3"
    shift 3
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "'$name' already running (pid $(cat "$pidfile"))"
        return
    fi
    setsid nohup "$@" >"$logfile" 2>&1 < /dev/null &
    echo $! > "$pidfile"
    echo "'$name' started (pid $(cat "$pidfile"))"
}

run_services() {
    stage "services"
    mkdir -p "$LOGS" "$PID_DIR"
    pkill -f "gunicorn.*$API_PORT" 2>/dev/null || true
    pkill -f "next-server" 2>/dev/null || true
    sleep 2
    start_service gunicorn "$PID_DIR/api.pid" "$LOGS/api.log" \
        "$VENV_PY" -m gunicorn -k uvicorn.workers.UvicornWorker \
        --workers "$GUNICORN_WORKERS" --bind "0.0.0.0:$API_PORT" \
        --timeout 120 -p "$PID_DIR/gunicorn.pid" app.main:app
    wait_http "http://localhost:$API_PORT/health"
    (cd frontend && start_service next "$PID_DIR/next.pid" "$LOGS/frontend.log" \
        "$SCRIPT_DIR/frontend/node_modules/.bin/next" start -p "$NEXT_PORT")
    wait_http "http://localhost:$NEXT_PORT/"
}

run_cron() {
    stage "cron"
    local lock="$SCRIPT_DIR/backend/data/update.lock"
    local log="$LOGS/update_index.log"
    local line="*/15 * * * * flock -n $lock nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"
    crontab -l 2>/dev/null | grep -vF "update_index.py" | crontab -
    ( crontab -l 2>/dev/null; echo "$line" ) | crontab -
    echo "cron installed: */15 * * * * update_index.py"
}

run_nginx() {
    stage "nginx"
    local conf="/etc/nginx/sites-available/search-nlp-rag"
    local site="/etc/nginx/sites-enabled/search-nlp-rag"
    if ! have nginx && ! [ -d /etc/nginx ]; then
        echo "ERROR: nginx not installed." >&2
        exit 1
    fi
    sudo tee "$conf" >/dev/null <<NGINX
server {
    listen $PUBLIC_PORT;
    server_name _;

    location /search { proxy_pass http://127.0.0.1:$API_PORT; }
    location /ask    { proxy_pass http://127.0.0.1:$API_PORT; }
    location /health { proxy_pass http://127.0.0.1:$API_PORT; }

    location / {
        proxy_pass http://127.0.0.1:$NEXT_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX
    sudo ln -sf "$conf" "$site"
    sudo nginx -t
    sudo systemctl reload nginx
    echo "nginx configured on port $PUBLIC_PORT"
}

STAGES=()
ALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        all) ALL=1 ;;
        -h|--help) usage; exit 0 ;;
        deps|backend|index|frontend|services|cron|nginx) STAGES+=("$1") ;;
        *) echo "unknown stage: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ $ALL -eq 1 ]; then
    STAGES=(deps backend index frontend services cron nginx)
fi
if [ ${#STAGES[@]} -eq 0 ]; then
    usage
    exit 1
fi

for s in "${STAGES[@]}"; do
    case "$s" in
        deps) run_deps ;;
        backend) run_backend ;;
        index) run_index ;;
        frontend) run_frontend ;;
        services) run_services ;;
        cron) run_cron ;;
        nginx) run_nginx ;;
    esac
done

echo
echo "setup complete."
echo "app:        http://localhost:$PUBLIC_PORT/"
echo "api:        http://localhost:$API_PORT/health"
echo "qdrant:     http://localhost:$QDRANT_PORT/"
