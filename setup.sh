#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

QDRANT_PORT="${QDRANT_PORT:-6333}"
REDIS_PORT="${REDIS_PORT:-6379}"
API_PORT="${API_PORT:-8001}"
NEXT_PORT="${NEXT_PORT:-3000}"
PUBLIC_PORT="${PUBLIC_PORT:-80}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"

# Pinned docker images with digests for reproducibility. IMPORTANT: the Qdrant
# version must be >= the version that wrote an existing collection (older
# versions cannot deserialize newer storage formats). Current default matches
# the deployment that created the live collection.
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.19.0@sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"

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
  services   start gunicorn + next (pm2) in the background
  pm2-startup   install systemd unit so pm2 restores the frontend on boot
  stop-backend   stop gunicorn (API) only
  stop-frontend  stop next (frontend) only
  stop       stop both backend + frontend
  cron       install the 15-minute incremental sync
  nginx      write + enable nginx config (public port -> app + API)

  all        deps backend index frontend services pm2-startup cron nginx

env overrides:
  QDRANT_PORT REDIS_PORT API_PORT NEXT_PORT PUBLIC_PORT GUNICORN_WORKERS
  PUBLIC_BASE_URL   e.g. http://your-host (baked into the Next.js build)
  QDRANT_IMAGE REDIS_IMAGE   pinned docker image tags (defaults qdrant/qdrant:v1.19.0, redis:7-alpine)
  ALLOW_UNSUPPORTED_PY   set to 1 to silence the python >= 3.13 warning
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

ensure_pm2() {
    if ! have pm2; then
        echo "installing pm2..."
        npm install -g pm2 >/tmp/pm2-install.log 2>&1 || {
            echo "pm2 install failed:" >&2; tail -3 /tmp/pm2-install.log >&2; return 1
        }
        local nbin
        nbin="$(npm prefix -g)/bin"
        mkdir -p ~/.local/bin
        ln -sf "$nbin/pm2" ~/.local/bin/pm2
        ln -sf "$nbin/pm2-dev" ~/.local/bin/pm2-dev
    fi
    if have pm2; then
        echo "pm2 $(pm2 -v) available"
    else
        echo "pm2 not on PATH" >&2
        return 1
    fi
}

docker_up() {
    if ! have docker || ! docker info >/dev/null 2>&1; then
        echo "ERROR: docker is required (install it, then re-run)." >&2
        exit 1
    fi
}

check_python() {
    local py="${1:-python3}"
    local ver major minor
    ver="$("$py" -c 'import sys; print("%d.%d" % (sys.version_info[0], sys.version_info[1]))' 2>/dev/null || echo "0.0")"
    major="${ver%%.*}"; minor="${ver##*.}"
    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 11 ]; }; then
        echo "ERROR: '$py' is Python $ver; this project requires Python >= 3.11 and < 3.13." >&2
        echo "       Install python3.11 or python3.12 (e.g. 'sudo apt install python3.12') and re-run." >&2
        exit 1
    fi
    if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 13 ]; }; then
        if [ "${ALLOW_UNSUPPORTED_PY:-0}" = "1" ]; then
            echo "WARNING: '$py' is Python $ver (outside supported 3.11-3.12); continuing because ALLOW_UNSUPPORTED_PY=1"
        else
            echo "WARNING: '$py' is Python $ver, newer than the supported range (3.11-3.12)." >&2
            echo "         torch 2.x may lack wheels for it; prefer 'python3.12'." >&2
            echo "         Continuing anyway (set ALLOW_UNSUPPORTED_PY=1 to silence this)." >&2
        fi
    fi
}

# Succeeds when every published host port of the container binds to 127.0.0.1
# (a 0.0.0.0/"" bind means it is reachable from the network).
container_binds_localhost() {
    if docker inspect -f \
        '{{range $k, $v := .HostConfig.PortBindings}}{{range $v}}{{if ne .HostIp "127.0.0.1"}}PUBLIC_BIND{{end}}{{end}}{{end}}' \
        "$1" 2>/dev/null | grep -q PUBLIC_BIND; then
        return 1
    fi
    return 0
}

# Bound docker json-file logs so they can't fill the disk (20MB x 3 files each).
DOCKER_LOG_OPTS="--log-driver json-file --log-opt max-size=20m --log-opt max-file=3"

rebind_container_ports() {
    local name="$1" image="$2" ports="$3" volume="$4"
    echo "container '$name' exists with non-localhost port bindings; recreating bound to 127.0.0.1..."
    docker stop "$name" >/dev/null 2>&1 || true
    docker rm "$name" >/dev/null 2>&1 || true
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume $DOCKER_LOG_OPTS "$image"
    echo "container '$name' recreated (bound to 127.0.0.1 only)"
}

ensure_docker_container() {
    local name="$1" image="$2" ports="$3" volume="$4"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
        if container_binds_localhost "$name"; then
            if [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ]; then
                echo "container '$name' already running (bound to 127.0.0.1)"
                return
            fi
            echo "starting existing '$name' container..."
            docker start "$name"
            return
        fi
        rebind_container_ports "$name" "$image" "$ports" "$volume"
        return
    fi
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume $DOCKER_LOG_OPTS "$image"
    echo "container '$name' started (bound to 127.0.0.1 only)"
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
    check_python python3
    docker_up
    ensure_docker_container qdrant "$QDRANT_IMAGE" \
        "127.0.0.1:$QDRANT_PORT:6333" "-v $SCRIPT_DIR/qdrant_data:/qdrant/storage"
    ensure_docker_container redis "$REDIS_IMAGE" "127.0.0.1:$REDIS_PORT:6379" ""
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
    ensure_pm2
    pm2 delete vccircle-backend >/dev/null 2>&1 || true
    pm2 delete vccircle-frontend >/dev/null 2>&1 || true
    sleep 2

    (cd backend && pm2 start "$VENV_PY" \
        --name vccircle-backend -- -m gunicorn \
        -k uvicorn.workers.UvicornWorker \
        --workers "$GUNICORN_WORKERS" --bind "0.0.0.0:$API_PORT" \
        --timeout 120 app.main:app)
    wait_http "http://localhost:$API_PORT/health"

    (cd frontend && pm2 start "$SCRIPT_DIR/frontend/node_modules/.bin/next" \
        --name vccircle-frontend -- start -p "$NEXT_PORT")
    pm2 save >/dev/null 2>&1
    wait_http "http://localhost:$NEXT_PORT/"
}

stop_service() {
    local name="$1" pidfile="$2"
    if [ -f "$pidfile" ]; then
        local pid
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "'$name' stopped (pid $pid)"
        else
            echo "'$name' was not running"
        fi
        rm -f "$pidfile"
    else
        echo "'$name' was not running"
    fi
}

run_pm2_startup() {
    stage "pm2-startup"
    ensure_pm2
    if [ -d "/etc/systemd/system" ] && have systemctl; then
        sudo env "PATH=$PATH" pm2 startup systemd -u "$USER" --hp "$HOME" >/dev/null
        pm2 save >/dev/null
        systemctl is-enabled "pm2-$USER" >/dev/null 2>&1 \
            && echo "pm2 boot-start enabled (pm2-$USER)" \
            || echo "pm2 startup completed"
    else
        echo "systemd not available; pm2 manual start only"
    fi
}

run_stop_backend() {
    stage "stop-backend"
    if have pm2; then
        pm2 delete vccircle-backend >/dev/null 2>&1 || true
    fi
    stop_service gunicorn "$PID_DIR/api.pid"
}

run_stop_frontend() {
    stage "stop-frontend"
    stop_service next "$PID_DIR/next.pid"
    if have pm2; then
        if pm2 delete vccircle-frontend >/dev/null 2>&1; then
            echo "'vccircle-frontend' stopped (pm2)"
        else
            echo "'vccircle-frontend' not managed by pm2"
        fi
    fi
}

run_stop() {
    stage "stop"
    run_stop_backend
    run_stop_frontend
}

run_cron() {
    stage "cron"
    local log="$LOGS/update_index.log"
    # update_index.py takes its own flock(2) on data/update.lock (LOCK_EX|LOCK_NB)
    # and skips when another run holds it, so no external flock wrapper is needed
    # (wrapping with `flock -n` would conflict with the script's own lock and
    # cause every run to be skipped).
    local line="*/15 * * * * nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"
    local tmp
    tmp="$(mktemp)"
    crontab -l 2>/dev/null | grep -vF "update_index.py" > "$tmp" || true
    printf '%s\n' "$line" >> "$tmp"
    crontab "$tmp"
    rm -f "$tmp"
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

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location /search { proxy_pass http://127.0.0.1:$API_PORT; }
    location /health { proxy_pass http://127.0.0.1:$API_PORT; }
    location /live { proxy_pass http://127.0.0.1:$API_PORT; }
    location /ready { proxy_pass http://127.0.0.1:$API_PORT; }
    location /facets { proxy_pass http://127.0.0.1:$API_PORT; }
    location /analytics { proxy_pass http://127.0.0.1:$API_PORT; }
    location /ask {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_read_timeout 300s;
    }

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
    sudo rm -f /etc/nginx/sites-enabled/default
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
        deps|backend|index|frontend|services|pm2-startup|stop-backend|stop-frontend|stop|cron|nginx) STAGES+=("$1") ;;
        *) echo "unknown stage: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ $ALL -eq 1 ]; then
    STAGES=(deps backend index frontend services pm2-startup cron nginx)
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
        pm2-startup) run_pm2_startup ;;
        stop-backend) run_stop_backend ;;
        stop-frontend) run_stop_frontend ;;
        stop) run_stop ;;
        cron) run_cron ;;
        nginx) run_nginx ;;
    esac
done

echo
echo "setup complete."
echo "app:        http://localhost:$PUBLIC_PORT/"
echo "api:        http://localhost:$API_PORT/health"
echo "qdrant:     http://localhost:$QDRANT_PORT/"
