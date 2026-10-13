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
NGINX_CONF="${NGINX_CONF:-/etc/nginx/sites-available/search-nlp-rag}"
NGINX_LINK="${NGINX_LINK:-/etc/nginx/sites-enabled/search-nlp-rag}"







NGINX_CHAT_LIMIT_RATE="${NGINX_CHAT_LIMIT_RATE:-10r/m}"
NGINX_CHAT_LIMIT_BURST="${NGINX_CHAT_LIMIT_BURST:-10}"




PM2_VERSION="${PM2_VERSION:-7.0.4}"



LOGROTATE_CONF="${LOGROTATE_CONF:-/etc/logrotate.d/vccircle}"
CERTBOT_WEBROOT="${CERTBOT_WEBROOT:-/var/www/certbot}"
LE_ROOT="${LE_ROOT:-/etc/letsencrypt}"
LE_DOMAIN="${LE_DOMAIN:-}"
LE_EMAIL="${LE_EMAIL:-}"


NGINX_TLS="${NGINX_TLS:-auto}"
LE_LIVE="$LE_ROOT/live/$LE_DOMAIN"
LE_CERT="$LE_LIVE/fullchain.pem"
LE_KEY="$LE_LIVE/privkey.pem"


















LE_DOMAIN_RECOVERED=0
if [ -z "$LE_DOMAIN" ]; then
    _le_name=""
    if [ -r "$NGINX_CONF" ]; then
        _le_line=""
        while IFS= read -r _le_line || [ -n "$_le_line" ]; do

            while [ "${_le_line# }" != "$_le_line" ]; do _le_line="${_le_line# }"; done
            while [ "${_le_line#	}" != "$_le_line" ]; do _le_line="${_le_line#	}"; done
            case "$_le_line" in
                "ssl_certificate "*)
                    _le_path="${_le_line#ssl_certificate }"
                    _le_path="${_le_path%%;*}"
                    while [ "${_le_path% }" != "$_le_path" ]; do _le_path="${_le_path% }"; done

                    _le_dir="${_le_path%/*}"
                    _le_parent="${_le_dir%/*}"
                    if [ "${_le_parent##*/}" = "live" ] && [ "${_le_dir##*/}" ]; then
                        _le_name="${_le_dir##*/}"
                        break
                    fi
                    ;;
            esac
        done < "$NGINX_CONF"
    fi
    if [ -n "$_le_name" ]; then
        LE_DOMAIN="$_le_name"
        LE_LIVE="$LE_ROOT/live/$LE_DOMAIN"
        LE_CERT="$LE_LIVE/fullchain.pem"
        LE_KEY="$LE_LIVE/privkey.pem"


        LE_DOMAIN_RECOVERED=1
    fi
    unset _le_name _le_line _le_path _le_dir _le_parent
fi




















API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"
API_MAX_RESTARTS="${API_MAX_RESTARTS:-10}"
FRONTEND_MAX_MEMORY="${FRONTEND_MAX_MEMORY:-1G}"
RESTART_BACKOFF_MS="${RESTART_BACKOFF_MS:-100}"
MIN_UPTIME_MS="${MIN_UPTIME_MS:-30000}"











QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.19.0@sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine@sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499}"






PROMETHEUS_IMAGE="${PROMETHEUS_IMAGE:-quay.io/prometheus/prometheus:v3.13.4@sha256:87861b8cf91579109319ebc300f3f1060e6da9c05d6ae8ad15a20c879e84e32e}"

PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"

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
  prometheus    start the Prometheus container (scrapes 127.0.0.1:8001/metrics)
  stop-backend   stop gunicorn (API) only
  stop-frontend  stop next (frontend) only
  stop-prometheus stop the Prometheus container
  stop       stop backend + frontend + prometheus
  cron       install the 15-minute incremental sync
  nginx      write + enable nginx config (public port -> app + API)
  tls        get a Let's Encrypt cert (webroot) and add the :443 server
  logrotate  render deploy/logrotate.conf for THIS host + install the policy

  all        deps backend index frontend services pm2-startup cron nginx

env overrides:
  QDRANT_PORT REDIS_PORT API_PORT NEXT_PORT PUBLIC_PORT GUNICORN_WORKERS
  API_MAX_MEMORY API_MAX_RESTARTS FRONTEND_MAX_MEMORY RESTART_BACKOFF_MS
  MIN_UPTIME_MS
     pm2 process tuning; must match ecosystem.config.js (kept in step by hand)
  PUBLIC_BASE_URL   e.g. http://your-host (baked into the Next.js build)
  QDRANT_IMAGE REDIS_IMAGE   docker images pinned by digest (defaults
              qdrant/qdrant:v1.19.0@sha256:057ee3a8..., redis:7-alpine@sha256:858f009f...)
  LOGROTATE_CONF   where the logrotate stage installs the rendered policy
              (default /etc/logrotate.d/vccircle)
  NGINX_CHAT_LIMIT_RATE NGINX_CHAT_LIMIT_BURST   nginx rate limit for the SSE
              chat stream only (defaults 10r/m burst 10)
  PM2_VERSION   exact pm2 version to install (default 7.0.4)
  ALLOW_UNSUPPORTED_PY   set to 1 to silence the python >= 3.13 warning
  NGINX_TLS   off | on | auto (default auto: on once LE_DOMAIN has a *usable*
              certificate: non-empty fullchain+privkey, not expired)
  LE_DOMAIN LE_EMAIL   required by the tls stage; renewals mail LE_EMAIL
  LE_ROOT CERTBOT_WEBROOT   override the certificate and ACME challenge paths
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
    TARBALL="node-$VER-linux-$ARCH.tar.xz"
    BASE="https://nodejs.org/dist/$VER"










    if ! have sha256sum; then
        echo "ERROR: sha256sum not found; refusing to install an unverified node." >&2
        return 1
    fi
    curl -fsSL --max-time 20 -o /tmp/node.tar.xz "$BASE/$TARBALL"
    curl -fsSL --max-time 20 -o /tmp/node-shasums.txt "$BASE/SHASUMS256.txt"
    expected="$(awk -v f="$TARBALL" '$2 == f {print $1}' /tmp/node-shasums.txt)"
    if [ -z "$expected" ]; then
        echo "ERROR: $TARBALL is not listed in $BASE/SHASUMS256.txt" >&2
        echo "       Nothing installed; refusing to run an unverified node." >&2
        rm -f /tmp/node.tar.xz /tmp/node-shasums.txt
        return 1
    fi
    if ! printf '%s  %s\n' "$expected" /tmp/node.tar.xz | sha256sum -c - >/dev/null 2>&1; then
        echo "ERROR: checksum mismatch for $TARBALL" >&2
        echo "       expected $expected" >&2
        echo "       Nothing installed; remove /tmp/node.tar.xz and re-run." >&2
        rm -f /tmp/node.tar.xz /tmp/node-shasums.txt
        return 1
    fi
    rm -f /tmp/node-shasums.txt
    echo "node $VER sha256 verified ($expected)"
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
        echo "installing pm2@$PM2_VERSION..."



        npm install -g --no-audit --no-fund "pm2@$PM2_VERSION" >/tmp/pm2-install.log 2>&1 || {
            echo "pm2 install failed:" >&2; tail -3 /tmp/pm2-install.log >&2; return 1
        }
        local nbin
        nbin="$(npm prefix -g)/bin"
        mkdir -p ~/.local/bin
        ln -sf "$nbin/pm2" ~/.local/bin/pm2
        ln -sf "$nbin/pm2-dev" ~/.local/bin/pm2-dev
    fi




    local installed
    installed="$(pm2 -v 2>/dev/null || echo unknown)"
    if [ "$installed" != "$PM2_VERSION" ]; then
        echo "WARNING: pm2 $installed is installed but the pin is $PM2_VERSION." >&2
        echo "         To move it: npm install -g pm2@$PM2_VERSION && ./setup.sh services" >&2
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



container_binds_localhost() {
    if docker inspect -f \
        '{{range $k, $v := .HostConfig.PortBindings}}{{range $v}}{{if ne .HostIp "127.0.0.1"}}PUBLIC_BIND{{end}}{{end}}{{end}}' \
        "$1" 2>/dev/null | grep -q PUBLIC_BIND; then
        return 1
    fi
    return 0
}


random_secret() {
    if have openssl; then
        openssl rand -hex 32
        return
    fi
    "$VENV_PY" -c 'import secrets; print(secrets.token_hex(32))'
}












store_secrets() {
    local redis_pass qdrant_key
    redis_pass="$(env_value "$ENV_FILE" REDIS_PASSWORD)"
    qdrant_key="$(env_value "$ENV_FILE" QDRANT_API_KEY)"
    if [ -z "$redis_pass" ]; then
        redis_pass="$(random_secret)"
        echo "REDIS_PASSWORD=$redis_pass" >> "$ENV_FILE"
        echo "generated REDIS_PASSWORD in $ENV_FILE"
    fi
    if [ -z "$qdrant_key" ]; then
        qdrant_key="$(random_secret)"
        echo "QDRANT_API_KEY=$qdrant_key" >> "$ENV_FILE"
        echo "generated QDRANT_API_KEY in $ENV_FILE"
    fi
    REDIS_PASSWORD="$redis_pass"
    QDRANT_API_KEY="$qdrant_key"
}


DOCKER_LOG_OPTS="--log-driver json-file --log-opt max-size=20m --log-opt max-file=3"












container_config_has() {
    docker inspect -f '{{.Config.Cmd}}{{.Config.Env}}' "$1" 2>/dev/null | grep -qF -- "$2"
}

rebind_container_ports() {
    local name="$1" image="$2" ports="$3" volume="$4" envargs="$5" cmd="$6"
    echo "container '$name' exists with non-localhost port bindings; recreating bound to 127.0.0.1..."
    docker stop "$name" >/dev/null 2>&1 || true
    docker rm "$name" >/dev/null 2>&1 || true
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume $envargs $DOCKER_LOG_OPTS "$image" $cmd
    echo "container '$name' recreated (bound to 127.0.0.1 only)"
}







ensure_docker_container() {
    local name="$1" image="$2" ports="$3" volume="$4" envargs="$5" cmd="$6" needle="$7"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
        if [ -n "$needle" ] && ! container_config_has "$name" "$needle"; then
            echo "container '$name' is running with different credentials than this .env; recreating..."
            docker stop "$name" >/dev/null 2>&1 || true
            docker rm "$name" >/dev/null 2>&1 || true
            echo "pulling + starting '$name'..."
            docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume $envargs $DOCKER_LOG_OPTS "$image" $cmd
            echo "container '$name' recreated with the configured credentials"
            return
        fi
        if container_binds_localhost "$name"; then
            if [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ]; then
                echo "container '$name' already running (bound to 127.0.0.1)"
                return
            fi
            echo "starting existing '$name' container..."
            docker start "$name"
            return
        fi
        rebind_container_ports "$name" "$image" "$ports" "$volume" "$envargs" "$cmd"
        return
    fi
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" -p "$ports" --restart unless-stopped $volume $envargs $DOCKER_LOG_OPTS "$image" $cmd
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





env_value() {
    local file="$1" key="$2" line value
    if [ ! -f "$file" ]; then
        return 0
    fi
    line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
    if [ -z "$line" ]; then
        return 0
    fi
    value="${line#*=}"
    value="${value%$'\r'}"
    case "$value" in
        \"*\") value="${value#\"}"; value="${value%\"}" ;;
        \'*\') value="${value#\'}"; value="${value%\'}" ;;
    esac
    printf '%s\n' "$value"
}


















harden_permissions() {
    local root="${1:-$SCRIPT_DIR}"
    local backend="$root/backend"
    local env_file="$backend/.env"
    local backups="$backend/backups"
    local chat auth db

    chat="$(env_value "$env_file" CHAT_DB_PATH)"
    auth="$(env_value "$env_file" AUTH_DB_PATH)"
    chat="${chat:-data/chat.db}"
    auth="${auth:-data/auth.db}"
    case "$chat" in /*) ;; *) chat="$backend/$chat" ;; esac
    case "$auth" in /*) ;; *) auth="$backend/$auth" ;; esac

    if [ -f "$env_file" ]; then
        chmod 600 "$env_file"
        echo "chmod 600 $env_file"
    else
        echo "skip (no .env yet): $env_file"
    fi

    for db in "$chat" "$auth"; do




        if [ -f "$db" ]; then
            chmod 600 "$db"
            echo "chmod 600 $db"
        else
            echo "skip (not created yet): $db"
        fi
    done

    mkdir -p "$backups"
    chmod 700 "$backups"
    echo "chmod 700 $backups"
}

run_deps() {
    stage "deps"
    ensure_node
}



























migrate_xff_trust() {
    local env_file="$1"


    local assignment='^[[:space:]]*(export[[:space:]]+)?AUTH_TRUST_X_FORWARDED_FOR[[:space:]]*='









    local bare="$assignment[[:space:]]*(1|true|yes|on)([[:space:]]+#.*)?[[:space:]]*$"
    local double_quoted="$assignment[[:space:]]*\"[[:space:]]*(1|true|yes|on)[[:space:]]*\"([[:space:]]*#.*)?[[:space:]]*$"
    local single_quoted="$assignment[[:space:]]*'[[:space:]]*(1|true|yes|on)[[:space:]]*'([[:space:]]*#.*)?[[:space:]]*$"

    if ! grep -qE "$assignment" "$env_file"; then
        echo "AUTH_TRUST_X_FORWARDED_FOR=auto" >> "$env_file"
    elif grep -qiE -e "$bare" -e "$double_quoted" -e "$single_quoted" "$env_file"; then








        echo "WARNING: AUTH_TRUST_X_FORWARDED_FOR is set to a forced-true value" >&2
        echo "         (1/true/yes/on), which trusts X-Forwarded-For from ANY" >&2
        echo "         peer, so a client reaching :8001 directly can forge it to" >&2
        echo "         dodge a rate limit. Set it to 'auto' (the new default) if" >&2
        echo "         your reverse proxy runs on this host; keep it forced if the" >&2
        echo "         proxy runs on another host." >&2
    fi
}

run_backend() {
    stage "backend"
    check_python python3
    docker_up









    if [ ! -f "$ENV_FILE" ]; then
        echo "creating backend/.env from example (fill in credentials!)"
        cp backend/.env.example "$ENV_FILE"
    fi
    if [ ! -x "$VENV_PY" ]; then
        echo "creating venv..."
        python3 -m venv "$VENV"
    fi
    store_secrets


    chmod 600 "$ENV_FILE"






    ensure_docker_container qdrant "$QDRANT_IMAGE" \
        "127.0.0.1:$QDRANT_PORT:6333" "-v $SCRIPT_DIR/qdrant_data:/qdrant/storage" \
        "-e QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY" "" \
        "QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY"
    ensure_docker_container redis "$REDIS_IMAGE" "127.0.0.1:$REDIS_PORT:6379" "" "" \
        "redis-server --requirepass $REDIS_PASSWORD" \
        "--requirepass $REDIS_PASSWORD"


    wait_http "http://localhost:$QDRANT_PORT/healthz"

    "$VENV_PY" -m pip install -q --upgrade pip
    "$VENV_PY" -m pip install -q -r backend/requirements.txt






    if ! grep -qE '^REDIS_URL=redis://[^/@]*@' "$ENV_FILE"; then
        if grep -q '^REDIS_URL=' "$ENV_FILE"; then




            local escaped="$REDIS_PASSWORD"
            escaped="${escaped//\//\\/}"
            escaped="${escaped//&/\\&}"
            sed -i -E "s#^REDIS_URL=(.*)\$#REDIS_URL=redis://:${escaped}@\\1#" "$ENV_FILE"
            echo "REDIS_URL now carries the redis password"
        else
            echo "REDIS_URL=redis://:$REDIS_PASSWORD@localhost:$REDIS_PORT/0" >> "$ENV_FILE"
        fi
    fi







    migrate_xff_trust "$ENV_FILE"
    echo "backend ready"
    harden_permissions
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








report_readiness_reason() {
    local body
    body="$(curl -sS -m 5 "http://localhost:$API_PORT/ready/deep" 2>/dev/null || true)"
    if [ -n "$body" ]; then
        echo "       readiness report:" >&2
        printf '%s\n' "$body" | sed 's/^/         /' >&2
    else
        echo "       (no report body; the probe did not answer)" >&2
    fi
}

run_services() {
    stage "services"
    mkdir -p "$LOGS" "$PID_DIR"
    harden_permissions
    ensure_pm2
    pm2 delete vccircle-backend >/dev/null 2>&1 || true
    pm2 delete vccircle-frontend >/dev/null 2>&1 || true
    sleep 2
















    (cd "$SCRIPT_DIR" && \
        VCCIRCLE_ROOT="$SCRIPT_DIR" \
        GUNICORN_WORKERS="$GUNICORN_WORKERS" \
        API_PORT="$API_PORT" \
        NEXT_PORT="$NEXT_PORT" \
        MIN_UPTIME_MS="$MIN_UPTIME_MS" \
        API_MAX_MEMORY="$API_MAX_MEMORY" \
        FRONTEND_MAX_MEMORY="$FRONTEND_MAX_MEMORY" \
        API_MAX_RESTARTS="$API_MAX_RESTARTS" \
        RESTART_BACKOFF_MS="$RESTART_BACKOFF_MS" \
        pm2 start ecosystem.config.js)
    pm2 save >/dev/null 2>&1
    wait_http "http://localhost:$API_PORT/health"
    wait_http "http://localhost:$NEXT_PORT/"

















    if ! wait_http "http://localhost:$API_PORT/ready/deep"; then
        report_readiness_reason
        return 1
    fi
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













ensure_prometheus_container() {
    local name="prometheus"
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ]; then
            echo "container '$name' already running (host network)"
            return
        fi
        echo "starting existing '$name' container..."
        docker start "$name"
        return
    fi
    echo "pulling + starting '$name'..."
    docker run -d --name "$name" --network host --restart unless-stopped \
        -v "$SCRIPT_DIR/deploy/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
        -v "$SCRIPT_DIR/prometheus_data:/prometheus" \
        $DOCKER_LOG_OPTS \
        "$PROMETHEUS_IMAGE" \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.listen-address=127.0.0.1:$PROMETHEUS_PORT
    echo "container '$name' started (host network, bound to 127.0.0.1:$PROMETHEUS_PORT)"
}

run_prometheus() {
    stage "prometheus"
    docker_up






    mkdir -p "$SCRIPT_DIR/prometheus_data"
    chown -R 65534:65534 "$SCRIPT_DIR/prometheus_data"
    ensure_prometheus_container
    wait_http "http://localhost:$PROMETHEUS_PORT/-/healthy"
    echo "prometheus: http://localhost:$PROMETHEUS_PORT/"
}

run_stop_prometheus() {
    stage "stop-prometheus"
    if docker ps -a --format '{{.Names}}' | grep -qx prometheus; then
        docker stop prometheus >/dev/null 2>&1 || true
        echo "'prometheus' container stopped"
    else
        echo "'prometheus' was not running"
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
    run_stop_prometheus
}

run_cron() {
    stage "cron"
    local log="$LOGS/update_index.log"
    local hc_log="$LOGS/healthcheck.log"




    local line_idx="*/15 * * * * nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"








    local line_hc="*/5 * * * * BASE=\"http://localhost:$API_PORT\" HEALTHCHECK_WEBHOOK_URL=\"${HEALTHCHECK_WEBHOOK_URL:-}\" LOG=$hc_log $SCRIPT_DIR/deploy/healthcheck.sh"
    local tmp
    tmp="$(mktemp)"

























    crontab -l 2>/dev/null \
        | grep -vFx "$line_idx" \
        | awk -v p="$SCRIPT_DIR/deploy/healthcheck.sh" \
            'index($0, p) && $0 !~ /^[[:space:]]*#/ { next } { print }' > "$tmp" \
        || true
    printf '%s\n' "$line_idx" >> "$tmp"
    printf '%s\n' "$line_hc" >> "$tmp"
    crontab "$tmp"
    rm -f "$tmp"
    echo "cron installed: */15 * * * * update_index.py"
    echo "cron installed: */5  * * * * healthcheck.sh"
}




















LOGROTATE_TEMPLATE="$SCRIPT_DIR/deploy/logrotate.conf"

render_logrotate_conf() {
    local rendered user group
    if [ ! -f "$LOGROTATE_TEMPLATE" ]; then
        echo "ERROR: logrotate template not found: $LOGROTATE_TEMPLATE" >&2
        return 1
    fi
    user="$(id -un)"
    group="$(id -gn)"
    if ! rendered="$(sed \
        -e "s#^/path/to/search-nlp-rag/logs/\\*\\.log#$LOGS/*.log#" \
        -e "s#^/path/to/\\.pm2/logs/\\*\\.log#$HOME/.pm2/logs/*.log#" \
        -e "s|^\( *\)# su deploy-user deploy-group\$|\1su $user $group|" \
        "$LOGROTATE_TEMPLATE")"; then
        echo "ERROR: could not read $LOGROTATE_TEMPLATE" >&2
        return 1
    fi




    local expected
    for expected in \
        '/path/to/search-nlp-rag/logs/*.log' \
        '/path/to/.pm2/logs/*.log' \
        'su deploy-user deploy-group'
    do
        if ! grep -qF -- "$expected" "$LOGROTATE_TEMPLATE"; then
            echo "ERROR: $LOGROTATE_TEMPLATE no longer contains" >&2
            echo "         $expected" >&2
            echo "       which render_logrotate_conf rewrites. Nothing installed;" >&2
            echo "       update the template and the renderer together." >&2
            return 1
        fi
    done
    printf '%s\n' "$rendered"
}

run_logrotate() {
    stage "logrotate"
    if ! have logrotate; then
        echo "ERROR: logrotate is not installed (e.g. apt install logrotate)." >&2
        echo "       Nothing installed; logs will grow unrotated." >&2
        return 1
    fi
    local tmp
    tmp="$(mktemp)"


    trap 'rm -f "$tmp"' RETURN
    if ! render_logrotate_conf > "$tmp"; then
        echo "ERROR: could not render the logrotate policy; nothing installed." >&2
        return 1
    fi


    sudo install -m 644 "$tmp" "$LOGROTATE_CONF"
    rm -f "$tmp"
    trap - RETURN
    echo "logrotate installed: $LOGROTATE_CONF (mode 644, root)"
    echo "inspect it with: sudo logrotate -d $LOGROTATE_CONF"
}

nginx_server_name() {


    if [ -n "$LE_DOMAIN" ]; then
        echo "$LE_DOMAIN"
    else
        echo "_"
    fi
}




















nginx_tls_cert_valid() {
    [ -n "$LE_DOMAIN" ] || return 1
    [ -f "$LE_CERT" ] && [ -s "$LE_CERT" ] || return 1
    [ -f "$LE_KEY" ] && [ -s "$LE_KEY" ] || return 1
    return 0
}





















nginx_tls_cert_state() {
    if [ ! -f "$LE_CERT" ]; then echo "missing"; return 0; fi
    if [ ! -s "$LE_CERT" ]; then echo "empty"; return 0; fi
    if [ ! -r "$LE_CERT" ]; then echo "unreadable"; return 0; fi
    if ! have openssl; then echo "unknown"; return 0; fi
    if [ -z "$(openssl x509 -noout -enddate -in "$LE_CERT" 2>/dev/null)" ]; then
        echo "corrupt"
        return 0
    fi


    if openssl x509 -checkend 0 -noout -in "$LE_CERT" >/dev/null 2>&1; then
        echo "current"
    else
        echo "expired"
    fi
}



nginx_conf_serves_tls() {
    local line
    [ -r "$NGINX_CONF" ] || return 1
    while IFS= read -r line || [ -n "$line" ]; do
        while [ "${line# }" != "$line" ]; do line="${line# }"; done
        while [ "${line#	}" != "$line" ]; do line="${line#	}"; done
        case "$line" in
            "listen 443 ssl"*) return 0 ;;
        esac
    done < "$NGINX_CONF"
    return 1
}













nginx_tls_mode() {
    case "$NGINX_TLS" in
        off|0|false|no) echo "off" ;;
        on|1|true|yes) echo "on" ;;
        *)
            if nginx_tls_cert_valid || nginx_conf_serves_tls; then
                echo "on"
            else
                echo "off"
            fi
            ;;
    esac
}



nginx_locations() {
    cat <<NGINX













    location /search {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /health {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
    }
    location /live {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
    }
    location /ready {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /readyz {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }









    location /ready/deep { return 404; }
    location /facets {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }






























    location ^~ /api/chat/ {
        limit_req zone=vccircle_chat_stream burst=$NGINX_CHAT_LIMIT_BURST nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_read_timeout 300s;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    location /api {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_read_timeout 300s;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    location /recommend/ {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /analytics/click {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /analytics/summary {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
    }
    location /analytics/chat {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
    }
    location /analytics/users {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
    }
    location /analytics { proxy_pass http://127.0.0.1:$NEXT_PORT; }

    location / {
        proxy_pass http://127.0.0.1:$NEXT_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
NGINX
}





render_nginx_config() {
    local mode="$1"
    local name
    name="$(nginx_server_name)"
    {











        cat <<NGINX
limit_req_zone \$binary_remote_addr zone=vccircle_chat_stream:10m rate=$NGINX_CHAT_LIMIT_RATE;

NGINX
        cat <<NGINX
server {
    listen $PUBLIC_PORT;
    server_name $name;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;



    location ^~ /.well-known/acme-challenge/ {
        root $CERTBOT_WEBROOT;
        default_type text/plain;
    }
NGINX
        if [ "$mode" = "on" ]; then


            cat <<'NGINX'

    location / { return 301 https://$host$request_uri; }
NGINX
        else
            echo
            nginx_locations
        fi
        echo "}"
        if [ "$mode" = "on" ]; then





            cat <<NGINX

server {
    listen 443 ssl http2;
    server_name $name;

    ssl_certificate $LE_CERT;
    ssl_certificate_key $LE_KEY;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
NGINX
            echo
            nginx_locations
            echo "}"
        fi
    }
}











nginx_site_config() {
    local mode
    mode="$(nginx_tls_mode)"
    if [ "$mode" = "on" ]; then




        if [ ! -f "$LE_CERT" ] || [ ! -s "$LE_CERT" ]; then
            echo "ERROR: TLS is on but there is no certificate at $LE_CERT." >&2
            echo "       Get one with: LE_DOMAIN=... LE_EMAIL=... ./setup.sh tls" >&2
            echo "       Or go back to HTTP with: NGINX_TLS=off ./setup.sh nginx" >&2
            return 1
        fi
        if [ ! -f "$LE_KEY" ] || [ ! -s "$LE_KEY" ]; then
            echo "ERROR: TLS is on but there is no private key at $LE_KEY." >&2
            echo "       Get one with: LE_DOMAIN=... LE_EMAIL=... ./setup.sh tls" >&2
            echo "       Or go back to HTTP with: NGINX_TLS=off ./setup.sh nginx" >&2
            return 1
        fi
    fi
    render_nginx_config "$mode"
}

run_nginx() {
    stage "nginx"
    if ! have nginx && ! [ -d /etc/nginx ]; then
        echo "ERROR: nginx not installed." >&2
        exit 1
    fi
    local mode
    mode="$(nginx_tls_mode)"




    local state
    state="$(nginx_tls_cert_state)"
    if [ "$mode" = "off" ]; then








        if { [ -n "$LE_DOMAIN" ] || [ "$state" != "missing" ]; } && [ "${TLS_BOOTSTRAP:-0}" != "1" ]; then



            echo "WARNING: this site is being served over plain HTTP." >&2
            echo "         No certificate and private key were found at" >&2
            echo "           $LE_CERT" >&2
            echo "           $LE_KEY" >&2
            echo "         Passwords, bearer tokens and chat content will cross" >&2
            echo "         the wire in cleartext until that pair exists." >&2
            echo "         Fix it with: LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
        fi
    elif [ "$state" = "expired" ]; then




        echo "WARNING: the certificate at $LE_CERT has expired." >&2
        echo "         The site is still being served over HTTPS, but browsers" >&2
        echo "         will warn and clients may refuse the connection." >&2
        echo "         Renew it with: LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    elif [ "$state" = "corrupt" ]; then




        echo "WARNING: the certificate at $LE_CERT cannot be read as a" >&2
        echo "         certificate at all. This is not an expiry problem and" >&2
        echo "         re-running the tls stage will not fix it: issuance uses" >&2
        echo "         --keep-until-expiring, so certbot skips a file it cannot" >&2
        echo "         parse and leaves it untouched. Remove it, then renew:" >&2
        echo "           sudo rm -f $LE_CERT" >&2
        echo "           LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    fi




    local tmp
    tmp="$(mktemp)"


    if ! nginx_site_config > "$tmp"; then
        echo "ERROR: could not render the nginx config; $NGINX_CONF left untouched." >&2
        rm -f "$tmp"
        return 1
    fi
    if [ -f "$NGINX_CONF" ]; then
        sudo cp "$NGINX_CONF" "$NGINX_CONF.bak"
    fi
    sudo install -m 644 "$tmp" "$NGINX_CONF"
    rm -f "$tmp"
    sudo ln -sf "$NGINX_CONF" "$NGINX_LINK"
    sudo rm -f /etc/nginx/sites-enabled/default
    if ! sudo nginx -t; then

        echo "ERROR: nginx rejected the new config; rolling back." >&2
        if [ -f "$NGINX_CONF.bak" ]; then
            sudo mv -f "$NGINX_CONF.bak" "$NGINX_CONF"
        else
            sudo rm -f "$NGINX_CONF"
        fi
        return 1
    fi

    sudo systemctl reload nginx
    if [ "$mode" = "on" ]; then
        echo "nginx configured on port $PUBLIC_PORT with TLS (https://$LE_DOMAIN/)"
    else
        echo "nginx configured on port $PUBLIC_PORT (plain HTTP)"
    fi
    echo "roll back to plain HTTP: NGINX_TLS=off ./setup.sh nginx"




    if [ "$mode" = "on" ]; then
        if [ "$LE_DOMAIN_RECOVERED" = "1" ]; then
            echo "serving: https via $LE_DOMAIN (domain recovered from the installed config; export LE_DOMAIN=$LE_DOMAIN to manage it)"
        else
            echo "serving: https via $LE_DOMAIN"
        fi
    elif [ -n "$LE_DOMAIN" ]; then
        echo "serving: http only (no certificate at $LE_CERT)"
    else
        echo "serving: http only (no certificate and no domain; set LE_DOMAIN and run './setup.sh tls')"
    fi
}

run_tls() {
    stage "tls"
    if [ -z "$LE_DOMAIN" ]; then
        echo "ERROR: LE_DOMAIN is not set, e.g. LE_DOMAIN=example.com ./setup.sh tls" >&2
        return 1
    fi
    if [ -z "$LE_EMAIL" ]; then
        echo "ERROR: LE_EMAIL is not set; Let's Encrypt expiry warnings go there." >&2
        return 1
    fi
    if ! have certbot; then
        echo "ERROR: certbot is not installed (e.g. sudo apt-get install -y certbot)." >&2
        return 1
    fi






















    local preflight
    NGINX_TLS=auto
    preflight="$(nginx_tls_mode)"
    if ! TLS_BOOTSTRAP=1 run_nginx; then
        echo "WARNING: the nginx pre-flight could not install a config; the one" >&2
        echo "         already on disk is untouched and still serving. Continuing" >&2
        echo "         so certbot can still be attempted. If that config was not" >&2
        echo "         written by this script, check that it serves" >&2
        echo "         $CERTBOT_WEBROOT/.well-known/acme-challenge/ before trusting" >&2
        echo "         the result." >&2
    fi
    sudo mkdir -p "$CERTBOT_WEBROOT/.well-known/acme-challenge"





    if ! sudo certbot certonly \
        --webroot -w "$CERTBOT_WEBROOT" \
        --cert-name "$LE_DOMAIN" -d "$LE_DOMAIN" \
        --email "$LE_EMAIL" --agree-tos --non-interactive \
        --keep-until-expiring \
        --deploy-hook 'systemctl reload nginx'; then




        if [ "$preflight" = "on" ]; then
            echo "ERROR: certbot failed; the existing TLS config is still installed and serving." >&2
        else
            echo "ERROR: certbot failed; the plain-HTTP config is still installed and serving." >&2
        fi
        echo "       The usual cause is that http://$LE_DOMAIN/.well-known/acme-challenge/" >&2
        echo "       is not reaching this host: check that the domain's A/AAAA record" >&2
        echo "       points here and that port 80 is open in the firewall and the" >&2
        echo "       cloud security group. certbot's own log:" >&2
        echo "       sudo journalctl -u certbot -n 50 --no-pager" >&2
        return 1
    fi

    NGINX_TLS=on
    run_nginx
    if systemctl is-enabled certbot.timer >/dev/null 2>&1; then
        echo "renewal: handled by the systemd certbot.timer"
    else



        local line_renew="17 3 * * * certbot renew --quiet --deploy-hook 'systemctl reload nginx'"
        local tmp
        tmp="$(mktemp)"
        sudo crontab -l 2>/dev/null | grep -vFx "$line_renew" > "$tmp" || true
        printf '%s\n' "$line_renew" >> "$tmp"
        sudo crontab "$tmp"
        rm -f "$tmp"
        echo "renewal: certbot.timer is not enabled; installed a daily certbot renew crontab line"
    fi
    echo "TLS enabled: https://$LE_DOMAIN/"
}

main() {
    STAGES=()
    local ALL=0
    while [ $# -gt 0 ]; do
        case "$1" in
            all) ALL=1 ;;
            -h|--help) usage; return 0 ;;
            deps|backend|index|frontend|services|pm2-startup|stop-backend|stop-frontend|stop-prometheus|stop|cron|nginx|tls|logrotate|prometheus) STAGES+=("$1") ;;
            *) echo "unknown stage: $1"; usage; return 1 ;;
        esac
        shift
    done

    if [ "$ALL" -eq 1 ]; then






        STAGES=(deps backend index frontend services pm2-startup cron nginx)
    fi
    if [ ${#STAGES[@]} -eq 0 ]; then
        usage
        return 1
    fi

    for s in "${STAGES[@]}"; do
        case "$s" in
            deps) run_deps ;;
            backend) run_backend ;;
            index) run_index ;;
            frontend) run_frontend ;;
            services) run_services ;;
            pm2-startup) run_pm2_startup ;;
            prometheus) run_prometheus ;;
            stop-backend) run_stop_backend ;;
            stop-frontend) run_stop_frontend ;;
            stop-prometheus) run_stop_prometheus ;;
            stop) run_stop ;;
            cron) run_cron ;;
            nginx) run_nginx ;;
            tls) run_tls ;;
            logrotate) run_logrotate ;;
        esac
    done

    echo
    echo "setup complete."
    echo "app:        http://localhost:$PUBLIC_PORT/"
    if [ "$(nginx_tls_mode)" = "on" ]; then
        echo "app:        https://$LE_DOMAIN/"
    fi
    echo "api:        http://localhost:$API_PORT/health"
    echo "qdrant:     http://localhost:$QDRANT_PORT/"
    echo "prometheus: http://localhost:$PROMETHEUS_PORT/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
