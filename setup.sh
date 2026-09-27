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
CERTBOT_WEBROOT="${CERTBOT_WEBROOT:-/var/www/certbot}"
LE_ROOT="${LE_ROOT:-/etc/letsencrypt}"
LE_DOMAIN="${LE_DOMAIN:-}"
LE_EMAIL="${LE_EMAIL:-}"
# auto|on|off. "auto" means: TLS once a domain is configured and its
# certificate is actually usable (see nginx_tls_cert_valid), plain HTTP
# otherwise.
NGINX_TLS="${NGINX_TLS:-auto}"
LE_LIVE="$LE_ROOT/live/$LE_DOMAIN"
LE_CERT="$LE_LIVE/fullchain.pem"
LE_KEY="$LE_LIVE/privkey.pem"

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
  tls        get a Let's Encrypt cert (webroot) and add the :443 server

  all        deps backend index frontend services pm2-startup cron nginx

env overrides:
  QDRANT_PORT REDIS_PORT API_PORT NEXT_PORT PUBLIC_PORT GUNICORN_WORKERS
  PUBLIC_BASE_URL   e.g. http://your-host (baked into the Next.js build)
  QDRANT_IMAGE REDIS_IMAGE   pinned docker image tags (defaults qdrant/qdrant:v1.19.0, redis:7-alpine)
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
    # Per-IP rate limiting keys on the client IP, which behind nginx comes from
    # X-Forwarded-For. An .env that predates the per-IP public rate limits has
    # no trust setting at all, so every proxied request keys on the nginx peer
    # (127.0.0.1) and the whole site shares one rate-limit bucket. Append the
    # shipped default in that one case.
    if ! grep -q '^AUTH_TRUST_X_FORWARDED_FOR=' "$ENV_FILE"; then
        echo "AUTH_TRUST_X_FORWARDED_FOR=auto" >> "$ENV_FILE"
    elif grep -qiE '^AUTH_TRUST_X_FORWARDED_FOR=[[:space:]]*(1|true|yes|on)[[:space:]]*$' "$ENV_FILE"; then
        # The spellings above are exactly the ones config._env_tristate reads as
        # a forced True, so this warning covers every value that leaves the
        # header trusted from any peer -- not just the literal "true".
        # Warn, never rewrite. A forced True is the correct setting when the
        # proxy runs on ANOTHER host, and silently downgrading it to 'auto'
        # would collapse exactly that deployment back into the single-bucket
        # outage. Such a host is already rate-limiting per IP correctly; its
        # residual risk is that the header is trusted from ANY peer, which only
        # matters when :8001 is also reachable directly (gunicorn binds
        # 0.0.0.0 -- see issue #245). 'auto' closes that and is safe whenever
        # the proxy is on this host, but the operator's value is theirs.
        echo "WARNING: AUTH_TRUST_X_FORWARDED_FOR is set to a forced-true value" >&2
        echo "         (1/true/yes/on), which trusts X-Forwarded-For from ANY" >&2
        echo "         peer, so a client reaching :8001 directly can forge it to" >&2
        echo "         dodge a rate limit. Set it to 'auto' (the new default) if" >&2
        echo "         your reverse proxy runs on this host; keep it forced if the" >&2
        echo "         proxy runs on another host." >&2
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
    local hc_log="$LOGS/healthcheck.log"
    # update_index.py takes its own flock(2) on data/update.lock (LOCK_EX|LOCK_NB)
    # and skips when another run holds it, so no external flock wrapper is needed
    # (wrapping with `flock -n` would conflict with the script's own lock and
    # cause every run to be skipped).
    local line_idx="*/15 * * * * nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"
    # cron does not inherit the operator's shell environment, so pass the
    # webhook URL (and a minimal PATH via healthcheck.sh) explicitly. Empty
    # webhook is harmless: healthcheck.sh treats an unset/empty value as "no
    # webhook". Keep the entry stable for idempotent re-runs.
    local line_hc="*/5 * * * * HEALTHCHECK_WEBHOOK_URL=\"${HEALTHCHECK_WEBHOOK_URL:-}\" LOG=$hc_log $SCRIPT_DIR/deploy/healthcheck.sh"
    local tmp
    tmp="$(mktemp)"
    # Remove only the exact managed entries this script writes; preserve any
    # user-added crontab lines (including manual HEALTHCHECK_WEBHOOK_URL=...
    # augmentations) that reference the same scripts.
    crontab -l 2>/dev/null | grep -vFx "$line_idx" | grep -vFx "$line_hc" > "$tmp" || true
    printf '%s\n' "$line_idx" >> "$tmp"
    printf '%s\n' "$line_hc" >> "$tmp"
    crontab "$tmp"
    rm -f "$tmp"
    echo "cron installed: */15 * * * * update_index.py"
    echo "cron installed: */5  * * * * healthcheck.sh"
}

nginx_server_name() {
    # A certificate is only valid for a named vhost; "_" (the catch-all) is the
    # right server_name only while there is no domain.
    if [ -n "$LE_DOMAIN" ]; then
        echo "$LE_DOMAIN"
    else
        echo "_"
    fi
}

# True when the certificate pair nginx needs is actually usable. This is the
# whole meaning of NGINX_TLS=auto, so it has to mean "usable", not "present":
#
#   * both halves, as regular non-empty files. certbot writes exactly
#     fullchain.pem and privkey.pem and nginx reads exactly those; a directory
#     that exists, or a half-written pair, is not something to point a live
#     server at. A config naming an unreadable certificate is rejected by
#     `nginx -t`, which means the rollback path, which means an outage.
#   * the leaf not expired, so a stale certificate is re-issued rather than
#     kept. An expired certificate still loads, so this is not a load-time
#     failure, but serving one is precisely the broken-TLS state this stage
#     exists to end -- and answering "off" here is only ever a pre-flight
#     verdict, because certbot runs immediately afterwards and puts TLS back.
#
# Without openssl the expiry cannot be established, so the file test stands
# alone. Guessing "not valid" there would downgrade a working HTTPS site
# because a tool is missing, which is the failure this whole check prevents.
nginx_tls_cert_valid() {
    [ -n "$LE_DOMAIN" ] || return 1
    [ -f "$LE_CERT" ] && [ -s "$LE_CERT" ] || return 1
    [ -f "$LE_KEY" ] && [ -s "$LE_KEY" ] || return 1
    if have openssl; then
        openssl x509 -checkend 0 -noout -in "$LE_CERT" >/dev/null 2>&1 || return 1
    fi
    return 0
}

# Echoes exactly "on" or "off" so callers can use the result as a boolean
# instead of re-parsing NGINX_TLS themselves.
nginx_tls_mode() {
    case "$NGINX_TLS" in
        off|0|false|no) echo "off" ;;
        on|1|true|yes) echo "on" ;;
        # auto: TLS only once a domain is configured *and* its certificate is
        # really usable, so a fresh install keeps serving plain HTTP and a
        # re-run never downgrades a site that is already encrypted.
        *)
            if nginx_tls_cert_valid; then
                echo "on"
            else
                echo "off"
            fi
            ;;
    esac
}

# The proxy locations, shared verbatim by the plain-HTTP :80 server and the
# TLS :443 server so the two can never drift apart.
nginx_locations() {
    cat <<NGINX
    # Every API location must forward the client IP. The per-IP rate limiter on
    # /search, /facets, /analytics/click and /ready keys on this header; without
    # it every proxied request looks like 127.0.0.1 and the whole site shares a
    # single rate-limit bucket.
    location /search {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /health { proxy_pass http://127.0.0.1:$API_PORT; }
    location /live { proxy_pass http://127.0.0.1:$API_PORT; }
    location /ready {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /readyz {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /facets {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
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
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /analytics/click {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    location /analytics/summary { proxy_pass http://127.0.0.1:$API_PORT; }
    location /analytics/chat { proxy_pass http://127.0.0.1:$API_PORT; }
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

# The emitting half: the resolved mode arrives as a positional parameter and
# only the path/port knobs are read from the environment. It never consults
# NGINX_TLS and never probes the certificate store, so either mode can be
# rendered deterministically.
render_nginx_config() {
    local mode="$1"
    local name
    name="$(nginx_server_name)"
    {
        # Port 80 always serves the ACME challenge, in both modes: Let's Encrypt
        # validates over plain HTTP, so redirecting it away would break renewal.
        # certbot's webroot plugin reads the token from CERTBOT_WEBROOT, which
        # must therefore be PUBLIC_PORT-reachable.
        cat <<NGINX
server {
    listen $PUBLIC_PORT;
    server_name $name;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # "^~" makes this prefix location win over the catch-all "location /",
    # which is what keeps renewals working once that becomes a redirect.
    location ^~ /.well-known/acme-challenge/ {
        root $CERTBOT_WEBROOT;
        default_type text/plain;
    }
NGINX
        if [ "$mode" = "on" ]; then
            # Everything else goes to https, host and path preserved. Quoted
            # heredoc so $host/$request_uri stay for nginx to expand.
            cat <<'NGINX'

    location / { return 301 https://$host$request_uri; }
NGINX
        else
            echo
            nginx_locations
        fi
        echo "}"
        if [ "$mode" = "on" ]; then
            # "ssl http2" on the listen line works on both old and new nginx
            # (the standalone "http2 on;" directive needs nginx >= 1.25).
            # Mozilla intermediate settings, none of which need a resolver.
            # No Strict-Transport-Security here on purpose: the Next.js config
            # emits that header, and two sources of truth for max-age drift.
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

# The complete site config on stdout. Refuses to emit a config that points at a
# certificate that cannot be read: nginx -t would reject it and the reload
# would fail, so the gate is here, before anything is written.
nginx_site_config() {
    local mode
    mode="$(nginx_tls_mode)"
    if [ "$mode" = "on" ]; then
        # Both halves, not just the certificate: nginx reads privkey.pem from
        # the very same config, so a missing key fails `nginx -t` exactly like
        # a missing certificate does. NGINX_TLS=on is the only way to get here
        # without nginx_tls_cert_valid having already checked both.
        if [ ! -r "$LE_CERT" ]; then
            echo "ERROR: TLS is on but no readable certificate at $LE_CERT." >&2
            echo "       Get one with: LE_DOMAIN=... LE_EMAIL=... ./setup.sh tls" >&2
            echo "       Or go back to HTTP with: NGINX_TLS=off ./setup.sh nginx" >&2
            return 1
        fi
        if [ ! -r "$LE_KEY" ]; then
            echo "ERROR: TLS is on but no readable private key at $LE_KEY." >&2
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
    # A domain configured without a certificate is the silent-plaintext failure
    # mode this stage exists to prevent, so say so loudly instead of quietly
    # TLS_BOOTSTRAP=1 marks the deliberate pass-through to plain HTTP that
    # run_tls makes before certbot runs; that one is not the silent failure.
    if [ "$mode" = "off" ] && [ -n "$LE_DOMAIN" ] && [ "${TLS_BOOTSTRAP:-0}" != "1" ]; then
        echo "WARNING: LE_DOMAIN=$LE_DOMAIN is configured but there is no readable" >&2
        echo "         certificate at $LE_CERT, so this site is being served over" >&2
        echo "         plain HTTP. Passwords, bearer tokens and chat content will" >&2
        echo "         cross the wire in cleartext until that file exists." >&2
        echo "         Fix it with: LE_DOMAIN=$LE_DOMAIN LE_EMAIL=you@example.com ./setup.sh tls" >&2
    fi
    local tmp
    tmp="$(mktemp)"
    # Render to a temp file first: nothing reaches the live site until nginx has
    # accepted the config.
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
        # Put the previous config back instead of leaving a rejected one behind.
        echo "ERROR: nginx rejected the new config; rolling back." >&2
        if [ -f "$NGINX_CONF.bak" ]; then
            sudo mv -f "$NGINX_CONF.bak" "$NGINX_CONF"
        else
            sudo rm -f "$NGINX_CONF"
        fi
        return 1
    fi
    # reload, not restart: existing connections and in-flight requests survive.
    sudo systemctl reload nginx
    if [ "$mode" = "on" ]; then
        echo "nginx configured on port $PUBLIC_PORT with TLS (https://$LE_DOMAIN/)"
    else
        echo "nginx configured on port $PUBLIC_PORT (plain HTTP)"
    fi
    echo "roll back to plain HTTP: NGINX_TLS=off ./setup.sh nginx"
    # Last line on purpose: the operator must be left knowing, in one glance,
    # whether the site they now serve is encrypted.
    if [ "$mode" = "on" ]; then
        echo "serving: https via $LE_DOMAIN"
    elif [ -n "$LE_DOMAIN" ]; then
        echo "serving: http only (no certificate at $LE_CERT)"
    else
        echo "serving: http only (no domain configured; set LE_DOMAIN and run './setup.sh tls')"
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
    # The challenge must be servable BEFORE certbot asks Let's Encrypt to fetch
    # it. On a host whose nginx config predates the ACME location, the token
    # would fall through "location /" to Next.js, 404, and validation would fail
    # on the very first run. So the config is installed first, and it is what
    # makes "./setup.sh tls" work standalone.
    #
    # "auto", not "off": the mode-80 server serves the ACME challenge in BOTH
    # modes (only "location /" becomes a redirect, and "^~" outranks it), so
    # dropping to plain HTTP is only ever needed on a first run. Forcing "off"
    # here rewrote a live HTTPS site to cleartext and reloaded nginx even when a
    # perfectly good certificate was already on disk, so a certbot hiccup
    # during a routine re-run left the site in cleartext. Under "auto" an
    # existing, usable certificate is kept and a missing one still falls back
    # to plain HTTP for the challenge.
    local preflight
    NGINX_TLS=auto
    preflight="$(nginx_tls_mode)"
    TLS_BOOTSTRAP=1 run_nginx || return 1
    sudo mkdir -p "$CERTBOT_WEBROOT/.well-known/acme-challenge"
    # webroot, never --standalone: --standalone needs port 80 free, so on the
    # live site it would fail with the port taken (or force nginx to stop and
    # take the site down). webroot only needs the challenge location nginx
    # already serves. --keep-until-expiring makes a re-run a no-op instead of
    # burning the Let's Encrypt rate limit.
    if ! sudo certbot certonly \
        --webroot -w "$CERTBOT_WEBROOT" \
        --cert-name "$LE_DOMAIN" -d "$LE_DOMAIN" \
        --email "$LE_EMAIL" --agree-tos --non-interactive \
        --keep-until-expiring \
        --deploy-hook 'systemctl reload nginx'; then
        # Report the posture that is actually installed, not a fixed one: on a
        # re-run the pre-flight above kept the existing TLS config, and telling
        # the operator the site is on plain HTTP is the one thing they must not
        # be left believing here.
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
    # The certificate is on disk now, so the :443 server can be rendered.
    NGINX_TLS=on
    run_nginx
    if systemctl is-enabled certbot.timer >/dev/null 2>&1; then
        echo "renewal: handled by the systemd certbot.timer"
    else
        # Same idempotent pattern as run_cron: drop only our exact line and keep
        # the operator's other entries. Root's crontab, because the
        # certificate lives in LE_ROOT and certbot needs write access there.
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
            deps|backend|index|frontend|services|pm2-startup|stop-backend|stop-frontend|stop|cron|nginx|tls) STAGES+=("$1") ;;
            *) echo "unknown stage: $1"; usage; return 1 ;;
        esac
        shift
    done

    if [ "$ALL" -eq 1 ]; then
        # tls is deliberately not part of "all": it needs a domain, an email and
        # network access that an unattended bootstrap must not require.
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
            stop-backend) run_stop_backend ;;
            stop-frontend) run_stop_frontend ;;
            stop) run_stop ;;
            cron) run_cron ;;
            nginx) run_nginx ;;
            tls) run_tls ;;
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
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
