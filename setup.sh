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
# certificate pair is there to be served (see nginx_tls_cert_valid),
# plain HTTP otherwise.
NGINX_TLS="${NGINX_TLS:-auto}"
LE_LIVE="$LE_ROOT/live/$LE_DOMAIN"
LE_CERT="$LE_LIVE/fullchain.pem"
LE_KEY="$LE_LIVE/privkey.pem"

# LE_DOMAIN is env-only and is never written anywhere, so an ordinary
# `./setup.sh nginx` (or the nginx stage inside `./setup.sh all`) in a shell
# that does not export it used to see no domain at all, resolve "auto" to
# plain HTTP, and rewrite a live HTTPS site to cleartext -- silently, with
# exit 0. When it is unset, recover the domain from the certificate the
# INSTALLED CONFIG already names, and from nowhere else.
#
# "And from nowhere else" is the whole safety argument. Recovery is only ever
# justified by evidence that the certificate belongs to this site, and the one
# piece of that evidence available is the config this site is already serving.
# Falling back to "the only directory under $LE_ROOT/live" looked equivalent
# and is not: /etc/letsencrypt is shared, so on a host where this site was
# never on TLS it would adopt an unrelated service's cert-name and repoint
# both server_name and ssl_certificate at that other domain -- serving a
# certificate for a domain this site does not answer for. Guessing is only
# safe when the guess cannot be acted on, and here it very much can.
#
# Pure bash on purpose: this runs while the script is sourced, and
# `test_refuses_when_certbot_is_missing` sources it with almost nothing on
# PATH.
# Forced to 0 here rather than defaulted later: it describes what this block
# did, so an inherited value from the environment must not leak into it.
LE_DOMAIN_RECOVERED=0
if [ -z "$LE_DOMAIN" ]; then
    _le_name=""
    if [ -r "$NGINX_CONF" ]; then
        _le_line=""
        while IFS= read -r _le_line || [ -n "$_le_line" ]; do
            # ltrim, then match the directive itself
            while [ "${_le_line# }" != "$_le_line" ]; do _le_line="${_le_line# }"; done
            while [ "${_le_line#	}" != "$_le_line" ]; do _le_line="${_le_line#	}"; done
            case "$_le_line" in
                "ssl_certificate "*)
                    _le_path="${_le_line#ssl_certificate }"
                    _le_path="${_le_path%%;*}"
                    while [ "${_le_path% }" != "$_le_path" ]; do _le_path="${_le_path% }"; done
                    # .../live/<cert-name>/fullchain.pem -> <cert-name>
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
        # So the nginx stage can say "recovered from the installed config"
        # rather than implying the operator configured it in this shell.
        LE_DOMAIN_RECOVERED=1
    fi
    unset _le_name _le_line _le_path _le_dir _le_parent
fi

# pm2 process tuning. These MUST stay equal to the values in
# ecosystem.config.js — backend/tests/test_deploy_config.py fails if the two
# process definitions disagree, because `./setup.sh services` re-registers
# pm2 from this file and would otherwise silently drop the OOM auto-restart
# guard that ecosystem.config.js declares.
API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"
API_MAX_RESTARTS="${API_MAX_RESTARTS:-10}"
FRONTEND_MAX_MEMORY="${FRONTEND_MAX_MEMORY:-1G}"
RESTART_BACKOFF_MS="${RESTART_BACKOFF_MS:-100}"


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
  API_MAX_MEMORY API_MAX_RESTARTS FRONTEND_MAX_MEMORY RESTART_BACKOFF_MS
     pm2 process tuning; must match ecosystem.config.js (tests enforce it)
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
        --name vccircle-backend \
        --max-memory-restart "$API_MAX_MEMORY" \
        --max-restarts "$API_MAX_RESTARTS" \
        --exp-backoff-restart-delay "$RESTART_BACKOFF_MS" \
        -- -m gunicorn \
        -k uvicorn.workers.UvicornWorker \
        --workers "$GUNICORN_WORKERS" --bind "127.0.0.1:$API_PORT" \
        --timeout 120 app.main:app)
    wait_http "http://localhost:$API_PORT/health"

    (cd frontend && pm2 start "$SCRIPT_DIR/frontend/node_modules/.bin/next" \
        --name vccircle-frontend \
        --max-memory-restart "$FRONTEND_MAX_MEMORY" \
        --max-restarts "$API_MAX_RESTARTS" \
        --exp-backoff-restart-delay "$RESTART_BACKOFF_MS" \
        -- start -p "$NEXT_PORT")
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

# True when a certificate pair is there to be served: LE_DOMAIN configured, and
# both halves present as regular non-empty files. certbot writes exactly
# fullchain.pem and privkey.pem and nginx reads exactly those, so a directory
# that exists, or a half-written pair from an interrupted run, is not something
# to point a live server at.
#
# Deliberately NOT part of this: whether the invoking user can READ the files,
# and whether the leaf has expired.
#
#   * Readability. This runs as the operator, but `nginx -t` runs as root.
#     certbot writes privkey.pem 0600 root:root, so a readability test here
#     refuses to emit a config that nginx would load happily -- and then points
#     the operator at the very command they just ran. `nginx -t` is the
#     authority on readability, and it is already the gate that rolls back.
#   * Expiry. A lapsed certificate still loads; nothing breaks. Dropping the
#     :443 server for one trades a browser warning for cleartext, and this
#     function is reachable from `./setup.sh nginx` and `./setup.sh all`,
#     where no certbot ever runs to put TLS back. So a lapsed certificate is
#     reported by nginx_tls_cert_expired and renewed by `./setup.sh tls` (a
#     certificate that has already lapsed is "until expiring" to certbot) --
#     never used as a reason to go quiet on the wire.
nginx_tls_cert_valid() {
    [ -n "$LE_DOMAIN" ] || return 1
    [ -f "$LE_CERT" ] && [ -s "$LE_CERT" ] || return 1
    [ -f "$LE_KEY" ] && [ -s "$LE_KEY" ] || return 1
    return 0
}

# The full state of the leaf, as one word. Reporting only -- it never feeds
# nginx_tls_mode, because no state of a certificate is a reason to remove a
# live HTTPS server (see nginx_tls_mode).
#
# "unreadable" and "corrupt" are kept apart from "expired" on purpose.
# `openssl x509 -checkend` exits non-zero when the certificate has expired, when
# it will not parse, and when it cannot be opened at all, and the three need
# different advice. So the certificate is parsed FIRST (-enddate must print
# something) and only then is its expiry asked about:
#
#   missing     no file at all
#   empty       present but zero bytes (an interrupted certbot)
#   unreadable  present, non-empty, and not readable by whoever is running this
#               -- certbot's privkey is 0600 root:root, so this is normal, and
#               it is emphatically NOT expired. nginx runs as root and will
#               decide.
#   corrupt     readable, non-empty, and openssl cannot parse it. --keep-until-
#               expiring will not fix this, so it needs its own remedy.
#   unknown     no openssl to ask. The absence of a tool is not evidence.
#   expired     parsed, and past its notAfter
#   current     parsed, and in date
nginx_tls_cert_state() {
    if [ ! -f "$LE_CERT" ]; then echo "missing"; return 0; fi
    if [ ! -s "$LE_CERT" ]; then echo "empty"; return 0; fi
    if [ ! -r "$LE_CERT" ]; then echo "unreadable"; return 0; fi
    if ! have openssl; then echo "unknown"; return 0; fi
    if [ -z "$(openssl x509 -noout -enddate -in "$LE_CERT" 2>/dev/null)" ]; then
        echo "corrupt"
        return 0
    fi
    # -checkend exits non-zero when the certificate HAS expired, so the verdict
    # is the other way round from what it looks like.
    if openssl x509 -checkend 0 -noout -in "$LE_CERT" >/dev/null 2>&1; then
        echo "current"
    else
        echo "expired"
    fi
}

# True when the config currently installed is already serving :443. Used to
# make the "auto" default prefer the status quo.
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

# Echoes exactly "on" or "off" so callers can use the result as a boolean
# instead of re-parsing NGINX_TLS themselves.
#
# "auto" is deliberately status-quo-biased: if the config already installed is
# serving :443, it keeps serving :443. Removing TLS is something an operator
# does on purpose with NGINX_TLS=off, not something a routine re-run does
# because a probe came back empty. Every input this function cannot fully
# vouch for -- an unset domain, a key this user cannot read, a certificate
# openssl will not parse, a missing openssl, an ambiguous LE_ROOT -- is "I do
# not know", and "I do not know" now means "do not change what is serving".
# That is the whole class of bug, not one instance of it: three separate
# downgrade paths came from deciding the posture from transient state at the
# moment the nginx stage happened to run.
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
# certificate pair that is not there at all: `nginx -t` would reject it and the
# reload would fail, so the gate is here, before anything is written.
#
# Existence and non-emptiness, NOT readability. This gate runs as whoever
# invoked the script, while `nginx -t` runs as root and certbot writes
# privkey.pem 0600 root:root, so testing readability here refused to emit a
# config that nginx loads perfectly well -- and the error told the operator to
# run the command they had just run. `nginx -t` is the authority on what nginx
# can read, and it is already the gate that rolls back.
nginx_site_config() {
    local mode
    mode="$(nginx_tls_mode)"
    if [ "$mode" = "on" ]; then
        # Both halves: nginx reads privkey.pem from the very same config, so a
        # missing key fails `nginx -t` exactly like a missing certificate.
        # NGINX_TLS=on is the only way to get here without nginx_tls_cert_valid
        # having already checked both.
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
    # Report what this site is actually going to serve, before it serves it.
    # Gated on the certificate being PRESENT rather than on LE_DOMAIN being
    # set: the interesting case is a pair on disk that is not being served, and
    # a recovered domain would otherwise silence the very warning that matters.
    local state
    state="$(nginx_tls_cert_state)"
    if [ "$mode" = "off" ]; then
        # TLS_BOOTSTRAP=1 marks the pre-flight run_tls makes before certbot
        # runs. Only THIS advisory is silenced there: certbot is genuinely
        # about to make "you are on plain HTTP" irrelevant, so saying it would
        # be noise. The diagnoses below are not silenced, because a corrupt or
        # lapsed file is exactly what certbot will NOT fix -- it is issued
        # with --keep-until-expiring and skips anything it cannot read -- and
        # the pre-flight is often the first run that notices.
        # Braces, because `A || B && C` relies on left-associativity to mean
        # `(A || B) && C` and reads as something else entirely.
        if { [ -n "$LE_DOMAIN" ] || [ "$state" != "missing" ]; } && [ "${TLS_BOOTSTRAP:-0}" != "1" ]; then
            # "no readable certificate" was wrong twice over: readability is
            # not what this mode is decided on, and the certificate is often
            # there and merely unusable. Name the pair instead.
            echo "WARNING: this site is being served over plain HTTP." >&2
            echo "         No certificate and private key were found at" >&2
            echo "           $LE_CERT" >&2
            echo "           $LE_KEY" >&2
            echo "         Passwords, bearer tokens and chat content will cross" >&2
            echo "         the wire in cleartext until that pair exists." >&2
            echo "         Fix it with: LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
        fi
    elif [ "$state" = "expired" ]; then
        # Reporting, not a mode change. The site stays on HTTPS: a lapsed
        # certificate is a browser warning, and removing the :443 server
        # over one would trade that warning for cleartext -- on a path
        # (`./setup.sh nginx`, `./setup.sh all`) where nothing renews it.
        echo "WARNING: the certificate at $LE_CERT has expired." >&2
        echo "         The site is still being served over HTTPS, but browsers" >&2
        echo "         will warn and clients may refuse the connection." >&2
        echo "         Renew it with: LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    elif [ "$state" = "corrupt" ]; then
        # Its own remedy, because re-running the tls stage will NOT fix
        # this. The stage issues with --keep-until-expiring, which leaves
        # anything certbot cannot parse exactly as it is -- so the operator
        # would loop forever on a command that keeps reporting success.
        echo "WARNING: the certificate at $LE_CERT cannot be read as a" >&2
        echo "         certificate at all. This is not an expiry problem and" >&2
        echo "         re-running the tls stage will not fix it: issuance uses" >&2
        echo "         --keep-until-expiring, so certbot skips a file it cannot" >&2
        echo "         parse and leaves it untouched. Remove it, then renew:" >&2
        echo "           sudo rm -f $LE_CERT" >&2
        echo "           LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    fi
    # "unreadable" and "unknown" are silent on purpose. An unreadable
    # certificate is normal -- certbot's key is 0600 root:root -- and nginx
    # runs as root and will have the last word via `nginx -t`, which
    # triggers the rollback if the certificate really is unusable. Claiming
    # expiry there would be the false warning this case exists to avoid.
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
    # whether the site they now serve is encrypted. The domain may have been
    # recovered rather than configured, and saying so is the difference between
    # "I know my domain" and "I found a certificate and guessed it is mine".
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
    #
    # This pre-flight is BEST EFFORT, which is a different policy from the one
    # run_nginx applies to itself, and deliberately so. run_nginx exists to put
    # a config in front of a live site, so when it cannot render one it stops
    # and says so. This step does not exist to manage the config -- it exists to
    # obtain a certificate. It also cannot improve on what is already there:
    # when the status-quo default resolves to TLS and the pair is gone or
    # unreadable, the config already installed is a TLS config, and a TLS config
    # serves the ACME challenge exactly as well as a plain one. So a failure
    # here means "leave it alone", not "give up" -- and aborting is what made
    # a live HTTPS site whose certificate had been deleted impossible to
    # re-provision: the stage died before certbot ran and told the operator to
    # run the very command they had just run.
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
