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
# The SSE chat stream only: it has no application-layer limit, while a second edge
# limiter on the already-limited routes would shadow PUBLIC_*_RATE_PER_MIN.
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

# Unset LE_DOMAIN recovers the domain from the certificate the INSTALLED config
# names -- the only evidence that the cert is this site's. Never from "the only
# dir under $LE_ROOT/live": /etc/letsencrypt is shared, and that would adopt
# another service's cert. Pure bash: this runs while the script is sourced.
# Forced to 0 rather than defaulted later: it records what this block did, so an
# inherited value from the environment must not leak into it.
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
        LE_DOMAIN_RECOVERED=1
    fi
    unset _le_name _le_line _le_path _le_dir _le_parent
fi

# These MUST stay equal to ecosystem.config.js; run_services EXPORTS them so both
# start paths produce the same process.
# min_uptime: pm2 defaults it to 1000ms, so a frontend that starts cleanly and
# dies seconds later counts as STABLE -- never unstable, so its backoff and
# restart limit never act.
API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"
API_MAX_RESTARTS="${API_MAX_RESTARTS:-10}"
FRONTEND_MAX_MEMORY="${FRONTEND_MAX_MEMORY:-1G}"
RESTART_BACKOFF_MS="${RESTART_BACKOFF_MS:-100}"
MIN_UPTIME_MS="${MIN_UPTIME_MS:-30000}"


# Digest, not tag: a tag is a mutable name, so a re-pointed registry is pulled
# silently. Qdrant must be >= the version that wrote an existing collection.
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.19.0@sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine@sha256:858f009f9709ce576febc734aa78b8f6d624b82571f9ddb6bda4377c833b3499}"

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
  logrotate  render deploy/logrotate.conf for THIS host + install the policy

  all        deps backend index frontend services pm2-startup cron nginx

env overrides:
  QDRANT_PORT REDIS_PORT API_PORT NEXT_PORT PUBLIC_PORT GUNICORN_WORKERS
  API_MAX_MEMORY API_MAX_RESTARTS FRONTEND_MAX_MEMORY RESTART_BACKOFF_MS
  MIN_UPTIME_MS
     pm2 process tuning; must match ecosystem.config.js (tests enforce it)
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
    # sha256 read from the manifest published beside the tarball: this checks
    # INTEGRITY, not authenticity (same origin, unsigned). It is read rather than
    # hardcoded so it stays correct across the LTS auto-discovery above.
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
        # Pinned: pm2 supervises both services, so an unattended bootstrap must not
        # land a new major on a live box.
        npm install -g --no-audit --no-fund "pm2@$PM2_VERSION" >/tmp/pm2-install.log 2>&1 || {
            echo "pm2 install failed:" >&2; tail -3 /tmp/pm2-install.log >&2; return 1
        }
        local nbin
        nbin="$(npm prefix -g)/bin"
        mkdir -p ~/.local/bin
        ln -sf "$nbin/pm2" ~/.local/bin/pm2
        ln -sf "$nbin/pm2-dev" ~/.local/bin/pm2-dev
    fi
    # `have pm2` short-circuits, so the pin never downgrades what is installed;
    # this stage installs and warns. Moving a live process manager is an operator
    # decision.
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

# True when every published host port binds to 127.0.0.1; a "" HostIp binds
# 0.0.0.0, which is reachable from the network, and fails.
container_binds_localhost() {
    if docker inspect -f \
        '{{range $k, $v := .HostConfig.PortBindings}}{{range $v}}{{if ne .HostIp "127.0.0.1"}}PUBLIC_BIND{{end}}{{end}}{{end}}' \
        "$1" 2>/dev/null | grep -q PUBLIC_BIND; then
        return 1
    fi
    return 0
}
# Hex, not base64: never sent over the network, but pasted into a URL, a docker
# argv and an unquoted shell.
random_secret() {
    if have openssl; then
        openssl rand -hex 32
        return
    fi
    "$VENV_PY" -c 'import secrets; print(secrets.token_hex(32))'
}

# Read from .env first and generated only when absent: regenerating would rotate
# the password out from under the running container, and every request would then
# be rejected for a reason invisible from outside. Storing a generated secret is
# what lets ensure_docker_container detect a container provisioned before
# credentials existed.
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

# Does the RUNNING container's own config carry this exact string? It answers "is
# this container the one this .env describes", catching both a box provisioned
# before credentials existed (app then fails every request with NOAUTH) and a
# password edited in .env. Cmd and Env are both searched because redis takes a
# command flag and qdrant an environment variable.
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

# ensure_docker_container <name> <image> <ports> <volume> <envargs> <cmd> <needle>
# needle is the literal whose presence in the existing container proves its auth
# already matches this run; empty disables the check.
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

# One KEY from a KEY=VALUE .env, quotes and trailing CR stripped. Prints nothing
# when absent. Last match wins, matching python-dotenv on a repeated key.
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

# Secrets and user data at rest: .env holds the JWT secret and the database
# credentials, the SQLite files hold every account and chat message, and all are
# world-readable by default. DB paths come from .env because the app resolves a
# relative one against the backend working directory. Called from BOTH ends of the
# lifecycle: ./setup.sh services never re-runs the backend stage.
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
        # A database that does not exist yet is not an error: the first request creates it.
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

# Appends the shipped default when the key is absent; warns, never rewrites, when
# the operator forced trust from ANY peer -- then a client reaching :8001 directly
# can forge X-Forwarded-For and dodge the rate limit. (A forced true is correct
# when the proxy runs on another host; the operator's value is theirs.)
# The greps match the SHAPE load_dotenv accepts, not `KEY=value`: the app strips
# blanks, an `export` prefix and quotes, so a forced true written as
# `AUTH_TRUST_X_FORWARDED_FOR = "true"` is invisible to a `^KEY=` guard -- and
# setup.sh then appends a SECOND key that python-dotenv resolves last, silently
# downgrading the operator's value into the branch meant to warn about it.
# The three spellings are config._TRUE_SPELLINGS and must agree in BOTH
# directions; copied in shell because this runs before the venv exists.
migrate_xff_trust() {
    local env_file="$1"
    local assignment='^[[:space:]]*(export[[:space:]]+)?AUTH_TRUST_X_FORWARDED_FOR[[:space:]]*='
    # The two quotes must match (`"true'` is a dotenv parse error, not a forced
    # True), and the tails differ: dotenv takes a trailing comment after at least
    # one blank on a bare value, and none inside quotes.
    # `.` and not `[^\r\n]`: a backslash inside a POSIX bracket expression is a
    # literal, so that class excluded `r` and `n` too.
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

    # Order is load-bearing: the containers need the credentials to start AT ALL;
    # the credentials are generated with the interpreter (openssl is not
    # guaranteed), so the venv must exist first; and .env must exist before
    # either, since store_secrets reads it and only generates when the key is
    # absent -- that read is what keeps a re-run from rotating the password out
    # from under a running container.
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

    # qdrant takes its API key as an environment variable, redis its password as
    # a command flag, so the two needles differ in shape.
    ensure_docker_container qdrant "$QDRANT_IMAGE" \
        "127.0.0.1:$QDRANT_PORT:6333" "-v $SCRIPT_DIR/qdrant_data:/qdrant/storage" \
        "-e QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY" "" \
        "QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY"
    ensure_docker_container redis "$REDIS_IMAGE" "127.0.0.1:$REDIS_PORT:6379" "" "" \
        "redis-server --requirepass $REDIS_PASSWORD" \
        "--requirepass $REDIS_PASSWORD"
    # The qdrant healthz endpoint is unauthenticated by design, so this probe
    # still works with the key set.
    wait_http "http://localhost:$QDRANT_PORT/healthz"

    "$VENV_PY" -m pip install -q --upgrade pip
    "$VENV_PY" -m pip install -q -r backend/requirements.txt

    # The password is carried IN the URL rather than beside it, because every
    # redis client in the app is built from config.REDIS_URL and none of them
    # take a separate password argument, so the URL is the single place that
    # makes all of them authenticate. A URL that already carries credentials is
    # left alone.
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

    # An .env predating these per-IP limits has no trust setting at all, so every
    # proxied request keys on the nginx peer and the whole site shares one
    # rate-limit bucket.
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

# Names why the readiness gate rejected the deploy, so a 30s timeout is not a bare
# "not ready". No -f here: curl -f suppresses the body on exactly the >= 400 (or
# failed-connection) responses this only ever runs on, which made it dead code.
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

    # Start from ecosystem.config.js, the single definition of these two processes.
    # The reason is min_uptime: pm2's CLI has no `--min-uptime` in any released
    # version, while `min_uptime` in an ecosystem file IS honoured at runtime
    # (lib/God.js). Under `set -e` the flag would abort this stage and leave the
    # box with neither service running. Every knob is exported rather than left to
    # the ecosystem fallback, so the overrides work on the path that starts them.
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

    # Readiness, not liveness: /health is a stub that answers 200 with a dead
    # Qdrant client, unloaded models or the placeholder GEMINI_API_KEY.
    # /ready/deep is the uncached, unrated, loopback-only form, so this gate
    # cannot pass on a warm readiness cache or be throttled into a false "not
    # ready". Placed LAST because it can legitimately fail and `set -e` aborts on
    # it: running it before the frontend started turned a misconfigured key into
    # a torn-down deployment. The verdict is the app's own classifier, printed by
    # report_readiness_reason -- a shell copy of its sentinel list would drift.
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
    # update_index.py takes its own flock(2) on data/update.lock, so an external
    # `flock -n` wrapper would make every run skip.
    local line_idx="*/15 * * * * nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"
    # cron inherits no shell environment, so the webhook URL and BASE are passed
    # explicitly (an empty webhook is harmless). Without BASE, an operator who
    # overrode API_PORT would have the watchdog probe a closed port and restart a
    # healthy backend every 5 minutes.
    local line_hc="*/5 * * * * BASE=\"http://localhost:$API_PORT\" HEALTHCHECK_WEBHOOK_URL=\"${HEALTHCHECK_WEBHOOK_URL:-}\" LOG=$hc_log $SCRIPT_DIR/deploy/healthcheck.sh"
    local tmp
    tmp="$(mktemp)"
    # The healthcheck line is reclaimed by SCRIPT PATH, not by exact match: an
    # entry written by an older revision stopped matching the literal it had to be
    # deleted by, survived every run, and on a non-default API_PORT carried no
    # BASE=, so it probed a closed port and was read as "not alive" -- a pm2
    # restart against a healthy backend every five minutes.
    # ACCEPTED COST: a user's own healthcheck.sh line on a CUSTOM schedule is
    # replaced by the managed one. NOT touched: lines that run neither script, and
    # COMMENTED lines -- commenting the entry out is how an operator disables the
    # watchdog, and deleting that marker would silently re-arm one. The indexer
    # line is still matched exactly, so a hand-edited variant survives.
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

# deploy/logrotate.conf is a TEMPLATE: this host's log directories and the account
# logrotate drops privileges to are not knowable when the file is written.
# The three rewrites are anchored to the whole line -- a free-floating
# substitution would install a half-correct policy that rotates nothing while
# logrotate reports success -- and the guard checks the TEMPLATE, never the
# output, so a drifted template installs nothing at all.
# The third rewrite UNCOMMENTS the `su` directive, which ships disabled because
# the file is copied by hand as well as rendered: without it every rotation runs
# as root.
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
    # Both exit paths remove the temp file: a failed render must not leave the
    # rendered policy lying in /tmp.
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
    # A certificate is only valid for a named vhost, so "_" is right only while
    # there is no domain.
    if [ -n "$LE_DOMAIN" ]; then
        echo "$LE_DOMAIN"
    else
        echo "_"
    fi
}

# True when a certificate pair is there to serve: domain set, both halves present
# as regular non-empty files. A directory, or a half-written pair from an
# interrupted certbot run, is not something to point a live server at.
# Deliberately NOT readability: this runs as the operator while `nginx -t` runs
# as root and certbot writes privkey.pem 0600 root:root, so a readability test
# here refuses a config nginx would load happily -- `nginx -t` is the authority
# and is already the gate that rolls back.
# Deliberately NOT expiry: a lapsed certificate still loads, and dropping :443
# for one trades a browser warning for cleartext on a path where nothing renews
# it. It is reported by nginx_tls_cert_state instead.
nginx_tls_cert_valid() {
    [ -n "$LE_DOMAIN" ] || return 1
    [ -f "$LE_CERT" ] && [ -s "$LE_CERT" ] || return 1
    [ -f "$LE_KEY" ] && [ -s "$LE_KEY" ] || return 1
    return 0
}

# The leaf's state as one word. Reporting only: no state is a reason to remove a
# live HTTPS server. "unreadable" is normal (certbot's key is 0600 root:root) and
# is emphatically NOT expired.
# `openssl x509 -checkend` exits non-zero when the certificate HAS expired, when
# it will not parse and when it cannot be opened at all, so it is parsed FIRST
# (-enddate must print something) and only then asked about expiry: that is what
# keeps "corrupt" apart from "expired".
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

# True when the config currently installed is already serving :443, so "auto"
# prefers the status quo.
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

# Echoes exactly "on" or "off" so callers need not re-parse NGINX_TLS.
# "auto" is status-quo-biased: if the installed config already serves :443 it
# keeps serving :443. Removing TLS is an operator's deliberate NGINX_TLS=off,
# never a routine re-run reacting to a probe -- every input this cannot fully
# vouch for (unset domain, unreadable key, unparseable cert, no openssl,
# ambiguous LE_ROOT) means "do not change what is serving".
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
    #
    # Every API location must ALSO forward the public Host, for the same
    # reason: nginx's default proxy Host is \$proxy_host, i.e. the backend
    # address, so without this the app sees "Host: 127.0.0.1:8001" instead of
    # the host the browser addressed. The CSRF guard compares Origin against
    # Host, and browsers send Origin on same-origin unsafe requests too, so a
    # missing Host header makes every cookie-authenticated POST 403 -- which
    # took POST /recommend/interaction down for every user. Set it on all of
    # them, not just /api, so the next location added cannot repeat it.
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
    # The "location /ready" block above is a PREFIX match, so without this block
    # a public GET /ready/deep would be proxied to the API and held dark only by
    # its in-process host-local check. That check refuses everything arriving
    # through this vhost (nginx always sets X-Forwarded-For), but the endpoint is
    # deliberately uncached and unrated, so it gets a second, independent layer:
    # 404 at the proxy. The deploy gate in this script and the watchdog in
    # deploy/healthcheck.sh reach it on 127.0.0.1 directly and never come here.
    # NB: no backticks in this comment -- the heredoc is unquoted, so they would
    # be run as a command substitution.
    location /ready/deep { return 404; }
    location /facets {
        proxy_pass http://127.0.0.1:$API_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
    # The SSE chat stream, and only the SSE chat stream.
    #
    # This is the one expensive path with no application-layer limit: an open
    # stream holds a gunicorn worker and a client connection for up to 300s,
    # and nothing anywhere counts those. /search, /facets, /analytics/click and
    # /ready are already limited inside the application (public_rate_limit) and
    # the auth endpoints by _check_rate_limit, so a limiter on them here would
    # mean two layers emitting 429 with different bodies, and would make
    # PUBLIC_*_RATE_PER_MIN / AUTH_*_RATE_PER_MIN unreachable knobs.
    #
    # Keyed on \$binary_remote_addr because this is the edge: nginx is the only
    # layer that still has the real client address, and the application's own
    # limiter has to trust X-Forwarded-For to recover it.
    #
    # nodelay, because a burst here is a client starting a few streams at once
    # (regenerate, retry) rather than a flood. Without it nginx holds the burst
    # in a queue and releases it later, which delays exactly the requests that
    # are legitimate, while still leaving the worker pool pinned for as long as
    # the streams run.
    #
    # limit_req_status 429 rather than nginx's default 503: 429 is what the
    # application limiter already returns, so a client sees one status for "you
    # are going too fast" no matter which layer said it.
    #
    # proxy_read_timeout 300s is not optional -- the stream IS a 300s SSE
    # response, and nginx's 60s default would cut every single one of them
    # open, which breaks chat outright rather than merely failing to rate limit.
    #
    # "^~" so this prefix wins over the "location /api" below, which is the
    # longest matching prefix for the same requests and would otherwise be used.
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
        # Port 80 always serves the ACME challenge, in both modes: Let's Encrypt
        # validates over plain HTTP, so redirecting it away would break renewal.
        # limit_req_zone is http-scope and this file is included from inside
        # nginx's http block, so it is declared once outside every server:
        # per-server would give the HTTP and TLS servers independent buckets and a
        # client could double its rate by switching schemes.
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

    # "^~" makes this prefix location win over the catch-all "location /",
    # which is what keeps renewals working once that becomes a redirect.
    location ^~ /.well-known/acme-challenge/ {
        root $CERTBOT_WEBROOT;
        default_type text/plain;
    }
NGINX
        if [ "$mode" = "on" ]; then
            # Quoted heredoc so $host/$request_uri reach nginx unexpanded.
            cat <<'NGINX'

    location / { return 301 https://$host$request_uri; }
NGINX
        else
            echo
            nginx_locations
        fi
        echo "}"
        if [ "$mode" = "on" ]; then
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

# The complete site config on stdout. Refuses to point at a certificate pair that
# is not there: `nginx -t` would reject it and the reload would fail.
# Existence and non-emptiness, NOT readability -- `nginx -t` runs as root and is
# already the gate that rolls back.
nginx_site_config() {
    local mode
    mode="$(nginx_tls_mode)"
    if [ "$mode" = "on" ]; then
        # Both halves: a missing key fails `nginx -t` exactly like a missing
        # certificate.
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
    # Report the posture before serving it, gated on the certificate being PRESENT
    # rather than on LE_DOMAIN being set: the interesting case is a pair on disk
    # that is not being served.
    local state
    state="$(nginx_tls_cert_state)"
    if [ "$mode" = "off" ]; then
        # TLS_BOOTSTRAP=1 marks run_tls's pre-flight run before certbot; only THIS
        # advisory is silenced there. The diagnoses below are NOT: a corrupt or
        # lapsed file is exactly what certbot will not fix.
        # Braces, because `A || B && C` is left-associative and reads as
        # something else entirely.
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
        # Reporting, not a mode change: dropping :443 over a lapsed certificate
        # trades a browser warning for cleartext where nothing renews it.
        echo "WARNING: the certificate at $LE_CERT has expired." >&2
        echo "         The site is still being served over HTTPS, but browsers" >&2
        echo "         will warn and clients may refuse the connection." >&2
        echo "         Renew it with: LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    elif [ "$state" = "corrupt" ]; then
        # Its own remedy: --keep-until-expiring skips a file certbot cannot parse
        # and leaves it untouched, so the operator would loop on a command that
        # keeps reporting success.
        echo "WARNING: the certificate at $LE_CERT cannot be read as a" >&2
        echo "         certificate at all. This is not an expiry problem and" >&2
        echo "         re-running the tls stage will not fix it: issuance uses" >&2
        echo "         --keep-until-expiring, so certbot skips a file it cannot" >&2
        echo "         parse and leaves it untouched. Remove it, then renew:" >&2
        echo "           sudo rm -f $LE_CERT" >&2
        echo "           LE_DOMAIN=your.domain LE_EMAIL=you@example.com ./setup.sh tls" >&2
    fi
    # "unreadable" and "unknown" are silent on purpose: unreadable is normal
    # (certbot's key is 0600 root:root) and `nginx -t` runs as root, so claiming
    # expiry here is the false warning this case exists to avoid.
    local tmp
    tmp="$(mktemp)"
    # Nothing reaches the live site until nginx has accepted the config.
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
    # reload, not restart: existing connections and in-flight requests survive.
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
    # The challenge must be servable BEFORE certbot asks Let's Encrypt to fetch
    # it: on a host whose config predates the ACME location the token falls
    # through to Next.js, 404s, and validation fails on the very first run. So the
    # config is installed first, which is what makes "./setup.sh tls" standalone.
    # "auto", not "off": the mode-80 server serves the challenge in BOTH modes
    # (only "location /" becomes a redirect, and "^~" outranks it).
    # This pre-flight is BEST EFFORT, unlike run_nginx itself: run_nginx exists to
    # put a config in front of a live site and stops when it cannot render one,
    # while this exists to obtain a certificate and cannot improve on what is
    # already installed. A failure here means "leave it alone", not "give up".
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
    # webroot, never --standalone: --standalone needs port 80 free and would fail
    # with the port taken by the live site. --keep-until-expiring makes a re-run a
    # no-op instead of burning the Let's Encrypt rate limit.
    if ! sudo certbot certonly \
        --webroot -w "$CERTBOT_WEBROOT" \
        --cert-name "$LE_DOMAIN" -d "$LE_DOMAIN" \
        --email "$LE_EMAIL" --agree-tos --non-interactive \
        --keep-until-expiring \
        --deploy-hook 'systemctl reload nginx'; then
        # Report the posture actually installed: on a re-run the pre-flight above
        # kept the existing TLS config, and telling the operator the site is on
        # plain HTTP is the one thing they must not be left believing here.
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
        # Same idempotent pattern as run_cron, in root's crontab because the
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
            deps|backend|index|frontend|services|pm2-startup|stop-backend|stop-frontend|stop|cron|nginx|tls|logrotate) STAGES+=("$1") ;;
            *) echo "unknown stage: $1"; usage; return 1 ;;
        esac
        shift
    done

    if [ "$ALL" -eq 1 ]; then
        # tls is deliberately excluded: it needs a domain, an email and network
        # access an unattended bootstrap must not require. logrotate is excluded
        # too: it needs the logrotate binary, which a container or a fresh CI box
        # may lack, and its policy is host-specific, so rotation is an operator step.
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
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
