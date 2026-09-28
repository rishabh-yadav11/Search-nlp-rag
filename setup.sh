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
# nginx rate limit for the SSE chat stream only. The application already
# limits /search, /facets, /analytics/click and /ready per IP
# (public_rate_limit) and the auth endpoints (_check_rate_limit); adding a
# second limiter on those would mean two layers emitting 429 with different
# bodies and would make PUBLIC_*_RATE_PER_MIN / AUTH_*_RATE_PER_MIN
# unreachable. The chat stream has no application-layer limit at all, so the
# edge is the only place that can count it.
NGINX_CHAT_LIMIT_RATE="${NGINX_CHAT_LIMIT_RATE:-10r/m}"
NGINX_CHAT_LIMIT_BURST="${NGINX_CHAT_LIMIT_BURST:-10}"
# pm2 is the process manager for both long-running services, so an unpinned
# 'npm install -g pm2' means a new release can land on the box unattended and
# change how processes are started, restarted and reported. The version is an
# exact one, overridable so an operator can move it deliberately.
PM2_VERSION="${PM2_VERSION:-7.0.4}"
# The logrotate policy is host-specific: it names this machine's log
# directories and the account logrotate must drop privileges to, and both are
# wrong the instant the repo template is copied somewhere else. It is therefore
# rendered per host by the logrotate stage rather than copied, and this is
# where that render is installed.
LOGROTATE_CONF="${LOGROTATE_CONF:-/etc/logrotate.d/vccircle}"
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
# process definitions disagree. `./setup.sh services` starts pm2 from
# ecosystem.config.js and EXPORTS these to it, rather than passing them as
# `pm2 start` flags, so a default that differs here would produce a different
# process on the same host depending on which path started it.
#
# min_uptime needs a precise statement because the obvious one is wrong. pm2 is
# not missing the option by default: min_uptime defaults to 1000ms. At 1s, a
# Next.js frontend that starts cleanly and then dies four seconds later -- a
# port it cannot rebind after a half-dead previous process, a missing .next
# build, an OOM on the first render -- has comfortably cleared the bar, so pm2
# scores every one of those restarts as STABLE. Stable restarts never count
# toward max_restarts and never trigger exp_backoff_restart_delay, so pm2
# hot-loops the broken process forever, restarting it as fast as it can die, and
# the only symptom is a log file that fills up. Raising the bar to 30s is what
# reclassifies those restarts as unstable, which is the state pm2's backoff and
# restart limit actually act on. 30s also has to be long enough to cover a cold
# first render, or a healthy slow start would be treated as a crash.
#
# These are exported rather than passed as flags because pm2's CLI has no
# `--min-uptime` in any released version; see the comment in run_services.
API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"
API_MAX_RESTARTS="${API_MAX_RESTARTS:-10}"
FRONTEND_MAX_MEMORY="${FRONTEND_MAX_MEMORY:-1G}"
RESTART_BACKOFF_MS="${RESTART_BACKOFF_MS:-100}"
MIN_UPTIME_MS="${MIN_UPTIME_MS:-30000}"


# Pinned docker images by digest. IMPORTANT: the Qdrant version must be >= the
# version that wrote an existing collection (older versions cannot deserialize
# newer storage formats). Current default matches the deployment that created
# the live collection.
#
# Both images are pinned by DIGEST, not merely tagged, and that is a
# supply-chain control rather than a reproducibility nicety. A tag is a mutable
# name: whoever controls the registry account can re-point the redis tag at a
# different image tomorrow, and the next unattended `./setup.sh backend` pulls
# it and runs it. The digest names one immutable OCI image index, so a
# re-pushed tag no longer decides what runs. The redis digest is a multi-arch
# index, so the same pin still resolves on an arm64 host as on amd64.
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
    # nodejs.org publishes SHASUMS256.txt alongside every release, listing the
    # sha256 of each artifact in it. Downloading the tarball and extracting it
    # without checking that file means a truncated download, a CDN serving the
    # wrong bytes, or a tampered mirror all get installed into the Node that
    # builds and runs the frontend. The expected digest is read from the
    # published manifest rather than hardcoded here, so the check stays correct
    # across the auto-discovered version above.
    #
    # Both the tarball and the manifest come from the same origin, so this
    # verifies INTEGRITY -- the download arrived intact and is the artifact
    # that was published -- not AUTHENTICITY. Closing that gap needs a
    # signature or a hardcoded digest, which would freeze the version and defeat
    # the deliberate LTS auto-discovery, so it is called out rather than faked.
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
        # Pinned, because pm2 is what starts, restarts and supervises both
        # services. An unpinned install pulls whatever is newest when the stage
        # runs, so an unattended bootstrap can land a new major on a live box
        # and change restart/backoff behaviour with no code change to point at.
        # --no-audit/--no-fund only quieten the output.
        npm install -g --no-audit --no-fund "pm2@$PM2_VERSION" >/tmp/pm2-install.log 2>&1 || {
            echo "pm2 install failed:" >&2; tail -3 /tmp/pm2-install.log >&2; return 1
        }
        local nbin
        nbin="$(npm prefix -g)/bin"
        mkdir -p ~/.local/bin
        ln -sf "$nbin/pm2" ~/.local/bin/pm2
        ln -sf "$nbin/pm2-dev" ~/.local/bin/pm2-dev
    fi
    # The install above is pinned, but `have pm2` short-circuits when pm2 is
    # already on PATH, so on an upgraded host the pin alone does not make the
    # running version match. Say so rather than letting the pin read as a
    # guarantee it is not: this stage installs, it does not downgrade, because
    # moving a live process manager under a running backend is an operator
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
# A random secret, generated from the OS CSPRNG. This is not a token anyone
# sends over the network, so hex is fine and is the one encoding that survives
# being pasted into a URL, a docker argv and a shell without quoting rules.
random_secret() {
    if have openssl; then
        openssl rand -hex 32
        return
    fi
    "$VENV_PY" -c 'import secrets; print(secrets.token_hex(32))'
}

# The data stores' credentials, resolved ONCE and then reused.
#
# Both are read from backend/.env first and only generated when absent, and
# that ordering is the whole point. Regenerating either value on a re-run would
# change the password the running container was started with, so the very next
# request from the application would be rejected and the site would look broken
# for a reason that is invisible from the outside. So the value that is already
# in .env is the value that is used, and the container is only recreated when
# what it is actually running disagrees with it.
#
# The container, not .env, is the thing that can be stale: a box that was
# provisioned before this existed has an unauthenticated redis and qdrant
# running, and their .env has no credential for them. Storing a generated
# secret there is what lets the next step detect the gap and fix it.
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

# Bound docker json-file logs so they can't fill the disk (20MB x 3 files each).
DOCKER_LOG_OPTS="--log-driver json-file --log-opt max-size=20m --log-opt max-file=3"

# Does the container's OWN configuration carry this exact string?
#
# Read from `docker inspect` (what the container was created with), not from
# anything this script believes. It exists to answer one question on an upgraded
# host: is the running container the one this .env describes? A box provisioned
# before credentials existed has an unauthenticated redis/qdrant running, and
# .env now carries a password for it. Without this check the container looks
# healthy, is left alone, and the application then fails every request with
# NOAUTH because it is now correctly sending a password the server never asked
# for. The reverse case matters too: a password edited in .env must not be
# silently ignored either, and the same check catches that.
#
# Cmd and Env are both searched because the two containers express auth
# differently: redis takes a command flag, qdrant an environment variable.
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

# ensure_docker_container <name> <image> <ports> <volume> <envargs> <cmd> <auth-needle>
#
# envargs/cmd are the credential plumbing (empty for a container that needs
# none) and auth-needle is the string whose presence in the existing container
# proves its auth settings already match this run -- empty disables the check
# for a container that has no credentials to drift on.
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

# One KEY from a KEY=VALUE .env file, with the optional surrounding quotes an
# operator may have used and a trailing CR stripped. Prints nothing when the
# key is absent, which is the caller's signal to fall back to the application's
# own default. Last match wins, matching how python-dotenv resolves a repeated
# key.
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

# Secrets and user data at rest, tightened. backend/.env carries the JWT secret
# and the database credentials, and the two SQLite files carry every account and
# every chat message the site holds; both are world-readable by default, so any
# local account -- or any process running as another user on the same box --
# can read them. The backups directory gets 700 for the same reason: it holds
# copies of exactly those two files.
#
# The database paths are read out of .env rather than hard-coded because the
# application honours the same overrides (config.py reads CHAT_DB_PATH and
# AUTH_DB_PATH through load_dotenv) and resolves a relative one against the
# backend working directory. Hard-coding data/chat.db here would silently
# skip a host that moved its databases, which is the one host whose databases
# most need tightening.
#
# Called from BOTH ends of the lifecycle on purpose. run_backend is where .env
# is created or migrated, so that is the only place the file is guaranteed to
# exist afterwards; run_services is where a database that was already on disk
# before this change existed gets tightened on an upgraded host, since ./setup.sh
# services never re-runs the backend stage. One call would leave one of the two
# upgrade paths unprotected.
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
        # A database that does not exist is not an error. A fresh install has
        # not served a request, and the first request is what creates the file;
        # failing here would make the bootstrap fail on the host that most needs
        # it to succeed. The next stage, or the next boot, catches it.
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

# Repair the AUTH_TRUST_X_FORWARDED_FOR line in a backend .env: append the
# shipped default when the key is absent, warn when the operator has forced the
# header to be trusted from any peer.
#
# A forced True is a legitimate setting -- it is what a host whose reverse proxy
# runs on another host needs -- so nothing here ever rewrites a value. The
# warning is the point of the branch: with the header trusted from ANY peer, a
# client that reaches the API port directly can forge its rate-limit bucket
# (#245), and this is the only place an operator is ever told.
#
# Both greps match the line SHAPE python-dotenv accepts, not "KEY=value" and
# nothing else. The application reads .env through load_dotenv (app/config.py),
# which strips an `export` prefix, blanks around the `=` and the quotes around
# a value before the value ever reaches config._env_tristate -- so
# `AUTH_TRUST_X_FORWARDED_FOR = "true"` is a forced True in the app while a
# guard anchored at `^AUTH_TRUST_X_FORWARDED_FOR=` sees no key at all. That is
# not merely a missing warning. The presence check missed the same shape, so
# the script appended a SECOND `AUTH_TRUST_X_FORWARDED_FOR=auto`; python-dotenv
# resolves a repeated key to the last one, so the operator's forced True was
# silently downgraded on every run, into the branch that was supposed to warn
# about it, which never ran.
#
# The spellings in the three value patterns are config._TRUE_SPELLINGS and they
# have to be: that set is the definition of a forced True, and
# backend/tests/test_setup_script.py fails if the two disagree in EITHER
# direction -- a value that forces trust unmentioned, and a posture the
# operator is not actually in. This is a copy in shell rather than an import
# because it runs before the venv is guaranteed to exist, so the test is what
# holds the copy to the set. Scope: ASCII whitespace only. A POSIX bracket
# expression is ASCII in GNU grep under the C locale and under a UTF-8 one
# alike (measured on this host), so a value padded with U+00A0 is still read as
# a forced True by config and is still not matched here.
migrate_xff_trust() {
    local env_file="$1"
    # KEY= in every shape load_dotenv accepts: leading blanks, an optional
    # `export`, blanks around the `=`.
    local assignment='^[[:space:]]*(export[[:space:]]+)?AUTH_TRUST_X_FORWARDED_FOR[[:space:]]*='
    # The value in the three forms that mean a forced True: bare, double quoted,
    # single quoted. The two quotes must match -- `"true'` is a dotenv parse
    # error rather than a forced True, and a guard that matched it would be
    # warning about a posture the operator is not in. dotenv also allows a
    # trailing comment, after at least one blank on a bare value and after
    # none inside quotes, so the two tails differ.
    # `.` and not `[^\r\n]`: a backslash inside a POSIX bracket expression is a
    # literal, so that class excluded `r` and `n` as well and every comment
    # containing a word stopped matching.
    local bare="$assignment[[:space:]]*(1|true|yes|on)([[:space:]]+#.*)?[[:space:]]*$"
    local double_quoted="$assignment[[:space:]]*\"[[:space:]]*(1|true|yes|on)[[:space:]]*\"([[:space:]]*#.*)?[[:space:]]*$"
    local single_quoted="$assignment[[:space:]]*'[[:space:]]*(1|true|yes|on)[[:space:]]*'([[:space:]]*#.*)?[[:space:]]*$"

    if ! grep -qE "$assignment" "$env_file"; then
        echo "AUTH_TRUST_X_FORWARDED_FOR=auto" >> "$env_file"
    elif grep -qiE -e "$bare" -e "$double_quoted" -e "$single_quoted" "$env_file"; then
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
}

run_backend() {
    stage "backend"
    check_python python3
    docker_up

    # .env first, then the venv, then the secrets, then the containers.
    #
    # The order is load-bearing and each step depends on the one before it.
    # The containers need the credentials to be started AT ALL, so they cannot
    # come first. The credentials go in .env and are generated with the
    # interpreter (openssl is not guaranteed on a minimal host), so the venv has
    # to exist before they are generated. And .env has to exist before either,
    # since store_secrets reads what is already there and only generates when
    # the key is absent -- that read is what keeps a re-run from rotating the
    # password out from under a running container.
    if [ ! -f "$ENV_FILE" ]; then
        echo "creating backend/.env from example (fill in credentials!)"
        cp backend/.env.example "$ENV_FILE"
    fi
    if [ ! -x "$VENV_PY" ]; then
        echo "creating venv..."
        python3 -m venv "$VENV"
    fi
    store_secrets
    # The secrets are in the file now, so it stops being world-readable
    # immediately rather than at the end of the stage.
    chmod 600 "$ENV_FILE"

    # qdrant takes its API key as an environment variable; redis takes its
    # password as a command flag, which is why these two differ in shape. The
    # needle passed to each is the literal the container must already be
    # carrying, so a container started before credentials existed is detected
    # and recreated instead of being left in a state where the application
    # sends a password the server never asked for.
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
    # redis client in the app (cache, rate limiter, analytics, cost budget,
    # health, profiles) is built from config.REDIS_URL and none of them take a
    # separate password argument. Putting it in the URL is the single place that
    # makes all six authenticate, and redis-py parses the userinfo segment
    # natively. A URL that already carries credentials is left alone, so an
    # operator's hand-written password is never overwritten by the generated
    # one.
    if ! grep -qE '^REDIS_URL=redis://[^/@]*@' "$ENV_FILE"; then
        if grep -q '^REDIS_URL=' "$ENV_FILE"; then
            # Rewrite the existing line in place: it has no credentials, so it
            # cannot be one the operator chose deliberately. sed -i keeps the
            # line where it is instead of appending a second REDIS_URL that
            # python-dotenv would resolve to whichever it read last.
            local escaped="$REDIS_PASSWORD"
            escaped="${escaped//\//\\/}"
            escaped="${escaped//&/\\&}"
            sed -i -E "s#^REDIS_URL=(.*)\$#REDIS_URL=redis://:${escaped}@\\1#" "$ENV_FILE"
            echo "REDIS_URL now carries the redis password"
        else
            echo "REDIS_URL=redis://:$REDIS_PASSWORD@localhost:$REDIS_PORT/0" >> "$ENV_FILE"
        fi
    fi

    # Per-IP rate limiting keys on the client IP, which behind nginx comes from
    # X-Forwarded-For. An .env that predates the per-IP public rate limits has
    # no trust setting at all, so every proxied request keys on the nginx peer
    # (127.0.0.1) and the whole site shares one rate-limit bucket.
    # migrate_xff_trust appends the shipped default in that one case and warns
    # when the operator has forced the header to be trusted; the reasoning, and
    # the line shapes it has to recognise, live with the function.
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

# Name the reason the readiness gate rejected the deploy, so a 30-second
# timeout does not end in a bare "not ready".
#
# No -f here, deliberately. This only ever runs when the probe answered >= 400
# (or the connection failed), and `curl -f` suppresses the body on exactly those
# responses -- which made this function dead code and left the operator with the
# bare "not ready after 30s" it exists to prevent. The report body, including
# checks.llm.reason, is the app's own answer and is what gets printed.
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

    # Start from ecosystem.config.js, which is the single definition of these two
    # processes, rather than from inline `pm2 start` lines.
    #
    # The reason is min_uptime, and it is not a preference. pm2's CLI has no
    # `--min-uptime` flag in ANY released version -- 4.x, 5.x, 6.x and 7.x all
    # print "error: unknown option `--min-uptime'" and exit 1 -- while
    # `min_uptime` in an ecosystem file IS honoured at runtime (pm2 reads it in
    # lib/God.js when deciding whether a restart was stable). An inline start
    # cannot express the option at all, and because this script runs under
    # `set -e`, passing the flag would abort the services stage and leave the box
    # with neither service running. Verified against 4.5.0, 5.4.3, 6.0.14 and
    # 7.0.4 for the rejection, and against 7.0.4 for the ecosystem route storing
    # the value.
    #
    # Every knob setup.sh has always accepted is exported rather than left to
    # chance, so the overrides keep working on the path that actually starts the
    # services. ecosystem.config.js falls back to the same defaults when a
    # variable is absent, so a bare `pm2 start ecosystem.config.js` produces the
    # same two processes.
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

    # Readiness, not liveness (#279): /health is a stub that answers 200 with a
    # dead Qdrant client, unloaded models or the placeholder GEMINI_API_KEY, so
    # gating the deploy on it declared broken backends deployed. /ready/deep is
    # the uncached, unrated, loopback-only form, so this gate cannot pass on a
    # warm readiness cache and cannot be throttled into a false "not ready".
    #
    # Placed LAST on purpose. This gate can legitimately fail, and `set -e`
    # aborts on it, so running it before the frontend was started turned a
    # misconfigured key into a torn-down deployment with the frontend never
    # coming back. Here both services are up and the pm2 dump is saved, so the
    # operator is told the deploy is not ready with everything still running,
    # and can fix the key and re-run.
    #
    # There is deliberately NO shell-side check of GEMINI_API_KEY before the
    # teardown. app.config.classify_gemini_api_key is the only classifier, and
    # a shell copy of its sentinel list is a second one that silently drifts:
    # a shorter copy misses placeholders, a longer one refuses deploys the app
    # would accept. The verdict here is the app's own -- report_readiness_reason
    # prints checks.llm.reason from the response -- so it cannot disagree with
    # what /ready will actually say.
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
    # update_index.py takes its own flock(2) on data/update.lock (LOCK_EX|LOCK_NB)
    # and skips when another run holds it, so no external flock wrapper is needed
    # (wrapping with `flock -n` would conflict with the script's own lock and
    # cause every run to be skipped).
    local line_idx="*/15 * * * * nice -n 15 $VENV_PY $SCRIPT_DIR/backend/scripts/update_index.py >> $log 2>&1"
    # cron does not inherit the operator's shell environment, so pass the
    # webhook URL (and a minimal PATH via healthcheck.sh) explicitly. Empty
    # webhook is harmless: healthcheck.sh treats an unset/empty value as "no
    # webhook". Keep the entry stable for idempotent re-runs.
    # BASE is passed explicitly because the API is bound to 127.0.0.1:$API_PORT
    # and the watchdog defaults to 8001: an operator who overrides API_PORT
    # would otherwise have the watchdog probe a closed port and restart a
    # perfectly healthy backend every five minutes.
    local line_hc="*/5 * * * * BASE=\"http://localhost:$API_PORT\" HEALTHCHECK_WEBHOOK_URL=\"${HEALTHCHECK_WEBHOOK_URL:-}\" LOG=$hc_log $SCRIPT_DIR/deploy/healthcheck.sh"
    local tmp
    tmp="$(mktemp)"
    # The healthcheck line is removed by SCRIPT PATH, not by exact match.
    # `grep -vFx` matches whole lines, and this line's text has already changed
    # once (BASE= was added), so an entry written by a previous revision stopped
    # matching the literal it had to be deleted by: it survived every run, and
    # on a host with a non-default API_PORT that stale copy carried no BASE=,
    # fell back to the watchdog's :8001 default, got a refused connection and
    # was read as "not alive" -- pm2 restart against a healthy backend every
    # five minutes.
    #
    # Two reasons to filter on the path rather than on the line text:
    #   1. The P1 comes straight back the moment a future revision edits the
    #      schedule, because the stale line stops matching on that too. Anchoring
    #      on schedule AND path would work today and silently break then.
    #   2. `./setup.sh cron` is an explicit operator action, and reclaiming lines
    #      that run a script this script manages is the contract. A stale entry
    #      is not something an operator asked for.
    #
    # ACCEPTED COST, named so it is a decision and not an accident: a user's own
    # crontab line that runs healthcheck.sh on a CUSTOM schedule is replaced by
    # the managed one. Put the webhook or LOG overrides on the managed entry
    # instead. Two things are NOT touched: a line that does not run these two
    # scripts, and a COMMENTED line that merely mentions healthcheck.sh -- a
    # commented-out entry is how an operator disables the watchdog, and deleting
    # it would silently re-arm one they believe is off.
    #
    # A MANAGED_BY=<tag> env prefix would let us keep such a line, but it cannot
    # be the filter: an entry written before the tag existed carries no tag, so
    # filtering on the tag alone would leave precisely the stale entry this P1 is
    # about.
    #
    # The indexer line is still matched exactly -- this branch never changed its
    # text, and an exact match keeps a user's hand-edited variant (a different
    # schedule, a `nice` tweak) from being deleted out from under them. That
    # asymmetry is the cost of not wanting the same P1 there.
    #
    # The healthcheck filter matches on the path but skips COMMENTED lines, so a
    # user who disabled the watchdog by commenting its entry out keeps that
    # marker. A plain substring `grep -vF` deleted it, which would silently
    # re-arm a watchdog the operator believes they had switched off.
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

# The logrotate policy is machine-specific: it names this host's log
# directories and the account logrotate has to drop privileges to, and neither is
# knowable when the file is written. deploy/logrotate.conf is therefore a
# TEMPLATE carrying the placeholder paths, and this renders it for the host it
# runs on, which is what makes the installed policy correct on any host instead
# of only on the one the paths were written for.
#
# The three rewrites are anchored to the whole line, and that is the safety
# argument rather than a style preference. A free-floating substitution would
# happily "rewrite" a line that had been edited into something else and leave a
# half-correct policy installed, which is worse than no rotation at all because
# the operator has no reason to look again. So the patterns match the exact
# lines, and the guard inside the renderer asserts that the TEMPLATE still
# contains each of the three values those patterns match on: if one is gone,
# nothing was rewritten, and refusing to emit is the only safe answer.
#
# The third rewrite also UNCOMMENTS the `su` directive. It ships commented out
# (backend/tests/test_deploy_paths.py fails a shipped deploy file that carries an
# active `su` pinning one account, because that file is copied by hand as well as
# rendered), but an installed policy that names no account runs every rotation as
# root, so the renderer activates it with the account that invoked setup.sh.
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
    # The guard is on the TEMPLATE, never on the output. The reason is the same
    # one the rewrites are anchored to: what can actually go wrong is the
    # template drifting away from the three values the rewrites above match on.
    # If a value is gone, no line was rewritten, and a half-substituted policy
    # would be installed that rotates nothing while logrotate reports success.
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
    # Both exit paths remove the temp file: the render can fail, and a failed
    # render must not leave a copy of the policy lying in /tmp.
    trap 'rm -f "$tmp"' RETURN
    if ! render_logrotate_conf > "$tmp"; then
        echo "ERROR: could not render the logrotate policy; nothing installed." >&2
        return 1
    fi
    # The source operand is the rendered file. install takes SOURCE DEST, so
    # this is what actually creates $LOGROTATE_CONF.
    sudo install -m 644 "$tmp" "$LOGROTATE_CONF"
    rm -f "$tmp"
    trap - RETURN
    echo "logrotate installed: $LOGROTATE_CONF (mode 644, root)"
    echo "inspect it with: sudo logrotate -d $LOGROTATE_CONF"
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
        # limit_req_zone is http-scope: it is declared once, outside every
        # server block, and this file is included from inside nginx's http block
        # (sites-enabled/*), so top level HERE is http scope. Declaring it per
        # server would give the plain-HTTP and TLS servers two independent
        # buckets, so a client could double its rate by switching schemes.
        #
        # It covers the chat stream only, for the reason spelled out on the
        # location below: every other rate-limited route is already limited
        # inside the application, and a second limiter there would shadow those
        # knobs.
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
            deps|backend|index|frontend|services|pm2-startup|stop-backend|stop-frontend|stop|cron|nginx|tls|logrotate) STAGES+=("$1") ;;
            *) echo "unknown stage: $1"; usage; return 1 ;;
        esac
        shift
    done

    if [ "$ALL" -eq 1 ]; then
        # tls is deliberately not part of "all": it needs a domain, an email and
        # network access that an unattended bootstrap must not require.
        # logrotate is deliberately not part of "all" either: it needs the
        # logrotate binary, and a host that does not have it installed (a
        # container, a fresh CI box) must still be able to run the full
        # bootstrap. Unattended rotation is also the wrong default -- the
        # policy is host-specific, so it is an operator step, run once.
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
