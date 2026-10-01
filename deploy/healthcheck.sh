#!/usr/bin/env bash
set -u

BASE="${BASE:-http://localhost:8001}"
FRONTEND_BASE="${FRONTEND_BASE:-http://localhost:3000}"
APP="vccircle-backend"
FRONTEND_APP="vccircle-frontend"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$SCRIPT_DIR/../logs/healthcheck.log}"
RESTART_WAIT_SECONDS="${RESTART_WAIT_SECONDS:-8}"
STATE_FILE="${STATE_FILE:-$LOG.state}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-3600}"

BACKEND_LIVE_URL="$BASE/health"
BACKEND_READY_URL="$BASE/ready/deep"
FRONTEND_URL="${FRONTEND_BASE%/}/"

export PATH="$PATH:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

failures=""
fault_key=""

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

# Two stages: the HTTP status goes to stdout while the exit status carries curl's own, because a bare `curl -f` conflates a 404 with a refused connection and a watchdog blind to that difference exits 0 forever.
probe() {
  if [ "$2" = "strict" ]; then
    curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  else
    curl -sS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  fi
}

is_healthy() {
  case "$1" in
    0) ;;
    *) return 1 ;;
  esac
  case "$2" in
    '' | 000) return 1 ;;
  esac
  if [ "$3" = "strict" ]; then
    [ "$2" = "200" ]
  else
    return 0
  fi
}
readiness_verdict() {
  case "$1" in
    503)
      echo "VCCircle backend alive but not ready: /ready/deep returned HTTP 503 while the process is up ($BASE). Not restarting: check Qdrant, the loaded models, and GEMINI_API_KEY."
      ;;
    403 | 404)
      echo "VCCircle readiness probe REFUSED (HTTP $1) at $BASE/ready/deep -- this is not a verdict about the backend. It answers only a direct loopback caller sending no X-Forwarded-For, so BASE points somewhere it cannot be reached (a reverse proxy, another host, or a port the API is not bound to)."
      ;;
    429)
      echo "VCCircle readiness probe was rate limited (HTTP 429) at $BASE/ready/deep -- the watchdog is sharing a budget it should not share. This run says nothing about the backend's health."
      ;;
    500)
      echo "VCCircle readiness probe failed internally (HTTP 500) at $BASE/ready/deep -- the probe itself raised, so no dependency verdict was reached. See the backend log."
      ;;
    *)
      echo "VCCircle backend alive but readiness unreachable: /ready/deep returned HTTP ${1:-none} at $BASE. Not restarting: the process answered liveness."
      ;;
  esac
}


# `--update-env` refreshes the environment, not the options: pm2 replays its stored argv, so an option change needs `./setup.sh services`.
restart() {
  pm2 restart "$1" --update-env >/dev/null 2>&1 || true
}

post_webhook() {
  local msg="$1"
  [ -z "$WEBHOOK" ] && return 0
  if curl -fsS -m 10 -X POST "$WEBHOOK" \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"$msg\"}" >/dev/null 2>&1; then
    return 0
  fi
  log "WARNING: webhook delivery failed (best effort; the run's verdict stands)"
}

alert() {
  local key="$1" msg="$2" now last_key last_at age
  now=$(date -u +%s)
  last_key=""; last_at=0
  if [ -f "$STATE_FILE" ]; then
    read -r last_key last_at <"$STATE_FILE" 2>/dev/null || true
    case "$last_at" in
      '' | *[!0-9]*) last_at=0 ;;
    esac
  fi
  age=$((now - last_at))
  if [ "$last_key" = "$key" ] && [ "$age" -ge 0 ] && [ "$age" -lt "$ALERT_COOLDOWN_SECONDS" ]; then
    log "still failing ($key); alert suppressed for another $((ALERT_COOLDOWN_SECONDS - age))s"
    return 0
  fi
  if printf '%s %s\n' "$key" "$now" >"$STATE_FILE.tmp" 2>/dev/null; then
    mv -f "$STATE_FILE.tmp" "$STATE_FILE" 2>/dev/null || log "WARNING: could not update $STATE_FILE"
  else
    log "WARNING: could not write $STATE_FILE; this alert will not be deduplicated"
  fi
  log "$msg"
  post_webhook "$msg" || true
  echo "$msg"
}

add_failure() {
  if [ -z "$failures" ]; then
    failures="$2"
    fault_key="$1"
  else
    failures="$failures; $2"
    fault_key="$fault_key+$1"
  fi
}

mkdir -p "$(dirname "$LOG")"

restarted_backend=0
restarted_frontend=0

live_rc=0
live_code=$(probe "$BACKEND_LIVE_URL" strict) || live_rc=$?
if is_healthy "$live_rc" "$live_code" strict; then
  ready_rc=0
  ready_code=$(probe "$BACKEND_READY_URL" strict) || ready_rc=$?
else
  ready_rc=0
  ready_code=""
  log "backend not alive (liveness /health returned HTTP ${live_code:-none}); restarting $APP"
  restart "$APP"
  restarted_backend=1
fi

# Lenient, unlike the backend's strict probe: a Next.js `/` may legitimately answer 3xx, 404 or even 5xx, and any status at all proves the listener is up.
frontend_rc=0
frontend_code=$(probe "$FRONTEND_URL" lenient) || frontend_rc=$?
if ! is_healthy "$frontend_rc" "$frontend_code" lenient; then
  log "frontend unhealthy (HTTP ${frontend_code:-none}); restarting $FRONTEND_APP"
  restart "$FRONTEND_APP"
  restarted_frontend=1
fi

if [ "$restarted_backend" -eq 1 ] || [ "$restarted_frontend" -eq 1 ]; then
  sleep "$RESTART_WAIT_SECONDS"
fi

if [ "$restarted_backend" -eq 1 ]; then
  live_rc=0
  live_code=$(probe "$BACKEND_LIVE_URL" strict) || live_rc=$?
  ready_rc=0
  ready_code=$(probe "$BACKEND_READY_URL" strict) || ready_rc=$?
  if is_healthy "$live_rc" "$live_code" strict && is_healthy "$ready_rc" "$ready_code" strict; then
    log "backend recovered after restart"
  elif is_healthy "$live_rc" "$live_code" strict; then
    add_failure "live-not-ready:$ready_code" "$(readiness_verdict "$ready_code")"
  else
    add_failure "down:$live_code" "VCCircle backend down: liveness /health returned HTTP ${live_code:-none} after restart, readiness /ready/deep returned HTTP ${ready_code:-none} ($BASE)"
  fi
elif is_healthy "$live_rc" "$live_code" strict; then
  if ! is_healthy "$ready_rc" "$ready_code" strict; then
    add_failure "live-not-ready:$ready_code" "$(readiness_verdict "$ready_code")"
  fi
fi

if [ "$restarted_frontend" -eq 1 ]; then
  frontend_rc=0
  frontend_code=$(probe "$FRONTEND_URL" lenient) || frontend_rc=$?
  if is_healthy "$frontend_rc" "$frontend_code" lenient; then
    log "frontend recovered after restart"
  else
    add_failure "frontend-down:$frontend_code" "VCCircle frontend down: / returned HTTP ${frontend_code:-none} after restart ($FRONTEND_BASE)"
  fi
fi

if [ -n "$failures" ]; then
  # The cooldown is keyed on the whole fault set, so two simultaneous failures stay one alert instead of two.
  alert "$fault_key" "ALERT: $failures"
  exit 1
fi

rm -f "$STATE_FILE"
exit 0
