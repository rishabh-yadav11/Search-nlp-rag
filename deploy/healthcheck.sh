#!/usr/bin/env bash
# Health-check the backend AND the frontend, restart whichever is unhealthy, and
# report any service a restart did not bring back. Run from cron every few
# minutes.
#
# TWO SERVICES, probed differently, because "is it up" means two different
# things on each side of the API.
#
# Backend, two probes, because a present process and a usable service are
# different questions:
#   /ready/deep -- readiness. Qdrant reachable, models loaded, a usable
#                  GEMINI_API_KEY. Deliberately uncached, unrated and
#                  loopback-only: a watchdog acts on the answer it gets, and a
#                  cached verdict hides an outage for the rest of the TTL while
#                  a 429 is indistinguishable from an outage.
#   /health     -- liveness. It cannot fail by design, so it is the RIGHT probe
#                  for the one question a restart can answer and the WRONG one
#                  for "is the service healthy": it answers 200 with a dead
#                  Qdrant client and a placeholder key. Probing /health alone
#                  made a dependency outage invisible.
#
# Frontend, one lenient probe of `/`. A Next.js server answering `/` may
# legitimately answer 3xx, 404 or even 5xx, and every one of those proves the
# listener is up. Unhealthy means no HTTP status came back at all: curl exits
# non-zero and writes nothing or 000. `curl -f` would conflate those and
# restart a healthy frontend on every 404.
#
# Behaviour:
#   1. Probe both services.
#   2. Each unhealthy service is restarted via pm2; then, after a single wait,
#      every restarted service is re-probed and any that answers is logged as
#      recovered (exit 0 -- a restart that worked is not an incident).
#   3. A backend that is alive but not ready is NOT restarted: the process is
#      fine, and a restart cannot bring Qdrant back or fix a placeholder key.
#   4. Anything still down, plus every dependency verdict, -> one log line
#      naming every service still down, a single POST to
#      HEALTHCHECK_WEBHOOK_URL (if set), and a print. Exit 1.
#
# An alert is emitted once per fault, not once per run: an unfixed fault is
# logged as "still failing" and otherwise stays silent until
# ALERT_COOLDOWN_SECONDS has passed, because the cron cadence would otherwise
# mail and POST the same news hundreds of times a day. The exit code still
# reports the fault every run. The cooldown is keyed on the whole set of
# faults, so two faults in one run are one alert, however many services are in
# it.
#
# Override (env): BASE, FRONTEND_BASE, LOG, HEALTHCHECK_WEBHOOK_URL,
#                 RESTART_WAIT_SECONDS, STATE_FILE, ALERT_COOLDOWN_SECONDS.
set -u

BASE="${BASE:-http://localhost:8001}"
FRONTEND_BASE="${FRONTEND_BASE:-http://localhost:3000}"
APP="vccircle-backend"
FRONTEND_APP="vccircle-frontend"
# Derived from this script's own location so the watchdog log and its state file
# land in the app's real logs/ directory wherever the repo is checked out.
# Resolved, never absolute and never CWD-dependent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$SCRIPT_DIR/../logs/healthcheck.log}"
RESTART_WAIT_SECONDS="${RESTART_WAIT_SECONDS:-8}"
# A fault that stays unfixed must not re-alert every run: the cron cadence
# would mail and POST the same news hundreds of times a day. The state file
# holds "<key> <epoch>" for the fault being reported; an identical fault inside
# the cooldown is logged once and otherwise silent. A healthy run clears it.
STATE_FILE="${STATE_FILE:-$LOG.state}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-3600}"

# The three probe URLs, spelled out once so the probe, the alert and the comment
# above can never disagree. A trailing slash on FRONTEND_BASE is trimmed so the
# root probe cannot ask for "//".
BACKEND_LIVE_URL="$BASE/health"
BACKEND_READY_URL="$BASE/ready/deep"
FRONTEND_URL="${FRONTEND_BASE%/}/"

# cron has a minimal PATH, so pm2 may not be found. The usual locations are
# APPENDED, not prepended: prepending silently overrides a PATH the invoker
# set up (a harness shimming `curl` to target a staging host would be ignored),
# while appending still gives cron a pm2 and leaves explicit settings in charge.
export PATH="$PATH:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

# One sentence per service still down and one fault key per sentence, joined
# with "; ", so two failing services produce ONE message naming both instead of
# only whichever was checked last.
failures=""
fault_key=""

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

# probe <url> <mode> -- write the HTTP status to stdout and exit with curl's own
# status, so the caller can tell "the request never completed" from "it completed
# and returned a status I may or may not like". `strict` is the backend: -f makes
# curl fail on any non-2xx, so a 500 and a refused connection both arrive
# non-zero. `lenient` is the frontend: no -f, so the only question is whether
# anything came back at all.
probe() {
  if [ "$2" = "strict" ]; then
    curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  else
    curl -sS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  fi
}

# is_healthy <curl-status> <http-status> <mode> -- did the probe prove the
# service is up? Both conditions are needed: curl exited 0 (an exchange
# completed), and the status is a real three-digit code rather than 000 or
# nothing, which is what a killed or timed-out curl leaves. Then the mode
# decides: `strict` also requires 200; `lenient` accepts anything that came
# back over the wire, including 404 and 500.
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
# readiness_verdict <status> -- the sentence for a backend that is up but whose
# readiness probe is not 200. WHICH failure it is decides what the operator is
# told: a 503 is a real verdict about the dependencies, while a refused,
# throttled or broken PROBE says nothing about them and must not be dressed up
# as one.
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


# NOTE: this replays the argv pm2 stored at start time -- `--update-env`
# refreshes environment variables, not the argument list. Correct for
# recovering a sick process, but it can NEVER apply a change to the process
# options. Changing the API bind, the frontend port or any other pm2 option
# requires `./setup.sh services`.
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
  # Best effort: the alert is already logged and printed, so a webhook that is
  # down must not swallow it or change what this run reports.
  log "WARNING: webhook delivery failed (best effort; the run's verdict stands)"
}

# $1 is a stable name for the fault, $2 the sentence. Re-alerts only when the
# set of faults changes or the cooldown has passed.
alert() {
  local key="$1" msg="$2" now last_key last_at age
  now=$(date -u +%s)
  last_key=""; last_at=0
  if [ -f "$STATE_FILE" ]; then
    read -r last_key last_at <"$STATE_FILE" 2>/dev/null || true
    # The state file is a cache, never a gate: a corrupt or hand-edited one
    # must not be able to silence an alert. Anything that is not a plain
    # non-negative integer reads as "never alerted" -- an unquoted expansion of
    # a non-numeric value aborts the whole run under `set -u`.
    case "$last_at" in
      '' | *[!0-9]*) last_at=0 ;;
    esac
  fi
  age=$((now - last_at))
  # A stamp in the future means the clock moved backwards. It is not a
  # cooldown, so it must not suppress anything; a negative age is expired.
  if [ "$last_key" = "$key" ] && [ "$age" -ge 0 ] && [ "$age" -lt "$ALERT_COOLDOWN_SECONDS" ]; then
    log "still failing ($key); alert suppressed for another $((ALERT_COOLDOWN_SECONDS - age))s"
    return 0
  fi
  # Written atomically (tmp + mv on one filesystem) because `>` truncates
  # first: a crash between the two left an empty file, which the next run read
  # as "never alerted" -- safe, but it silently re-enabled every alert. A write
  # that fails still warns the operator rather than disabling deduplication
  # silently.
  if printf '%s %s\n' "$key" "$now" >"$STATE_FILE.tmp" 2>/dev/null; then
    mv -f "$STATE_FILE.tmp" "$STATE_FILE" 2>/dev/null || log "WARNING: could not update $STATE_FILE"
  else
    log "WARNING: could not write $STATE_FILE; this alert will not be deduplicated"
  fi
  log "$msg"
  post_webhook "$msg" || true
  echo "$msg"
}

# add_failure <fault-key> <sentence> -- the sentence carries no "ALERT:" prefix
# of its own, so a single fault is byte-for-byte the alert it has always been and
# a second fault is appended rather than restarting the sentence.
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

# --- the backend: liveness first, and on its own. If the process is not up
# there is no point asking it about its dependencies.
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

# --- the frontend: one lenient probe, and a restart only when nothing answered.
frontend_rc=0
frontend_code=$(probe "$FRONTEND_URL" lenient) || frontend_rc=$?
if ! is_healthy "$frontend_rc" "$frontend_code" lenient; then
  log "frontend unhealthy (HTTP ${frontend_code:-none}); restarting $FRONTEND_APP"
  restart "$FRONTEND_APP"
  restarted_frontend=1
fi

# One wait for however many services were restarted: both processes are already
# restarting, so waiting twice would only delay the alert.
if [ "$restarted_backend" -eq 1 ] || [ "$restarted_frontend" -eq 1 ]; then
  sleep "$RESTART_WAIT_SECONDS"
fi

# --- verdict: re-probe whatever was restarted, then decide per service.
if [ "$restarted_backend" -eq 1 ]; then
  live_rc=0
  live_code=$(probe "$BACKEND_LIVE_URL" strict) || live_rc=$?
  ready_rc=0
  ready_code=$(probe "$BACKEND_READY_URL" strict) || ready_rc=$?
  if is_healthy "$live_rc" "$live_code" strict && is_healthy "$ready_rc" "$ready_code" strict; then
    log "backend recovered after restart"
  elif is_healthy "$live_rc" "$live_code" strict; then
    # Came back alive but still not ready. Same taxonomy as below.
    add_failure "live-not-ready:$ready_code" "$(readiness_verdict "$ready_code")"
  else
    add_failure "down:$live_code" "VCCircle backend down: liveness /health returned HTTP ${live_code:-none} after restart, readiness /ready/deep returned HTTP ${ready_code:-none} ($BASE)"
  fi
elif is_healthy "$live_rc" "$live_code" strict; then
  if ! is_healthy "$ready_rc" "$ready_code" strict; then
    # Alive, not ready. WHICH of those it is decides what the operator is told:
    # a 503 is a real verdict about the dependencies, while a refused,
    # throttled or broken PROBE says nothing about them.
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

# --- the one alert for whatever is still wrong.
if [ -n "$failures" ]; then
  # One alert for the whole set: one mail and one webhook POST naming every
  # service, and the cooldown is keyed on that set, so two faults deduplicate as
  # the single fault reported.
  alert "$fault_key" "ALERT: $failures"
  exit 1
fi

# Clear the fault state, so if something breaks again the next run alerts
# immediately instead of inheriting an unexpired cooldown.
rm -f "$STATE_FILE"
exit 0
