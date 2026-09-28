#!/usr/bin/env bash
# Health-check the backend AND the frontend, restart whichever is unhealthy, and
# report any service a restart did not bring back. Run from cron every few
# minutes.
#
# TWO SERVICES, and they are not probed the same way, because "is it up" means
# two different things on each side of the API.
#
# Backend, two probes, because the process being present and the service being
# usable are different questions (#279):
#   /ready/deep -- readiness. Does this node actually serve? Qdrant reachable,
#                  models loaded, a usable GEMINI_API_KEY. Deliberately
#                  uncached and unrated (and loopback-only), because a watchdog
#                  acts on the answer it gets: a cached verdict hides an outage
#                  for the rest of the TTL, and a 429 is indistinguishable from
#                  an outage to a caller that restarts on any non-200.
#   /health     -- liveness. Is the process up at all? It cannot fail by
#                  design, so it is the RIGHT probe for the one question a
#                  restart can answer, and the WRONG probe for "is the service
#                  healthy": it answers 200 with a dead Qdrant client and a
#                  placeholder GEMINI_API_KEY. Probing /health alone is what made
#                  a dependency outage invisible -- the watchdog exited 0 forever.
#
# Frontend, one lenient probe of `/`. A Next.js server answering `/` may
# legitimately answer 3xx (a locale or auth redirect), 404 (no matching route)
# or even 5xx, and every one of those statuses proves the listener is up and
# serving, which is the only thing this probe is asked to establish. What makes
# the frontend unhealthy is that no HTTP status came back at all: curl exits
# non-zero and writes out either nothing or 000, which is connection refused,
# connection reset or a timeout. `curl -f` here would conflate those two cases
# and restart a perfectly healthy frontend on every 404. Before the frontend
# was covered, a wedged frontend was invisible: nginx kept proxying to port
# 3000, every page 502'd, and the backend -- the only service being checked --
# looked perfectly healthy, so this script reported success while the site was
# down.
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
#      HEALTHCHECK_WEBHOOK_URL (if set), and a print (cron mails on output if
#      MAILTO is set). Exit 1.
#
# An alert is emitted once per fault, not once per run: a fault that stays
# unfixed is logged as "still failing" and otherwise stays silent until
# ALERT_COOLDOWN_SECONDS has passed, because the */5 cron cadence would
# otherwise mail and POST the same news ~288 times a day. The exit code still
# reports the fault on every run. The cooldown is keyed on the whole set of
# faults, so a run where the backend is sick AND the frontend is down is one
# fault to the operator -- and it is still one fault, not two, however many
# services are in it.
#
# Override (env): BASE, FRONTEND_BASE, LOG, HEALTHCHECK_WEBHOOK_URL,
#                 RESTART_WAIT_SECONDS, STATE_FILE, ALERT_COOLDOWN_SECONDS.
set -u

BASE="${BASE:-http://localhost:8001}"
FRONTEND_BASE="${FRONTEND_BASE:-http://localhost:3000}"
APP="vccircle-backend"
FRONTEND_APP="vccircle-frontend"
# Derived from this script's own location so the watchdog log and its state file
# land in the app's real logs/ directory wherever the repo is checked out. This
# used to pin one hardcoded home-directory path under $HOME, which on any other
# checkout created a stray `~/search-nlp-rag/logs` tree disconnected from the
# logs logrotate actually rotates. Resolved, never absolute and never
# CWD-dependent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG:-$SCRIPT_DIR/../logs/healthcheck.log}"
RESTART_WAIT_SECONDS="${RESTART_WAIT_SECONDS:-8}"
# Cron runs this every few minutes, so a fault that stays fixed would re-alert
# forever: a persistent placeholder key is ~288 identical webhook POSTs and
# ~288 cron mails a day, which is how the one alert that matters gets ignored.
# The state file holds "<key> <epoch>" for the fault currently being reported;
# an identical fault inside the cooldown is logged once and otherwise silent.
# A healthy run clears it, so a later fault alerts immediately.
STATE_FILE="${STATE_FILE:-$LOG.state}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-3600}"

# The three probe URLs, spelled out once so the probe, the alert and the comment
# above can never disagree about what is being asked for. A trailing slash on
# FRONTEND_BASE is trimmed first so the root probe cannot end up asking for "//".
BACKEND_LIVE_URL="$BASE/health"
BACKEND_READY_URL="$BASE/ready/deep"
FRONTEND_URL="${FRONTEND_BASE%/}/"

# cron has a minimal PATH, so pm2 may not be found. The usual locations are
# APPENDED, not prepended. Prepending them is what this script used to do, and
# it meant that a `PATH` set up by whoever invoked the script lost every entry
# it had put in front: a test harness, or an operator, that shims `curl` to
# point this script at a staging host would silently be overridden by
# /usr/bin/curl and probe the real one instead. Appending still gives cron a pm2
# to find, and leaves anything explicitly provided in charge.
export PATH="$PATH:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

# One sentence per service that is still down, and one fault key per sentence,
# joined with "; " so that a single failing service produces exactly the alert it
# always has, and two failing services produce ONE message naming both of them
# instead of only reporting whichever happened to be checked last.
failures=""
fault_key=""

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

# probe <url> <mode> -- write the HTTP status to stdout and exit with curl's own
# status, so the caller can tell "the request never completed" apart from "the
# request completed and returned a status I may or may not like". `strict` is
# the backend: -f makes curl fail on any non-2xx, so a 500 and a refused
# connection both arrive as a non-zero exit. `lenient` is the frontend: no -f,
# so the status is whatever came back over the wire and the only question is
# whether anything came back at all.
probe() {
  if [ "$2" = "strict" ]; then
    curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  else
    curl -sS -m 10 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null
  fi
}

# is_healthy <curl-status> <http-status> <mode> -- did the probe prove the
# service is up? Two conditions, and both are needed: curl exited 0, meaning it
# completed an HTTP exchange, and the status it wrote out is a real three-digit
# code rather than 000 (or nothing at all, which is what a killed or timed-out
# curl leaves behind). Only then does the mode decide: `strict` additionally
# requires 200, `lenient` accepts anything that came back over the wire,
# including 404 and 500.
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
# readiness probe is not 200. WHICH of those it is decides what the operator is
# told: a 503 is a real verdict about the dependencies, while a refused,
# throttled or broken PROBE says nothing at all about them and must not be
# dressed up as one.
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
# refreshes environment variables, not the argument list. It is correct for
# recovering a sick process, but it can NEVER apply a change to the process
# options, for either app. Changing the API bind, the frontend port, or any
# other pm2 option requires `./setup.sh services`, which re-registers the
# processes from ecosystem.config.js and re-saves the dump.
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
  # Best effort, and deliberately not a decision: the alert has already been
  # logged and printed, so a webhook that is down must not swallow it or change
  # what this run reports.
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
    # The state file is a cache, never a gate: a corrupt, truncated or
    # hand-edited one must not be able to silence an alert. Everything that is
    # not a plain non-negative integer is treated as "never alerted" -- an
    # unquoted expansion of a non-numeric value aborts the whole run under
    # `set -u` from inside this function, which killed the watchdog outright.
    case "$last_at" in
      '' | *[!0-9]*) last_at=0 ;;
    esac
  fi
  age=$((now - last_at))
  # A stamp in the future means the clock moved backwards (a DST/ntp slip, a
  # copied state file). It is not a cooldown, so it must not suppress anything;
  # a negative age is treated as expired.
  if [ "$last_key" = "$key" ] && [ "$age" -ge 0 ] && [ "$age" -lt "$ALERT_COOLDOWN_SECONDS" ]; then
    log "still failing ($key); alert suppressed for another $((ALERT_COOLDOWN_SECONDS - age))s"
    return 0
  fi
  # Written atomically (tmp + mv on one filesystem) because `>` truncates first:
  # a crash between the two left an empty file, which the next run would read as
  # "never alerted" -- safe, but it silently re-enabled every alert.
  #
  # Two things are deliberately NOT covered by a test, and are not claimed to be:
  # the torn-write window itself (a test cannot kill the script between truncate
  # and write) and the mv failing. What IS tested is the consequence that
  # matters -- a state write that fails still logs a warning and still reaches
  # the operator, rather than silently disabling deduplication.
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
# of its own, so a single fault is byte-for-byte the alert it has always been
# and a second fault is appended to it rather than restarting the sentence.
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
# there is no point asking it about its dependencies, and the answer decides
# whether a restart is attempted at all.
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

# One wait for however many services were restarted, rather than one per service:
# both processes are already restarting by the time this starts, so waiting twice
# would only delay the alert.
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
    # Came back alive but still not ready. Same taxonomy as below: WHICH of
    # those it is decides what the operator is told.
    add_failure "live-not-ready:$ready_code" "$(readiness_verdict "$ready_code")"
  else
    add_failure "down:$live_code" "VCCircle backend down: liveness /health returned HTTP ${live_code:-none} after restart, readiness /ready/deep returned HTTP ${ready_code:-none} ($BASE)"
  fi
elif is_healthy "$live_rc" "$live_code" strict; then
  if ! is_healthy "$ready_rc" "$ready_code" strict; then
    # Alive, not ready. WHICH of those it is decides what the operator is told: a
    # 503 is a real verdict about the dependencies, while a refused, throttled or
    # broken PROBE says nothing at all about them and must not be dressed up as
    # one.
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
  # One alert for the whole set: the operator gets one mail and one webhook
  # POST naming every service, and the cooldown is keyed on that set, so a run
  # with two faults is deduplicated as the single fault it reports.
  alert "$fault_key" "ALERT: $failures"
  exit 1
fi

# Clear the fault state, so if something breaks again the next run alerts
# immediately instead of inheriting an unexpired cooldown.
rm -f "$STATE_FILE"
exit 0
