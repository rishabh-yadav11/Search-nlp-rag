#!/usr/bin/env bash
# Health-check the backend and restart it if unhealthy; log if a restart
# doesn't bring it back. Run from cron every few minutes.
#
# Which endpoint answers which question (#279):
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
#                  placeholder GEMINI_API_KEY.
# This script used to probe /health alone, which made a dependency outage or a
# placeholder key invisible -- the watchdog exited 0 forever.
#
# Behaviour:
#   1. Probe both. Ready -> exit 0, no output.
#   2. Alive but not ready -> alert WITHOUT restarting. The process is fine; a
#      restart cannot bring Qdrant back or fix a placeholder key, and restarting
#      anyway turns a dependency blip into an outage.
#   3. Not alive -> restart vccircle-backend via pm2, wait, re-probe.
#   4. Still not alive -> log, POST to HEALTHCHECK_WEBHOOK_URL (if set),
#      and print (cron mails on output if MAILTO is set).
#
# An alert is emitted once per fault, not once per run: a fault that stays
# unfixed is logged as "still failing" and otherwise stays silent until
# ALERT_COOLDOWN_SECONDS has passed, because the */5 cron cadence would
# otherwise mail and POST the same news ~288 times a day. The exit code still
# reports the fault on every run.
#
# Override (env): BASE, LOG, HEALTHCHECK_WEBHOOK_URL, RESTART_WAIT_SECONDS,
#                 STATE_FILE, ALERT_COOLDOWN_SECONDS.
set -u

BASE="${BASE:-http://localhost:8001}"
APP="vccircle-backend"
LOG="${LOG:-$HOME/search-nlp-rag/logs/healthcheck.log}"
RESTART_WAIT_SECONDS="${RESTART_WAIT_SECONDS:-8}"
# Cron runs this every few minutes, so a fault that stays fixed would re-alert
# forever: a persistent placeholder key is ~288 identical webhook POSTs and
# ~288 cron mails a day, which is how the one alert that matters gets ignored.
# The state file holds "<key> <epoch>" for the fault currently being reported;
# an identical fault inside the cooldown is logged once and otherwise silent.
# A healthy run clears it, so a later fault alerts immediately.
STATE_FILE="${STATE_FILE:-$LOG.state}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-3600}"

# cron has a minimal PATH, so pm2 may not be found. Include common locations.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

probe() {
  curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$BASE$1" 2>/dev/null
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

# $1 is a stable name for the fault, $2 the message. Re-alerts only when the
# fault changes or the cooldown has passed.
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

restarted=0

mkdir -p "$(dirname "$LOG")"

# Liveness first, and on its own: if the process is not up there is no point
# asking it about its dependencies, and the answer decides whether a restart is
# attempted at all.
live_code=$(probe /health)
if [ "$live_code" = "200" ]; then
  ready_code=$(probe /ready/deep)
else
  ready_code=""
  log "backend not alive (liveness /health returned HTTP ${live_code:-none}); restarting $APP"
  # NOTE: this replays the argv pm2 stored at start time — `--update-env`
  # refreshes environment variables, not the argument list. It is correct for
  # recovering a sick process, but it can NEVER apply a change to the process
  # options. Changing the API bind or any other pm2 option requires
  # `./setup.sh services`, which re-registers the process and re-saves the dump.
  pm2 restart "$APP" --update-env >/dev/null 2>&1 || true
  restarted=1

  sleep "$RESTART_WAIT_SECONDS"
  live_code=$(probe /health)
  ready_code=$(probe /ready/deep)
fi

if [ "$live_code" = "200" ] && [ "$ready_code" = "200" ]; then
  [ "$restarted" = "1" ] && log "backend recovered after restart"
  # Clear the fault state, so if the backend breaks again the next run alerts
  # immediately instead of inheriting an unexpired cooldown.
  rm -f "$STATE_FILE"
  exit 0
fi

if [ "$live_code" = "200" ]; then
  # Alive, not ready. WHICH of those it is decides what the operator is told: a
  # 503 is a real verdict about the dependencies, while a refused, throttled or
  # broken PROBE says nothing at all about them and must not be dressed up as one.
  case "$ready_code" in
    503)
      msg="ALERT: VCCircle backend alive but not ready: /ready/deep returned HTTP 503 while the process is up ($BASE). Not restarting: check Qdrant, the loaded models, and GEMINI_API_KEY."
      ;;
    403 | 404)
      msg="ALERT: VCCircle readiness probe REFUSED (HTTP $ready_code) at $BASE/ready/deep -- this is not a verdict about the backend. It answers only a direct loopback caller sending no X-Forwarded-For, so BASE points somewhere it cannot be reached (a reverse proxy, another host, or a port the API is not bound to)."
      ;;
    429)
      msg="ALERT: VCCircle readiness probe was rate limited (HTTP 429) at $BASE/ready/deep -- the watchdog is sharing a budget it should not share. This run says nothing about the backend's health."
      ;;
    500)
      msg="ALERT: VCCircle readiness probe failed internally (HTTP 500) at $BASE/ready/deep -- the probe itself raised, so no dependency verdict was reached. See the backend log."
      ;;
    *)
      msg="ALERT: VCCircle backend alive but readiness unreachable: /ready/deep returned HTTP ${ready_code:-none} at $BASE. Not restarting: the process answered liveness."
      ;;
  esac
  alert "live-not-ready:$ready_code" "$msg"
  exit 1
fi

msg="ALERT: VCCircle backend down: liveness /health returned HTTP ${live_code:-none} after restart, readiness /ready/deep returned HTTP ${ready_code:-none} ($BASE)"
alert "down:$live_code" "$msg"
exit 1
