#!/usr/bin/env bash
# Health-check the backend and restart it if unhealthy; alert if a restart
# doesn't bring it back. Run from cron every few minutes.
#
# Behaviour:
#   1. Probe /health (liveness). Healthy -> exit 0, no output.
#   2. Unhealthy -> restart vccircle-backend via pm2, wait, re-probe.
#   3. Still unhealthy -> print (cron mails on output if MAILTO is set) and,
#      when HEALTHCHECK_WEBHOOK_URL is set, POST a JSON alert to it.
#
# Overrides (env): HEALTHCHECK_WEBHOOK_URL, HEALTHCHECK_BACKEND_PORT.
set -u

BASE="${BASE:-http://localhost:8001}"
PORT="${HEALTHCHECK_BACKEND_PORT:-8001}"
APP="vccircle-backend"
LOG="${LOG:-/home/ubuntu/search-nlp-rag/logs/healthcheck.log}"
WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

probe() {
  curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null
}

alert() {
  local msg="$1"
  log "ALERT: $msg"
  if [ -n "$WEBHOOK" ]; then
    curl -fsS -m 10 -X POST -H 'Content-Type: application/json' \
      -d "{\"text\":\"$msg\"}" "$WEBHOOK" >/dev/null 2>&1 || log "alert webhook failed"
  fi
}

mkdir -p "$(dirname "$LOG")"

code=$(probe)
if [ "$code" = "200" ]; then
  exit 0
fi

log "backend unhealthy (HTTP $code); restarting $APP"
pm2 restart "$APP" --update-env >/dev/null 2>&1 || true

sleep 8
code=$(probe)
if [ "$code" = "200" ]; then
  log "backend recovered after restart (HTTP $code)"
  exit 0
fi

alert "VCCircle backend down: /health returned HTTP ${code:-none} after restart on port $PORT"
exit 1