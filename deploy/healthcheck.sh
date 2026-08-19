#!/usr/bin/env bash
# Health-check the backend and restart it if unhealthy; log if a restart
# doesn't bring it back. Run from cron every few minutes.
#
# Behaviour:
#   1. Probe /health (liveness). Healthy -> exit 0, no output.
#   2. Unhealthy -> restart vccircle-backend via pm2, wait, re-probe.
#   3. Still unhealthy -> print (cron mails on output if MAILTO is set) and log.
#
# Override (env): HEALTHCHECK_BACKEND_PORT.
set -u

BASE="${BASE:-http://localhost:8001}"
PORT="${HEALTHCHECK_BACKEND_PORT:-8001}"
APP="vccircle-backend"
LOG="${LOG:-/home/ubuntu/search-nlp-rag/logs/healthcheck.log}"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

probe() {
  curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null
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

log "ALERT: VCCircle backend down: /health returned HTTP ${code:-none} after restart on port $PORT"
echo "VCCircle backend down: /health returned HTTP ${code:-none} after restart on port $PORT"
exit 1
