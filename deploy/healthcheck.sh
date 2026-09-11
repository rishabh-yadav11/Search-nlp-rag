#!/usr/bin/env bash
# Health-check the backend and restart it if unhealthy; log if a restart
# doesn't bring it back. Run from cron every few minutes.
#
# Behaviour:
#   1. Probe /health (liveness). Healthy -> exit 0, no output.
#   2. Unhealthy -> restart vccircle-backend via pm2, wait, re-probe.
#   3. Still unhealthy -> log, POST to HEALTHCHECK_WEBHOOK_URL (if set),
#      and print (cron mails on output if MAILTO is set).
#
# Override (env): BASE, LOG, HEALTHCHECK_WEBHOOK_URL.
set -u

BASE="${BASE:-http://localhost:8001}"
APP="vccircle-backend"
LOG="${LOG:-/home/ubuntu/search-nlp-rag/logs/healthcheck.log}"

# cron has a minimal PATH, so pm2 may not be found. Include common locations.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

WEBHOOK="${HEALTHCHECK_WEBHOOK_URL:-}"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }

probe() {
  curl -fsS -m 10 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null
}

post_webhook() {
  local msg="$1"
  [ -z "$WEBHOOK" ] && return 0
  curl -fsS -m 10 -X POST "$WEBHOOK" \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"$msg\"}" >/dev/null 2>&1 || true
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
  log "backend recovered after restart"
  exit 0
fi

msg="ALERT: VCCircle backend down: /health returned HTTP ${code:-none} after restart ($BASE)"
log "$msg"
post_webhook "$msg"
echo "$msg"
exit 1
