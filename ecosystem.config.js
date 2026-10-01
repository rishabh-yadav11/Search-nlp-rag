// pm2 process definitions for the two long-running services: the gunicorn/FastAPI
// API and the Next.js frontend.
//
// SINGLE source of truth: `./setup.sh services` starts from this file, not from
// inline `pm2 start` lines. Not a style preference -- pm2's CLI has no
// `--min-uptime` flag in ANY released version (4.x through 7.x all answer
// `unknown option` and exit 1), so an inline start cannot express min_uptime at
// all, and under `set -e` passing the flag would abort the services stage and
// leave the box with no backend. One definition also keeps the two startup
// paths from drifting, which backend/tests/test_deploy_config.py guards.
//
// Four knobs come from the environment and each defaults to exactly what
// setup.sh defaults to, so the two startup paths cannot produce
// differently-shaped processes on the same host.
//
// min_uptime decides which restarts pm2 counts as stable, and pm2's own default
// (1000ms) is unsafe here: a frontend that starts cleanly and dies seconds
// later has cleared that bar, so pm2 scores it STABLE. Stable restarts never
// count toward max_restarts and never trigger exp_backoff_restart_delay, so pm2
// hot-loops the broken process and the only symptom is a log file filling up.
// At 30s those restarts are unstable -- the state pm2's backoff and restart
// limit act on -- while still covering a cold first render.
//
// There is deliberately NO `health_check` block, though pm2 supports one. pm2
// has no CLI equivalent for it, so nothing can drift, but a check on an
// endpoint the app only serves intermittently (a 300s SSE stream, a cold model
// load) would restart healthy processes. Liveness is watched outside pm2 by
// deploy/healthcheck.sh instead.
const path = require("path");

// APP_ROOT, not a hardcoded home directory: `pm2 start ecosystem.config.js` has
// to work in whatever checkout it is run from.
const APP_ROOT = process.env.VCCIRCLE_ROOT || path.resolve(__dirname);

// GUNICORN_WORKERS is interpolated straight into gunicorn's argument list, where
// a bad value is not a mis-tuned backend but a process definition gunicorn cannot
// parse -- pm2 reports a start error and leaves no API at all. A digit test alone
// is not enough: a digit string too long for a double loses precision and Number
// stringifies it in EXPONENTIAL form (`--workers 1e+21`), which gunicorn's int()
// rejects, so the value must be a safe integer. 1024 workers on one host is a
// fork bomb, not a tuning decision, so anything outside 1..MAX falls back to 4,
// the same default setup.sh uses.
const MAX_WORKERS = 1024;

function positiveInt(raw, fallback) {
  if (!/^[0-9]+$/.test(String(raw))) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 && value <= MAX_WORKERS ? value : fallback;
}

const WORKERS = positiveInt(process.env.GUNICORN_WORKERS, 4);

// positiveInt's 1024 ceiling is a fork-bomb guard and is only correct for the
// worker count: routing the ports or min_uptime through it would silently
// replace their fallbacks with the same numbers, looking correct until someone
// changes one. These are the real bounds: the largest port a TCP listener can be
// given, and a year of milliseconds.
const MAX_PORT = 65535;
const MAX_MIN_UPTIME_MS = 365 * 24 * 60 * 60 * 1000;

function boundedInt(raw, fallback, max) {
  if (!/^[0-9]+$/.test(String(raw))) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 && value <= max ? value : fallback;
}

// Both default to the values setup.sh defaults to, so an operator overriding
// API_PORT or NEXT_PORT gets the same answer from either startup path.
const API_PORT = boundedInt(process.env.API_PORT, 8001, MAX_PORT);
const NEXT_PORT = boundedInt(process.env.NEXT_PORT, 3000, MAX_PORT);

// An unusable min_uptime makes the restart policy meaningless (NaN never compares
// as "older than", so every restart would look stable), so a bad value degrades
// to the known-good 30s.
const MIN_UPTIME = boundedInt(process.env.MIN_UPTIME_MS, 30000, MAX_MIN_UPTIME_MS);

// `setup.sh services` starts from this file, so a literal here would silently
// override an operator who ran `API_MAX_MEMORY=8G ./setup.sh services`: the
// knob would be accepted and then ignored, which is worse than not offering it.
const API_MAX_MEMORY = process.env.API_MAX_MEMORY || "5G";
const FRONTEND_MAX_MEMORY = process.env.FRONTEND_MAX_MEMORY || "1G";
const API_MAX_RESTARTS = boundedInt(process.env.API_MAX_RESTARTS, 10, 1000);
const RESTART_BACKOFF_MS = boundedInt(process.env.RESTART_BACKOFF_MS, 100, 60000);

module.exports = {
  apps: [
    {
      name: "vccircle-backend",
      cwd: path.join(APP_ROOT, "backend"),
      script: "venv/bin/python",
      // 127.0.0.1, not 0.0.0.0: nginx is the only thing that should reach the API.
      args: `-m gunicorn -k uvicorn.workers.UvicornWorker --workers ${WORKERS} --bind 127.0.0.1:${API_PORT} --timeout 120 app.main:app`,
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: API_MAX_MEMORY,
      exp_backoff_restart_delay: RESTART_BACKOFF_MS,
      max_restarts: API_MAX_RESTARTS,
      min_uptime: MIN_UPTIME,
    },
    {
      name: "vccircle-frontend",
      cwd: path.join(APP_ROOT, "frontend"),
      script: "node_modules/.bin/next",
      // -H 127.0.0.1 for the same reason: both listeners are reached through nginx.
      args: `start -H 127.0.0.1 -p ${NEXT_PORT}`,
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: FRONTEND_MAX_MEMORY,
      exp_backoff_restart_delay: RESTART_BACKOFF_MS,
      max_restarts: API_MAX_RESTARTS,
      min_uptime: MIN_UPTIME,
    },
  ],
};
