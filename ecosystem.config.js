// pm2 process definitions for the two long-running services: the gunicorn/FastAPI
// API and the Next.js frontend.
//
// This file is the SINGLE source of truth for the pm2 process options, and
// `./setup.sh services` starts from it rather than from inline `pm2 start` lines.
// That is not a style preference. pm2's CLI has no `--min-uptime` flag in ANY
// released version (4.x, 5.x, 6.x and 7.x all answer `error: unknown option
// '--min-uptime'` and exit 1), while `min_uptime` in an ecosystem file is
// honoured at runtime — pm2 reads it in God.js when classifying a restart as
// stable or unstable. An inline `pm2 start` therefore cannot express it at all,
// and under `set -e` passing the flag would abort the services stage outright
// and leave the box with no backend. Starting from this file is what makes the
// option reachable, and having one definition is also what stops the two
// startup paths from drifting apart, which backend/tests/test_deploy_config.py
// guards.
//
// Four knobs are read from the environment, and each defaults to exactly what
// setup.sh defaults to: VCCIRCLE_ROOT is the checkout root (which is what
// setup.sh resolves its own SCRIPT_DIR to), GUNICORN_WORKERS is 4, and
// MIN_UPTIME_MS is 30000. A different default on either side would mean the two
// startup paths quietly produce differently-shaped processes on the same host.
//
// min_uptime decides which restarts pm2 counts as stable, and pm2's own default
// is not safe here: 1000ms. A Next.js frontend that starts cleanly and then
// dies four seconds later — a port it cannot rebind after a half-dead previous
// process, a missing `.next` build, an OOM on the first render — has cleared
// that bar, so pm2 scores every one of those restarts as STABLE. Stable
// restarts never count toward max_restarts and never trigger
// exp_backoff_restart_delay, so pm2 hot-loops the broken process forever,
// restarting it as fast as it can die, and the only symptom is a log file that
// fills up. At 30s those same restarts are reclassified as unstable, which is
// the state pm2's backoff and restart limit actually act on. 30s also has to be
// long enough to cover a cold first render, or a healthy slow start would be
// treated as a crash.
//
// There is deliberately NO `health_check` block here, even though pm2 supports
// one. pm2 has no CLI equivalent for the option, and since `./setup.sh services`
// now starts from this file there is nothing to drift — but a health_check that
// depends on an endpoint the app only serves intermittently (a 300s SSE stream,
// a cold model load) would restart a healthy process anyway, which is a worse
// failure than not checking at all. Liveness is watched from outside pm2
// instead, by deploy/healthcheck.sh, which probes the API and the frontend and
// restarts whichever stopped answering.
const path = require("path");

// APP_ROOT, not a hardcoded home directory: `pm2 start ecosystem.config.js` has
// to work in whatever checkout it is run from, not only on the box the path was
// written on.
const APP_ROOT = process.env.VCCIRCLE_ROOT || path.resolve(__dirname);

// GUNICORN_WORKERS is interpolated straight into gunicorn's argument list, which
// is a place where a typo is not a small problem: `--workers abc` or
// `--workers 0` is not a merely mis-tuned backend, it is a process definition
// gunicorn cannot parse, so pm2 reports a start error and leaves no API running
// at all, with the health check reporting on a port nothing is listening on.
// The digit test alone is not sufficient, and the gap is worth stating because
// it is the same failure this function exists to prevent, reached a different
// way. A "one or more digits" pattern accepts a string of any length, parseInt
// on one longer than a double can hold loses precision, and the resulting Number
// stringifies in EXPONENTIAL form: GUNICORN_WORKERS=999999999999999999999 yields
// `--workers 1e+21`, and gunicorn's int() rejects that outright with
// ValueError. So the value has to be a safe integer, not merely digits, and it
// has to be a plausible one: 1024 gunicorn workers on one host is not a tuning
// decision, it is a fork bomb, and letting it through converts a config mistake
// into an exhausted box. Anything outside 1..MAX falls back to 4, the same
// default setup.sh uses, so a bad value degrades to the known-good
// configuration instead of to an unbootable service.
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
// worker count, so it is NOT reused for the ports or for min_uptime: both are
// legitimately above 1024, and routing them through it would silently replace
// API_PORT=8001, NEXT_PORT=3000 and MIN_UPTIME_MS=30000 with the fallback,
// which happens to be the same number and so would look like it worked until
// anyone changed one of them. This bound is the real one for these three: the
// largest port a TCP listener can be given, and a year of milliseconds.
const MAX_PORT = 65535;
const MAX_MIN_UPTIME_MS = 365 * 24 * 60 * 60 * 1000;

function boundedInt(raw, fallback, max) {
  if (!/^[0-9]+$/.test(String(raw))) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 && value <= max ? value : fallback;
}

// The two ports. They were literals here, which made `./setup.sh services` and
// this file disagree about where the services listen the moment an operator
// overrode API_PORT or NEXT_PORT -- and since `setup.sh services` now starts
// from this file, that disagreement would have become a real one rather than a
// latent one. Both default to the values setup.sh defaults to.
const API_PORT = boundedInt(process.env.API_PORT, 8001, MAX_PORT);
const NEXT_PORT = boundedInt(process.env.NEXT_PORT, 3000, MAX_PORT);

// An unusable min_uptime makes the restart policy meaningless (NaN never
// compares as "older than", so every restart would look stable), so a bad value
// degrades to the known-good 30s rather than being passed through.
const MIN_UPTIME = boundedInt(process.env.MIN_UPTIME_MS, 30000, MAX_MIN_UPTIME_MS);

// The pm2 tuning knobs are read from the environment for the same reason and
// the same way. `setup.sh services` starts from this file, so a literal here
// would silently override an operator who ran
// `API_MAX_MEMORY=8G ./setup.sh services` -- the knob would be accepted and
// then ignored, which is worse than not offering it.
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
      // 127.0.0.1, not 0.0.0.0: nginx is the only thing that should reach the
      // API, so the bind must not publish it to every interface.
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
      // -H 127.0.0.1 for the same reason: both listeners are reached through
      // nginx and neither is meant to be reachable off-box.
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
