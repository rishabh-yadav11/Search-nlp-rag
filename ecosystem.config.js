// pm2 process definitions for the two long-running services: the gunicorn/FastAPI
// API and the Next.js frontend.
//

const path = require("path");

const APP_ROOT = process.env.VCCIRCLE_ROOT || path.resolve(__dirname);


const MAX_WORKERS = 1024;

function positiveInt(raw, fallback) {
  if (!/^[0-9]+$/.test(String(raw))) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 && value <= MAX_WORKERS ? value : fallback;
}

const WORKERS = positiveInt(process.env.GUNICORN_WORKERS, 4);

const MAX_PORT = 65535;
const MAX_MIN_UPTIME_MS = 365 * 24 * 60 * 60 * 1000;

function boundedInt(raw, fallback, max) {
  if (!/^[0-9]+$/.test(String(raw))) {
    return fallback;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 && value <= max ? value : fallback;
}

const API_PORT = boundedInt(process.env.API_PORT, 8001, MAX_PORT);
const NEXT_PORT = boundedInt(process.env.NEXT_PORT, 3000, MAX_PORT);


const MIN_UPTIME = boundedInt(process.env.MIN_UPTIME_MS, 30000, MAX_MIN_UPTIME_MS);

// The four knobs below, and MIN_UPTIME above, MUST stay equal to the matching
// defaults in setup.sh. `./setup.sh services` starts pm2 from this file and
// EXPORTS them to it, but the documented deploy step is `pm2 restart
// ecosystem.config.js --update-env`, and an ordinary shell exports none of
// them -- so on a redeploy these fallbacks are what production actually runs.
// Nothing checks them against setup.sh any more, so change them together.
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
