const path = require("path");

module.exports = {
  apps: [
    {
      name: "vccircle-backend",
      cwd: path.join(__dirname, "backend"),
      script: "venv/bin/python",
      args: "-m gunicorn -k uvicorn.workers.UvicornWorker --workers 4 --bind 127.0.0.1:8001 --timeout 120 app.main:app",
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: "5G",
      exp_backoff_restart_delay: 100,
      max_restarts: 10,
    },
    {
      name: "vccircle-frontend",
      cwd: path.join(__dirname, "frontend"),
      script: "node_modules/.bin/next",
      args: "start -H 127.0.0.1 -p 3000",
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: "1G",
      exp_backoff_restart_delay: 100,
      max_restarts: 10,
    },
  ],
};
