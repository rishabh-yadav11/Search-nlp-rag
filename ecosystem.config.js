module.exports = {
  apps: [
    {
      name: "vccircle-backend",
      cwd: "/home/ubuntu/search-nlp-rag/backend",
      script: "venv/bin/python",
      args: "-m gunicorn -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8001 --timeout 120 app.main:app",
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: "5G",
      exp_backoff_restart_delay: 100,
      max_restarts: 10,
    },
    {
      name: "vccircle-frontend",
      cwd: "/home/ubuntu/search-nlp-rag/frontend",
      script: "node_modules/.bin/next",
      args: "start -p 3000",
      env: {
        NODE_ENV: "production",
      },
      max_memory_restart: "1G",
      exp_backoff_restart_delay: 100,
      max_restarts: 10,
    },
  ],
};
