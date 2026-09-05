#!/usr/bin/env bash
# One-command local dev startup: runs the FastAPI backend and the Vite
# frontend together, tearing both down on Ctrl+C.
set -e

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "No .venv found - see README.md 'Installation' first." >&2
  exit 1
fi

source .venv/bin/activate

uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

( cd frontend && npm run dev ) &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT INT TERM

echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
wait
