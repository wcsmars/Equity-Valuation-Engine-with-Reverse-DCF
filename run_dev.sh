#!/usr/bin/env bash
# Launch the full Equity Research Automation: FastAPI backend + Next.js frontend.
#
#   cp .env.example .env   # add your ANTHROPIC_API_KEY / FMP_API_KEY (both optional)
#   ./run_dev.sh
#
# Then open http://127.0.0.1:3000
set -euo pipefail
cd "$(dirname "$0")"

# Load .env into the environment (so the backend sees the keys).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VENV=.venv
if [ ! -d "$VENV" ]; then
  echo "==> Creating Python venv"
  python3 -m venv "$VENV"
fi
echo "==> Installing backend deps (engine + FastAPI)"
"$VENV/bin/python" -m pip install -q -r requirements.txt -r backend/requirements-backend.txt

if [ ! -d frontend/node_modules ]; then
  echo "==> Installing frontend deps"
  (cd frontend && npm ci --no-audit --no-fund)
fi

# Run each server in its own process group so cleanup can stop the whole
# tree: npm -> sh -> next dev -> next-server would otherwise outlive a plain
# kill of the npm PID.
set -m

echo "==> Starting FastAPI on http://127.0.0.1:8000"
"$VENV/bin/python" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACK=$!

echo "==> Starting Next.js on http://127.0.0.1:3000"
(cd frontend && npm run dev -- --hostname 127.0.0.1) &
FRONT=$!

cleanup() {
  trap - EXIT INT TERM HUP
  kill -TERM -- "-$BACK" "-$FRONT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM HUP

echo ""
echo "  Backend : http://127.0.0.1:8000  (docs at /docs)"
echo "  Frontend: http://127.0.0.1:3000"
echo "  Ctrl-C to stop both."
echo ""
wait
