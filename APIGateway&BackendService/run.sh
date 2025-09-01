#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
exec uvicorn app.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
