#!/bin/bash
set -e

echo "Starting FinDebate Portal..."

cd "$(dirname "$0")/backend"
python3 -m pip install fastapi uvicorn python-dotenv httpx sse-starlette pydantic chromadb groq google-genai -q
FINDEBATE_P4_PROVIDER="${FINDEBATE_P4_PROVIDER:-groq}" \
FINDEBATE_P5_PROVIDER="${FINDEBATE_P5_PROVIDER:-groq}" \
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

cd ../frontend
npm install -q
npm run dev

kill "$BACKEND_PID" 2>/dev/null || true
