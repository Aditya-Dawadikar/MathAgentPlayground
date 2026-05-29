#!/bin/sh
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
OLLAMA_ENDPOINT="http://${OLLAMA_HOST}"

echo "Starting temporary Ollama bootstrap server for model setup"
OLLAMA_HOST="127.0.0.1:11434" ollama serve &
BOOTSTRAP_PID=$!

cleanup() {
  kill "$BOOTSTRAP_PID" 2>/dev/null || true
}

trap cleanup INT TERM

attempt=0
until ollama list >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 3 ]; then
    echo "Ollama did not become ready in time" >&2
    exit 1
  fi
  sleep 1
done

echo "Ollama bootstrap server is ready"

if ! ollama show "$OLLAMA_MODEL" >/dev/null 2>&1; then
  echo "Pulling model ${OLLAMA_MODEL}"
  ollama pull "$OLLAMA_MODEL"
else
  echo "Model ${OLLAMA_MODEL} already present"
fi

echo "Model ${OLLAMA_MODEL} is ready for serving"
echo "Stopping bootstrap server"
kill "$BOOTSTRAP_PID"
wait "$BOOTSTRAP_PID" 2>/dev/null || true

trap - INT TERM

echo "Starting Ollama server on ${OLLAMA_HOST}"
echo "Model ${OLLAMA_MODEL} is ready. You can now send requests to ${OLLAMA_ENDPOINT}/api/generate"
exec ollama serve