# Local Ollama Setup

This folder builds a small Ollama container that starts the Ollama server, pulls the model defined in `OLLAMA_MODEL`, and serves it on a local HTTP endpoint.

## Prerequisites

- Docker Desktop with WSL2 enabled
- NVIDIA GPU drivers installed on the host
- Docker GPU support working with `docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi`

## Files

- `Dockerfile`: extends the Ollama image and copies in the bootstrap script
- `docker-compose.yml`: builds and runs the container with GPU access and port `11434`
- `start-ollama.sh`: starts the server inside the container and pulls the configured model
- `.env.example`: configurable defaults for port and model name

## Start the service

From this folder:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
```

To choose a different model, edit `.env` before starting:

```text
OLLAMA_MODEL=qwen2.5:7b
```

The container bootstrap script will:

- start `ollama serve`
- wait for Ollama to become ready
- pull the model in `OLLAMA_MODEL` if it is not already present
- keep the Ollama server running on port `11434`

## Call the Ollama endpoint

Generate:

```powershell
$body = @{
  model = "llama3.2:1b"
  prompt = "Explain what Ollama does in one sentence."
  stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/generate" -ContentType "application/json" -Body $body
```

Chat:

```powershell
$body = @{
  model = "llama3.2:1b"
  messages = @(
    @{ role = "user"; content = "Say hello from the local Ollama endpoint." }
  )
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/chat" -ContentType "application/json" -Body $body
```

## OpenAI-compatible endpoint

Ollama also exposes OpenAI-compatible routes such as:

- `http://localhost:11434/v1/chat/completions`
- `http://localhost:11434/v1/embeddings`

## Notes

- The first container start can take a while because it may need to download the model.
- Model weights are persisted in the Docker volume `ollama-data`.
- If GPU access fails, verify the host Docker GPU setup first, then inspect `docker compose logs ollama`.