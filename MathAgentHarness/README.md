# LangGraph Agent API

A LangGraph ReAct agent exposed via FastAPI. Connects to a local Ollama instance for LLM inference.

## Prerequisites

- Python 3.11+
- Ollama running locally (see `../LocalLLM` for Docker setup)

## Setup

```powershell
# 1. Copy env config
Copy-Item .env.example .env

# 2. Install dependencies
pip install -r requirements.txt
```

Edit `.env` to change the model or Ollama URL:

```env
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Stateless single-turn chat |
| `POST` | `/sessions/{session_id}/chat` | Stateful multi-turn chat (history retained in memory per session) |

## Example Requests

### Stateless chat

```powershell
$body = @{ message = "What is 2 ** 10?" } | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/chat" `
  -ContentType "application/json" `
  -Body $body
```

### Stateless chat with a system prompt

```powershell
$body = @{
  message = "Who are you?"
  system_prompt = "You are a helpful assistant named Ada."
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/chat" `
  -ContentType "application/json" `
  -Body $body
```

### Stateful multi-turn chat

Each request to the same `session_id` continues the conversation:

```powershell
$session = "user-123"

# Turn 1
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/sessions/$session/chat" `
  -ContentType "application/json" `
  -Body (@{ message = "My name is Alice." } | ConvertTo-Json)

# Turn 2 — the agent remembers the name
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/sessions/$session/chat" `
  -ContentType "application/json" `
  -Body (@{ message = "What is my name?" } | ConvertTo-Json)
```

Session state lives in memory and is lost on server restart. Use a different `session_id` to start a fresh conversation.

## Built-in Tools

The LLM chooses the appropriate tool based on the operation requested.

| Tool | Operation | Description |
|------|-----------|-------------|
| `add` | `a + b` | Addition |
| `subtract` | `a - b` | Subtraction |
| `multiply` | `a * b` | Multiplication |
| `divide` | `a / b` | Division (errors on divide-by-zero) |
| `power` | `a ** b` | Exponentiation |
| `get_current_datetime` | — | Returns the current date and time |
