# MathAgentPlayground

A playground for building and **evaluating** small tool-using LLM agents that solve math problems. Everything runs against a local Ollama model. The focus is measuring how reliable an agent is, not only whether it works once:

- **Tool-call correctness**: does the agent pick the right tool with the right arguments?
- **Planning**: can a planner agent break a word problem into the correct ordered steps?
- **Execution traces**: does a planner + executor graph produce the expected step-by-step trace and final result?

## Repository layout

```
MathAgentPlayground/
├── LocalLLM/            # Docker setup for a GPU-backed Ollama server
└── MathAgentHarness/    # LangGraph agents, FastAPI service, tests, and experiments
    ├── app/             # Agent implementations + FastAPI app
    ├── tests/           # pytest suites + JSON case datasets
    └── experiments/     # Repeated-run reliability experiments + result screenshots
```

| Component | Description | Docs |
|-----------|-------------|------|
| `LocalLLM` | Builds an Ollama container, pulls `OLLAMA_MODEL`, and serves it on port `11434` | [LocalLLM/README.md](LocalLLM/README.md) |
| `MathAgentHarness` | LangGraph agents served through FastAPI, with evaluation tests | [MathAgentHarness/README.md](MathAgentHarness/README.md) |

## Agents

All agents share the same set of tools: `add`, `subtract`, `multiply`, `divide`, `power`, `get_current_datetime`.

| Module (`MathAgentHarness/app/`) | Description |
|----------------------------------|-------------|
| `agent_tool_call_correctness_v1.py` | Baseline LangGraph ReAct agent with a minimal prompt |
| `agent_tool_call_correctness_v2.py` | Adds a detailed system prompt with tool-calling rules and richer tool docstrings |
| `agent_tool_call_correctness_v3.py` | Same as v2, but defaults to the larger `llama3.1:8b` model |
| `planner_agent_v1.py` | Returns a structured JSON plan of tool calls (steps, arguments, order sensitivity) and declines non-math queries. Does not execute anything |
| `agent_execution_trace_v1.py` | A planner → executor `StateGraph` that runs the plan step by step, retries failed steps, and returns the plan, intermediate results, event and execution traces, and the final output |

## Quick start

### 1. Start the local LLM

You need Docker Desktop with WSL2 and NVIDIA GPU support.

```powershell
cd LocalLLM
Copy-Item .env.example .env
docker compose up --build -d
```

### 2. Install the harness

You need Python 3.11+.

```powershell
cd MathAgentHarness
Copy-Item .env.example .env
pip install -r requirements.txt
```

Settings in `MathAgentHarness/.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Model used by the agents |
| `OLLAMA_JUDGE_MODEL` | `llama3.1:8b` | Model used as the LLM-as-a-judge in planner tests |
| `EXECUTOR_MAX_STEP_RETRIES` | `2` | Retries per step in the execution-trace executor |

Pull every model you reference into Ollama, for example `docker exec local-ollama ollama pull llama3.1:8b`.

### 3. Run the API

`app/main.py` uses imports relative to the `app` folder, so start the server from inside `app/`:

```powershell
cd MathAgentHarness/app
uvicorn main:app --reload --port 8000
```

Interactive docs are at http://localhost:8000/docs.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Stateless single-turn chat (`{"message", "system_prompt"?}`) |
| `POST` | `/sessions/{session_id}/chat` | Multi-turn chat with in-memory history |
| `POST` | `/plan` | Planner agent: returns a JSON tool plan (`{"problem"}`) |
| `POST` | `/execution-trace` | Plans and executes, returns the full trace (`{"problem"}`). Returns `422` with partial traces if a step fails |

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/execution-trace" `
  -ContentType "application/json" `
  -Body (@{ problem = "Evaluate (18 + 7) * 4." } | ConvertTo-Json)
```

## Evaluation

The tests use [openevals](https://github.com/langchain-ai/openevals). The datasets are in `MathAgentHarness/tests/data/`.

| Test | Dataset | Method |
|------|---------|--------|
| `test_agent_tool_selection.py` | `tool_selection_cases.json` | Exact JSON match on tool name and arguments |
| `test_planner_agent.py` | `tool_planner_cases.json` | LLM-as-a-judge comparison against the expected plan |
| `test_agent_execution_trace.py` | `tool_execution_trace_cases.json` | JSON match on the normalized execution and event traces |

Run the tests from `MathAgentHarness/`. To test a different agent version, pass `--agent-module` or set the `AGENT_MODULE` environment variable:

```powershell
pytest tests -n auto
pytest tests/test_agent_tool_selection.py --agent-module app.agent_tool_call_correctness_v3
```

### Reliability experiments

LLM output is nondeterministic, so the experiment scripts run each suite several times and count failures per case:

```powershell
# Tool-call correctness (default 10 runs)
python experiments/tool_call_correctness/run_tool_selection_experiment.py --runs 10 --agent-module app.agent_tool_call_correctness_v3

# Planner (default 5 runs, parallel pytest workers)
python experiments/tool_planner/run_tool_planner_experiment.py --runs 5 --workers auto
```

Screenshots of past results (`agent_baseline.png`, `agent_v2.png`, `agent_v3.png`, `tool_planner.png`) are saved next to each script.
