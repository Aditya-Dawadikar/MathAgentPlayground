# MathAgentPlayground

A playground for building and **evaluating** tool-using LLM agents that solve math problems up to a 10th-grade level: algebra, geometry, trigonometry and word problems. The agents use a SymPy-backed toolset of 62 tools and run against a local Ollama model or Gemini. The focus is measuring how reliable an agent is, not only whether it works once:

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

## Interactive chat

Ask word problems in the terminal and watch the agent work: the plan, each step's intermediate result, verification and the final answer. It remembers the conversation, so you can ask follow-ups like "what if the perimeter were 30 instead?" or "why did you add 3?".

```powershell
cd MathAgentHarness
python -m app.cli --model qwen3:8b
```

See [MathAgentHarness/README.md](MathAgentHarness/README.md#interactive-chat-cli) for commands and how the memory works.

## Math tools

`MathAgentHarness/app/math_tools/` contains 62 SymPy-backed tools covering arithmetic, expressions, equations and inequalities, polynomials, functions, coordinate geometry, exponents and logarithms, sequences, word problems, geometry and trigonometry. Every tool parses input through a sandbox and returns a JSON result or error envelope instead of raising. The equation solvers verify candidate roots and report extraneous ones. See [MathAgentHarness/README.md](MathAgentHarness/README.md#math-tools-appmath_tools) for the full list.

## Agents

The v1 to v3 agents use only the original arithmetic tools (`add`, `subtract`, `multiply`, `divide`, `power`, `get_current_datetime`). The v4 agent, `planner_agent_v2` and `agent_execution_trace_v2` use the full math toolset.

| Module (`MathAgentHarness/app/`) | Description |
|----------------------------------|-------------|
| `agent_tool_call_correctness_v1.py` | Baseline LangGraph ReAct agent with a minimal prompt |
| `agent_tool_call_correctness_v2.py` | Adds a detailed system prompt with tool-calling rules and richer tool docstrings |
| `agent_tool_call_correctness_v3.py` | Same as v2, but defaults to the larger `llama3.1:8b` model |
| `planner_agent_v1.py` | Returns a structured JSON plan of tool calls (steps, arguments, order sensitivity) and declines non-math queries. Does not execute anything |
| `agent_execution_trace_v1.py` | A planner → executor `StateGraph` that runs the plan step by step, retries failed steps, and returns the plan, intermediate results, event and execution traces, and the final output |
| `agent_tool_call_correctness_v4.py` | ReAct agent with all math tools |
| `planner_agent_v2.py` | Planner over all math tools. Steps can reference earlier results (`"$step_1.solutions[0]"`), and plans end with `verify_solution` |
| `agent_execution_trace_v2.py` | Planner → executor over all math tools. It resolves step references, uses the LLM to repair steps whose tool returned an error, and reports whether the answer was `verified` |

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
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` (needs `GOOGLE_API_KEY` and `pip install langchain-google-genai`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Model used by the ReAct agents |
| `OLLAMA_MODEL_2` | `llama3.1:8b` | Model used by the planners |
| `OLLAMA_JUDGE_MODEL` | `llama3.1:8b` | Model used as the LLM-as-a-judge in planner tests |
| `EXECUTOR_MAX_STEP_RETRIES` | `2` | Retries per step in the execution-trace executor |
| `EXECUTOR_REPAIR` | `1` | Let the LLM repair a step whose tool returned an error |

Pull every model you reference into Ollama, for example `docker exec local-ollama ollama pull llama3.1:8b`.

### 3. Run the API

Start the server from `MathAgentHarness/`:

```powershell
cd MathAgentHarness
uvicorn app.main:app --reload --port 8000
```

Interactive docs are at http://localhost:8000/docs.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Stateless single-turn chat (`{"message", "system_prompt"?}`) |
| `POST` | `/sessions/{session_id}/chat` | Multi-turn chat with in-memory history |
| `POST` | `/plan` | Planner agent v2: returns a JSON tool plan (`{"problem"}`) |
| `POST` | `/execution-trace` | Plans, executes and verifies, then returns the full trace plus `verified` (`{"problem"}`). Returns `422` with partial traces if a step fails |

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/execution-trace" `
  -ContentType "application/json" `
  -Body (@{ problem = "Solve 2x^2 + 4x - 1 = 0 exactly." } | ConvertTo-Json)
```

## Evaluation

The tests use [openevals](https://github.com/langchain-ai/openevals). The datasets are in `MathAgentHarness/tests/data/`.

| Test | Dataset | Method |
|------|---------|--------|
| `test_agent_tool_selection.py` | `tool_selection_cases.json` | Exact JSON match on tool name and arguments |
| `test_planner_agent.py` | `tool_planner_cases.json` | LLM-as-a-judge comparison against the expected plan |
| `test_agent_execution_trace.py` | `tool_execution_trace_cases.json` | JSON match on the normalized execution and event traces |
| `test_math_agent_tool_selection.py` | `math_tool_selection_cases.json` | First tool and *semantically equivalent* arguments for every tool category, including sad paths |
| `test_math_agent_execution.py` | `math_execution_cases.json` | Final answers from the v2 planner → executor, including declines and no-solution cases |

Deterministic tests need no LLM. They cover happy and sad paths for every tool category, plus the v2 executor:

```powershell
pytest -m "not llm"
```

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

# Planner (default 5 runs, parallel pytest workers); --test-file repeats any other eval suite
python experiments/tool_planner/run_tool_planner_experiment.py --runs 5 --workers auto
python experiments/tool_planner/run_tool_planner_experiment.py --runs 5 --test-file tests/test_math_agent_execution.py
```

Screenshots of past results (`agent_baseline.png`, `agent_v2.png`, `agent_v3.png`, `tool_planner.png`) are saved next to each script.
