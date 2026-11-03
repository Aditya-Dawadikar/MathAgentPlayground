# Math Agent Harness

LangGraph math agents exposed through FastAPI, with a SymPy-backed toolset of 62 tools that covers 10th-grade algebra, geometry, trigonometry and word problems. Every tool is evaluated for correctness and reliability.

## Prerequisites

- Python 3.11+
- An LLM backend: either Ollama running locally (see `../LocalLLM` for Docker setup) or a Gemini API key

## Setup

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
```

Key settings in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | ReAct agent model |
| `OLLAMA_MODEL_2` | `llama3.1:8b` | Planner model (and step repair) |
| `OLLAMA_JUDGE_MODEL` | `llama3.1:8b` | LLM-as-a-judge for planner evals |
| `GOOGLE_API_KEY`, `GEMINI_MODEL` | `gemini-2.5-flash` | Used when `LLM_PROVIDER=gemini` (`pip install langchain-google-genai`) |
| `EXECUTOR_MAX_STEP_RETRIES` | `2` | Retries per plan step |
| `EXECUTOR_REPAIR` | `1` | Let the LLM repair a step whose tool returned an error |
| `MATH_TOOL_TIMEOUT_SECONDS` | `10` | Abandon a tool call after this long |

## Run

From this folder:

```powershell
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are at http://localhost:8000/docs.

| Method | Path | Agent | Description |
|--------|------|-------|-------------|
| `GET` | `/health` | | Health check |
| `POST` | `/chat` | `agent_tool_call_correctness_v4` | Stateless single-turn chat with all math tools |
| `POST` | `/sessions/{session_id}/chat` | `agent_tool_call_correctness_v4` | Multi-turn chat with in-memory history |
| `POST` | `/plan` | `planner_agent_v2` | Returns a JSON tool-use plan, without executing it |
| `POST` | `/execution-trace` | `agent_execution_trace_v2` | Plans, executes and verifies. Returns the plan, traces, `final_output` and `verified`. Returns `422` with partial traces if a step fails |

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/execution-trace" `
  -ContentType "application/json" `
  -Body (@{ problem = "A rectangle's length is 3 more than its width and its perimeter is 26. What is its area?" } | ConvertTo-Json)
```

## Math tools (`app/math_tools/`)

Every tool runs its input through a sandboxed SymPy parser. The parser accepts `2x^2 + 3(x-1)`, `sqrt`, `abs`, `log2`, `pi` and `e`. Tools never raise: they return either a result envelope or an error envelope.

```json
{"result": "x = 4", "exact": "4", "decimal": 4.0, "latex": "4", "steps": ["..."], "...": "tool-specific fields"}
{"error": "parse_error", "message": "could not parse '2x +'"}
```

| Module | Tools |
|--------|-------|
| `arithmetic` | `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`, `nth_root`, `modulo`, `absolute_value`, `gcd`, `lcm`, `prime_factorize`, `fraction_simplify`, `evaluate_expression` |
| `expressions` | `simplify`, `expand`, `factor`, `substitute`, `rationalize_denominator`, `simplify_radical`, `combine_rational` |
| `equations` | `solve_equation`, `solve_quadratic`, `solve_system`, `solve_inequality`, `solve_for`, `verify_solution` |
| `polynomials` | `polynomial_degree`, `polynomial_divide`, `find_roots`, `complete_the_square`, `discriminant` |
| `functions` | `evaluate_function`, `compose_functions`, `inverse_function`, `domain_range`, `find_intercepts`, `vertex_of_parabola` |
| `coordinate` | `slope`, `line_equation`, `distance`, `midpoint`, `parallel_or_perpendicular` |
| `exponential` | `log`, `solve_exponential`, `arithmetic_sequence`, `geometric_sequence`, `sequence_from_terms` |
| `applied` | `percent`, `percent_change`, `ratio_proportion`, `unit_convert`, `simple_compound_interest` |
| `geometry` | `pythagorean`, `area`, `perimeter`, `volume`, `surface_area`, `trig`, `inverse_trig`, `solve_right_triangle` |
| `utility` | `get_current_datetime` |

Some behaviors worth knowing:

- **Equation solvers:** they verify every candidate root against the original equation and return extraneous roots separately. They also report identities (`all_real`) and contradictions (`none`).
- **Prompts:** `describe_tools()` generates the tool catalog used in the planner prompts, so new tools show up automatically.

## Agents (`app/`)

| Module | Description |
|--------|-------------|
| `agent_tool_call_correctness_v1` to `v3` | Baseline ReAct agents with only the 6 original arithmetic tools |
| `agent_tool_call_correctness_v4` | ReAct agent with all math tools |
| `planner_agent_v1` | Planner over the original arithmetic tools |
| `planner_agent_v2` | Planner over all math tools. Arguments can reference earlier steps (`"$step_1.solutions[0]"`, or inline `"2*($step_1) + 1"`), and equation plans end with `verify_solution` |
| `agent_execution_trace_v1` | Planner → deterministic executor for the arithmetic tools |
| `agent_execution_trace_v2` | Planner → executor for all math tools. It resolves step references and retries failures. When a tool returns an error, it asks the LLM to repair that step. Reports `verified` |
| `llm.py` | Chat-model factory that switches between Ollama and Gemini with `LLM_PROVIDER` |

## Tests and evals

Offline tests are deterministic and need no LLM:

```powershell
pytest -m "not llm"
```

- `tests/math_tools/`: happy and sad paths for every tool category, plus tests for the eval comparator
- `tests/test_execution_trace_v2_executor.py`: v2 executor reference resolution, verification, retries, repair and failure traces

LLM evals call a live model and are marked `llm`:

| Test | Dataset | Checks |
|------|---------|--------|
| `test_math_agent_tool_selection.py` | `math_tool_selection_cases.json` | First tool call and *semantically equivalent* arguments, plus key facts in the final answer. Sad paths include non-math prompts, divide by zero and no-solution cases |
| `test_math_agent_execution.py` | `math_execution_cases.json` | The final answer from the planner → executor, compared order-insensitively. Also checks that no verification step failed. Sad paths include declines, no solution and no real solution |
| `test_agent_tool_selection.py`, `test_planner_agent.py`, `test_agent_execution_trace.py` | original datasets | Baseline arithmetic evals |

```powershell
pytest tests/test_math_agent_tool_selection.py -n 4
pytest tests/test_math_agent_tool_selection.py --agent-module app.agent_tool_call_correctness_v3   # baseline
```

### Reliability experiments

```powershell
# Repeat any eval suite and count failures per case
python experiments/tool_planner/run_tool_planner_experiment.py --runs 5 --test-file tests/test_math_agent_execution.py
python experiments/tool_call_correctness/run_tool_selection_experiment.py --runs 10 --agent-module app.agent_tool_call_correctness_v4
```
