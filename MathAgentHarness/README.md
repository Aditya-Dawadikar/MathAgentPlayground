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
| `OLLAMA_REASONING` | unset | `false` turns off thinking on models such as `qwen3` (much faster JSON planning) |

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
| `POST` | `/sessions/{session_id}/solve` | `chat_session` | Step-by-step solving with conversation memory (follow-ups, explanations) |
| `DELETE` | `/sessions/{session_id}` | | Forget a step-by-step chat session |
| `POST` | `/execution-trace` | `agent_execution_trace_v2` | Plans, executes and verifies. Returns the plan, traces, `final_output` and `verified`. Returns `422` with partial traces if a step fails |

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/execution-trace" `
  -ContentType "application/json" `
  -Body (@{ problem = "A rectangle's length is 3 more than its width and its perimeter is 26. What is its area?" } | ConvertTo-Json)
```

## Interactive chat (CLI)

Ask word problems in the terminal and see the agent's work: the plan, each tool step with its intermediate result, verification and the final answer. It remembers the conversation, so follow-ups work.

```powershell
python -m app.cli                                  # interactive
python -m app.cli --model qwen3:8b                 # pick the planner model
python -m app.cli --no-trace                       # answers only
python -m app.cli --no-color                       # plain text (or set NO_COLOR=1)
python -m app.cli -f problem.txt -q "what if the area were 160 m² instead?"   # problem from a file, then a follow-up
python -m app.cli -q "Solve x^2 - 5x + 6 = 0" -q "what if the constant term were 4 instead?"   # scripted turns
```

```
you> A rectangle's length is 3 more than its width and its perimeter is 26. What is its area?
Plan (2 steps)
  1. solve_equation  Solve for the width (w) of the rectangle.
  2. substitute      Calculate the area using the width found in step 1.

Step 1  solve_equation(equation='2*(w + (w + 3)) = 26')
        -> w = 5
           - checked 1 candidate solution(s) in the original equation
Step 2  substitute(expression='w*(w + 3)', values={'w': 5})
        -> 40

Answer: 40

you> what if the perimeter were 30 instead?
Interpreting as: A rectangle's length is 3 more than its width and its perimeter is 30. What is its area?
...
Answer: 54

you> why did you add 3?
I added 3 because the length of the rectangle is 3 more than its width ...
```

Commands: `/help`, `/reset` (forget the conversation), `/history` (what the agent remembers), `/trace on|off`, `/exit`.

**Long and multi-line problems.** Problems are never cut short:
- **Paste** a multi-line word problem: the lines are collected into one message.
- **Type** `"""` on its own line, then the problem, then `"""` again, or end a line with `\` to continue it.
- **Load** a problem from a file with `-f problem.txt`. `-q` and `-f` turns run in the order you give them.

Pasted text is cleaned up without changing its content:
- Terminal escape codes (including bracketed-paste markers), stray control characters, tabs and Windows line endings are normalized.
- Backslashes and quotes are kept as typed, so `rac` or a literal `
` stays as it is.
- Unicode math is understood: `x²`, `×`, `÷`, `−`, `≤`, `≥`, `≠`, `π`, `√`, `½`.
- Memory keeps the full text of recent problems, so follow-ups on long problems lose nothing.

Output is colour-coded in a terminal:
- plan headings blue, tool names cyan, intermediate results green
- purposes and notes dimmed, repairs and notices yellow, errors red
- the final answer in bold green

Colours turn off automatically when output is piped or redirected, or when `NO_COLOR` is set. `FORCE_COLOR=1` forces them on. Edit `STYLES` in `app/cli.py` to change the scheme.

How it works (`app/chat_session.py`, `app/memory.py`):

1. **Router.** One LLM call reads a compact memory summary and classifies the message:
   - `solve`: a new problem, or a follow-up rewritten into a self-contained problem ("Interpreting as: ...")
   - `explain`: answered from the recorded steps, with no tools
   - `clarify`: a follow-up when there is nothing to follow up on
   - `decline`: not math
2. **Plan and execute.** Solve turns go through the v2 planner and executor. Steps are streamed to the terminal as they run.
3. **Memory.** Each turn's interpretation, steps, answer and verification are stored. The last 4 turns are kept in detail and older ones as one-liners, and the oldest are dropped.

The same memory is available over HTTP: `POST /sessions/{id}/solve` with `{"message": ...}`, and `DELETE /sessions/{id}` to forget a session.

Tested with `qwen3:8b` on Ollama. For thinking models, set `OLLAMA_REASONING=false` so routing and planning stay fast.

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
- `tests/test_chat_session.py`: chat memory, routing outcomes, follow-ups, explanations, declines, errors, the CLI (rendering, commands, interactive loop) and the session API, using a fake LLM

LLM evals call a live model and are marked `llm`:

| Test | Dataset | Checks |
|------|---------|--------|
| `test_math_agent_tool_selection.py` | `math_tool_selection_cases.json` | First tool call and *semantically equivalent* arguments, plus key facts in the final answer. Sad paths include non-math prompts, divide by zero and no-solution cases |
| `test_math_agent_execution.py` | `math_execution_cases.json` | The final answer from the planner → executor, compared order-insensitively. Also checks that no verification step failed. Sad paths include declines, no solution and no real solution |
| `test_math_chat_followups.py` | `math_followup_cases.json` | Multi-turn conversations: "what if ... instead?", "now find the length", "explain", and sad paths (follow-up with no context, after `/reset`, non-math) |
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
