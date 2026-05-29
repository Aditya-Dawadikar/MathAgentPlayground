import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TEST_MODULE_PATH = PROJECT_ROOT / "tests" / "test_calculator_agent.py"


def load_test_module():
    spec = importlib.util.spec_from_file_location("test_calculator_agent", TEST_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def evaluate_case(test_module, agent, case):
    result = await agent.ainvoke(
        {"messages": [test_module.HumanMessage(content=case["prompt"])]}
    )
    expected_tool_calls = [
        {
            "name": case["expected_tool"],
            "args": case["expected_arguments"],
        }
    ]
    actual_tool_calls = test_module.normalize_tool_calls(
        test_module.extract_tool_calls(result),
        expected_tool_calls,
    )
    eval_result = test_module.tool_eval(
        outputs=actual_tool_calls,
        reference_outputs=expected_tool_calls,
    )
    passed = eval_result[0]["score"] == 1

    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "passed": passed,
        "expected": expected_tool_calls,
        "actual": actual_tool_calls,
        "eval_result": eval_result,
    }


async def run_experiment(runs, agent_module_name=None):
    test_module = load_test_module()
    cases = test_module.TEST_CASES
    resolved_agent_module_name = agent_module_name or test_module.resolve_agent_module_name()
    agent_builder = test_module.load_build_agent(resolved_agent_module_name)
    case_counts = {
        case["id"]: {"passes": 0, "failures": 0, "last_failure": None}
        for case in cases
    }
    run_summaries = []

    for run_index in range(1, runs + 1):
        agent = agent_builder(with_memory=False)
        run_results = []

        for case in cases:
            outcome = await evaluate_case(test_module, agent, case)
            run_results.append(outcome)

            if outcome["passed"]:
                case_counts[case["id"]]["passes"] += 1
            else:
                case_counts[case["id"]]["failures"] += 1
                case_counts[case["id"]]["last_failure"] = {
                    "expected": outcome["expected"],
                    "actual": outcome["actual"],
                    "eval_result": outcome["eval_result"],
                }

        run_summaries.append(
            {
                "run": run_index,
                "passed": sum(1 for item in run_results if item["passed"]),
                "failed": sum(1 for item in run_results if not item["passed"]),
            }
        )

    return cases, case_counts, run_summaries, resolved_agent_module_name


def print_summary(cases, case_counts, run_summaries, runs, agent_module_name):
    print(
        f"Tool selection experiment finished: {runs} runs, {len(cases)} cases per run, agent={agent_module_name}"
    )
    print()
    print("Per-run summary")
    for summary in run_summaries:
        print(
            f"- Run {summary['run']}: {summary['passed']} passed, {summary['failed']} failed"
        )

    print()
    print("Failure counts by test case")
    sorted_case_ids = sorted(
        case_counts,
        key=lambda case_id: (-case_counts[case_id]["failures"], case_id),
    )

    for case_id in sorted_case_ids:
        counts = case_counts[case_id]
        print(
            f"- {case_id}: {counts['failures']} failures, {counts['passes']} passes"
        )
        # if counts["last_failure"] is not None:
        #     print(f"  expected: {counts['last_failure']['expected']}")
        #     print(f"  actual:   {counts['last_failure']['actual']}")
        #     print(f"  eval:     {counts['last_failure']['eval_result']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the tool-selection dataset repeatedly and count failures per case."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run the full dataset.",
    )
    parser.add_argument(
        "--agent-module",
        default=None,
        help="Python module path for the agent under test, for example app.agent or app.agent_v2.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    cases, case_counts, run_summaries, agent_module_name = await run_experiment(
        args.runs,
        args.agent_module,
    )
    print_summary(cases, case_counts, run_summaries, args.runs, agent_module_name)


if __name__ == "__main__":
    asyncio.run(main())