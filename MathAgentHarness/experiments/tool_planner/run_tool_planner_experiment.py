import argparse
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE_PATH = PROJECT_ROOT / "tests" / "test_planner_agent.py"


def extract_case_id(test_name):
	if "[" not in test_name or not test_name.endswith("]"):
		return test_name

	return test_name[test_name.rfind("[") + 1 : -1]


def build_pytest_command(workers, junit_xml_path, agent_module_name=None):
	command = [
		sys.executable,
		"-m",
		"pytest",
		str(TEST_FILE_PATH),
		"-n",
		str(workers),
		"--junitxml",
		str(junit_xml_path),
		"-q",
	]

	if agent_module_name:
		command.extend(["--agent-module", agent_module_name])

	return command


def parse_junit_results(junit_xml_path):
	root = ET.parse(junit_xml_path).getroot()
	results = []

	for testcase in root.iter("testcase"):
		test_name = testcase.attrib.get("name", "")
		case_id = extract_case_id(test_name)
		failure = testcase.find("failure")
		error = testcase.find("error")
		skipped = testcase.find("skipped")
		passed = failure is None and error is None and skipped is None

		results.append(
			{
				"id": case_id,
				"passed": passed,
				"failure_text": (failure.text if failure is not None else None)
				or (error.text if error is not None else None)
				or (skipped.text if skipped is not None else None),
			}
		)

	return results


def run_once(run_index, workers, agent_module_name=None):
	with tempfile.TemporaryDirectory() as temp_dir:
		junit_xml_path = Path(temp_dir) / f"planner-run-{run_index}.xml"
		command = build_pytest_command(workers, junit_xml_path, agent_module_name)
		print(f"[run {run_index}] starting: {' '.join(command)}", flush=True)
		start_time = time.perf_counter()
		completed = subprocess.run(
			command,
			cwd=PROJECT_ROOT,
			capture_output=True,
			text=True,
		)
		elapsed_seconds = time.perf_counter() - start_time
		print(
			f"[run {run_index}] finished with exit code {completed.returncode} in {elapsed_seconds:.1f}s",
			flush=True,
		)

		if completed.returncode not in (0, 1):
			stderr = completed.stderr.strip()
			stdout = completed.stdout.strip()

			if "unrecognized arguments: -n" in stderr or "No module named xdist" in stderr:
				raise RuntimeError(
					"pytest-xdist is required for parallel runs. Install it before running this experiment."
				)

			raise RuntimeError(
				f"Pytest run {run_index} failed to execute.\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
			)

		results = parse_junit_results(junit_xml_path)

		return {
			"run": run_index,
			"results": results,
			"passed": sum(1 for item in results if item["passed"]),
			"failed": sum(1 for item in results if not item["passed"]),
			"stdout": completed.stdout,
			"stderr": completed.stderr,
		}


def run_experiment(runs, workers, agent_module_name=None):
	run_summaries = []
	case_counts = {}
	print(
		f"Starting planner experiment: runs={runs}, workers={workers}, agent={agent_module_name or 'default test configuration'}",
		flush=True,
	)

	for run_index in range(1, runs + 1):
		run_summary = run_once(run_index, workers, agent_module_name)
		run_summaries.append(run_summary)
		print(
			f"[run {run_index}] summary: {run_summary['passed']} passed, {run_summary['failed']} failed",
			flush=True,
		)

		for outcome in run_summary["results"]:
			counts = case_counts.setdefault(
				outcome["id"],
				{"passes": 0, "failures": 0, "last_failure": None},
			)

			if outcome["passed"]:
				counts["passes"] += 1
			else:
				counts["failures"] += 1
				counts["last_failure"] = outcome["failure_text"]

	return case_counts, run_summaries


def print_summary(case_counts, run_summaries, runs, workers, agent_module_name):
	resolved_agent_module = agent_module_name or "default test configuration"
	print(
		f"Tool planner experiment finished: {runs} runs, workers={workers}, agent={resolved_agent_module}"
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
		print(f"- {case_id}: {counts['failures']} failures, {counts['passes']} passes")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run the planner test suite repeatedly with parallel pytest workers and count failures per case."
	)
	parser.add_argument(
		"--runs",
		type=int,
		default=5,
		help="Number of times to run the full planner test suite.",
	)
	parser.add_argument(
		"--workers",
		default="auto",
		help="Value passed to pytest -n, for example auto, 2, or 4.",
	)
	parser.add_argument(
		"--agent-module",
		default=None,
		help="Python module path for the agent under test, for example app.planner_agent_v1.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	case_counts, run_summaries = run_experiment(
		runs=args.runs,
		workers=args.workers,
		agent_module_name=args.agent_module,
	)
	print_summary(
		case_counts=case_counts,
		run_summaries=run_summaries,
		runs=args.runs,
		workers=args.workers,
		agent_module_name=args.agent_module,
	)


if __name__ == "__main__":
	main()
