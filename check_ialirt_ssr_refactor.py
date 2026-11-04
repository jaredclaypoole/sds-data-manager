"""Run isolated pre-commit checks and the I-ALiRT refactoring test suite.

Usage: poetry run python check_ialirt_ssr_refactor.py
"""

# ruff: noqa: S603

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter


class Status:
    """Print timestamps, total runtime, and time since the previous status."""

    def __init__(self):
        """Start the elapsed-time clock."""
        self.started = self.previous = perf_counter()

    def __call__(self, message):
        """Print and flush one status line."""
        now = perf_counter()
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(
            f"[{timestamp}] [elapsed {now - self.started:.2f}s; "
            f"+{now - self.previous:.2f}s] {message}",
            flush=True,
        )
        self.previous = now


def run(command, cwd, env, *, capture=False):
    """Run a command, preserving failure codes and optionally capturing stdout."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE if capture else None,
    )


def regular_test_paths(root: Path, *, ssr_only=False):
    """List SSR-only or all regular I-ALiRT tests relative to the repo root."""
    pattern = "test_ialirt_ssr*.py" if ssr_only else "test_ialirt*.py"
    tests = sorted(root.glob(f"tests/lambda_endpoints/{pattern}"))
    tests += sorted(root.glob(f"tests/infrastructure/{pattern}"))
    if ssr_only:
        return [str(path.relative_to(root)) for path in tests]
    return [str(path.relative_to(root)) for path in tests] + [
        "tests/infrastructure/test_api_key_dynamodb.py",
        "tests/lambda_endpoints/test_dependency_api.py",
    ]


def parse_args():
    """Parse test-scope options and the option to skip isolated pre-commit checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--no-regular-tests",
        action="store_true",
        help="Skip regular I-ALiRT tests; retain pre-commit and refactoring harnesses.",
    )
    scope.add_argument(
        "--ssr-only",
        action="store_true",
        help="Run only SSR tests and refactoring harnesses, retaining pre-commit.",
    )
    parser.add_argument(
        "--no-precommit",
        action="store_true",
        help="Skip pre-commit and its temporary working-tree copy.",
    )
    return parser.parse_args()


def selected_test_paths(root, args, status):
    """Report the selected test scope and return its regular test paths."""
    tests = []
    if args.no_regular_tests:
        status("Skipping regular I-ALiRT tests; running refactoring harnesses")
    elif args.ssr_only:
        status("Running SSR-specific tests and both refactoring harnesses")
        tests = regular_test_paths(root, ssr_only=True)
    else:
        status("Running all I-ALiRT tests and both refactoring harnesses")
        tests = regular_test_paths(root)
    return tests


def main():
    """Run checks against current files, keeping hook fixes in a temporary copy."""
    args = parse_args()
    status = Status()
    root = Path(__file__).resolve().parent
    env = {**os.environ, "SSR_UPDATE_TRACES": "0"}
    status("Preparing refactoring checks")
    try:
        with tempfile.TemporaryDirectory(prefix="ialirt-refactor-") as directory:
            temporary = Path(directory)
            env["COVERAGE_FILE"] = str(temporary / ".coverage")
            try:
                if args.no_precommit:
                    status("Skipping pre-commit")
                else:
                    copy = temporary / "repo"
                    status("Creating temporary Git repository")
                    run(
                        [
                            "git",
                            "clone",
                            "--quiet",
                            "--shared",
                            "--no-checkout",
                            "--",
                            str(root),
                            str(copy),
                        ],
                        root,
                        env,
                    )
                    run(["git", "read-tree", "HEAD"], copy, env)
                    status("Created temporary Git repository")

                    status("Copying current working-tree files")
                    paths = run(
                        [
                            "git",
                            "ls-files",
                            "--cached",
                            "--others",
                            "--exclude-standard",
                            "-z",
                        ],
                        root,
                        env,
                        capture=True,
                    ).stdout
                    for name in set(paths.split(b"\0")) - {b""}:
                        relative = Path(os.fsdecode(name))
                        source = root / relative
                        if not source.exists() and not source.is_symlink():
                            continue
                        target = copy / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target, follow_symlinks=False)
                    status("Copied current working-tree files")

                    status("Running pre-commit in temporary copy")
                    run(["git", "add", "--all"], copy, env)
                    run(
                        [
                            sys.executable,
                            "-m",
                            "pre_commit",
                            "run",
                            "--all-files",
                            "--show-diff-on-failure",
                        ],
                        copy,
                        env,
                    )
                    status("Pre-commit passed")

                status("Running behavior-aware harness with 100% coverage required")
                run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/lambda_endpoints/ssr_refactor/behavior_aware/"
                        "test_ialirt_ssr_contract.py",
                        "--cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing",
                        "--cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils",
                        "--cov-branch",
                        "--cov-report=term-missing",
                        "--cov-fail-under=100",
                    ],
                    root,
                    env,
                )
                status("Behavior-aware harness and coverage passed")

                tests = selected_test_paths(root, args, status)
                run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        *tests,
                        "tests/lambda_endpoints/ssr_refactor",
                    ],
                    root,
                    env,
                )
                status("All refactoring checks passed")
            finally:
                status("Cleaning up managed temporary directory")
    except subprocess.CalledProcessError as error:
        status(f"Checks failed (exit {error.returncode})")
        return error.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
