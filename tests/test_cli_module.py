import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "backend.cli.main", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_runs_as_a_module_from_the_repo_root():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "migrate" in result.stdout


def test_cli_offers_only_the_evidence_engine_commands():
    result = run_cli("--help")

    assert result.returncode == 0
    # `evaluate` drove the removed pipeline; benchmarks run as `python -m evaluation.run_*`.
    assert "evaluate" not in result.stdout
    assert run_cli("evaluate", "--help").returncode == 2  # no such command
    for command in ("check", "migrate", "worker"):
        assert command in result.stdout
