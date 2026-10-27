import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_runs_as_a_module_from_the_repo_root():
    result = subprocess.run(
        [sys.executable, "-m", "backend.cli.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "migrate" in result.stdout
