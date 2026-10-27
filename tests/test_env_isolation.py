import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.pg
def test_tests_ignore_halluciscope_variables_from_the_developer_environment():
    env = {
        **os.environ,
        "HALLUCISCOPE_ALLOW_REGISTRATION": "false",
        "HALLUCISCOPE_METRICS_TOKEN": "x",
    }

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_auth_routes.py", "-q", "-k",
         "register_sets_session"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
