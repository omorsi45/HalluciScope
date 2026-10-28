"""`halluciscope check`: a quick analysis run in-process against the test database, with fakes.

`build_services` loads real models, so every test replaces it. The command is invoked in a
worker thread: it starts its own event loop, which cannot nest inside the test's.
"""

import asyncio
import logging
import re
from uuid import UUID

import pytest
from rich.console import Console
from typer.testing import CliRunner

from backend.cli.main import app
from backend.llm.client import LLMUnavailable
from backend.services import CalibratorRejected, Services
from tests.fakes import (
    HashEmbedder,
    KeywordReranker,
    RuleNLIScorer,
    ScriptedLLM,
    make_test_calibrator,
)
from tests.test_analysis_routes import claims_json
from tests.test_parsers_pdf import make_pdf

pytestmark = pytest.mark.pg

DOCUMENT = "Refunds take 30 days. Exchanges take 14 days."
QUESTION = "How long do refunds take?"
ANSWER = "Refunds take 30 days [P1]. Refunds take 45 days [P1]. Warranty lasts two years [P1]."
VERDICTS = [
    ("Refunds take 30 days.", "supported"),
    ("Refunds take 45 days.", "contradicted"),
    ("Warranty lasts two years.", "insufficient"),
]
CLAIMS = claims_json(*enumerate(text for text, _ in VERDICTS))

runner = CliRunner()


@pytest.fixture
def cli(monkeypatch, pg_urls, db):
    """Point the command at the test database and keep Rich from wrapping the tables.

    Returns the `db` fixture so a test can read what the command stored; its teardown also
    empties every table the command filled.
    """
    monkeypatch.setenv("HALLUCISCOPE_DATABASE_URL", pg_urls["app"])
    monkeypatch.setattr("backend.cli.main.console", Console(width=200))
    return db


def fake_services(
    monkeypatch, *, responses=(ANSWER, CLAIMS), with_calibrator: bool = True
) -> list[Services]:
    """Replace `build_services` with one wiring the deterministic fakes onto the command's db."""
    built: list[Services] = []

    async def build_services(settings, db, http):
        llm = ScriptedLLM()
        llm.responses = list(responses)
        services = Services(
            settings,
            db,
            HashEmbedder(),
            KeywordReranker(),
            llm=llm,
            nli=RuleNLIScorer(),
            calibrator=make_test_calibrator() if with_calibrator else None,
        )
        built.append(services)
        return services

    monkeypatch.setattr("backend.cli.main.build_services", build_services)
    return built


async def run_cli(*args: str):
    return await asyncio.to_thread(runner.invoke, app, list(args), catch_exceptions=False)


def write_document(tmp_path, name: str = "handbook.txt", text: str = DOCUMENT):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


async def global_rows(db, query: str) -> list[dict]:
    """Rows of a table without row level security (users, tenants, memberships, jobs)."""
    async with db.system_tx() as conn:
        cur = await conn.execute(query)
        return await cur.fetchall()


async def cli_rows(db, query: str) -> list[dict]:
    """Rows of a tenant table, read as the tenant the command created."""
    (tenant,) = await global_rows(db, "select id from tenants where name = 'cli'")
    tenant_id: UUID = tenant["id"]
    async with db.tenant_tx(tenant_id) as conn:
        cur = await conn.execute(query)
        return await cur.fetchall()


async def test_cli_check_prints_verdicts(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch)
    doc = write_document(tmp_path)

    result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 0, result.output
    output = result.output
    assert f"Question: {QUESTION}" in output
    assert f"Answer: {ANSWER}" in output
    for header in ("Claim", "Verdict", "P(supported)"):
        assert header in output
    stored = await cli_rows(cli, "select text, verdict, p_supported from claims order by ordinal")
    assert [(row["text"], row["verdict"]) for row in stored] == VERDICTS
    for row in stored:
        # The row reads: claim, verdict, then P(supported) to two decimals as stored.
        expected = rf"{re.escape(row['text'])}\W+{row['verdict']}\W+{row['p_supported']:.2f}\b"
        assert re.search(expected, output), output
    (analysis,) = await cli_rows(cli, "select kind, status from analyses")
    assert (analysis["kind"], analysis["status"]) == ("quick", "succeeded")
    # Run in-process: no job was queued for the analysis or the upload.
    assert await global_rows(cli, "select id from jobs") == []


async def test_cli_check_verbose_shows_the_top_evidence(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch)
    doc = tmp_path / "handbook.pdf"
    doc.write_bytes(make_pdf([DOCUMENT]))

    plain = await run_cli("check", "--doc", str(doc), "--question", QUESTION)
    verbose = await run_cli("check", "--doc", str(doc), "--question", QUESTION, "--verbose")

    assert plain.exit_code == 0, plain.output
    assert verbose.exit_code == 0, verbose.output
    assert "handbook" not in plain.output
    # Document title, version, page, then the text of the span the verdict rests on.
    assert re.search(r"handbook v1 p\.1\W+Refunds take 30 days\.", verbose.output), verbose.output


async def test_cli_check_prints_text_that_looks_like_markup(cli, monkeypatch, tmp_path):
    answer = "Refunds take 30 days [/oops] [P1]."
    claims = claims_json((0, "Refunds take 30 days. [/oops]"))
    fake_services(monkeypatch, responses=(answer, claims))
    doc = write_document(tmp_path)

    result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 0, result.output
    assert "[/oops]" in result.output


async def test_cli_check_calibrator_missing_exit_1(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch, with_calibrator=False)
    doc = write_document(tmp_path)

    result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 1
    assert "calibrator_missing" in result.output
    assert "Claim" not in result.output
    (analysis,) = await cli_rows(cli, "select status, error_code from analyses")
    assert (analysis["status"], analysis["error_code"]) == ("failed", "calibrator_missing")


async def test_cli_check_exits_1_with_the_code_when_the_llm_is_down(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch, responses=(LLMUnavailable("SECRET-LLM-DETAIL"),))
    doc = write_document(tmp_path)

    result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 1
    assert "llm_unavailable" in result.output
    assert "SECRET-LLM-DETAIL" not in result.output
    (analysis,) = await cli_rows(cli, "select status, error_code from analyses")
    assert (analysis["status"], analysis["error_code"]) == ("failed", "llm_unavailable")


async def test_cli_check_refuses_an_unsupported_file(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch)
    doc = write_document(tmp_path, "tool.exe", "MZ not a document")

    result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 1
    assert "unsupported_type" in result.output
    assert await cli_rows(cli, "select id from collections") == []


async def test_cli_check_reports_a_rejected_calibrator_artifact(
    cli, monkeypatch, tmp_path, caplog
):
    async def build_services(settings, db, http):
        raise CalibratorRejected("nli")

    monkeypatch.setattr("backend.cli.main.build_services", build_services)
    doc = write_document(tmp_path)

    with caplog.at_level(logging.INFO, logger="backend.cli.main"):
        result = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert result.exit_code == 1
    messages = [r.getMessage() for r in caplog.records if r.name == "backend.cli.main"]
    assert "calibrator artifact rejected: nli" in messages
    # Nothing was stored: the services are built before the first write.
    assert await global_rows(cli, "select id from tenants") == []


async def test_cli_check_rejects_a_missing_file(cli, monkeypatch, tmp_path):
    built = fake_services(monkeypatch)

    result = await run_cli("check", "--doc", str(tmp_path / "absent.txt"), "--question", QUESTION)

    assert result.exit_code == 2
    assert built == []


async def test_cli_user_is_disabled_and_reused(cli, monkeypatch, tmp_path):
    fake_services(monkeypatch)
    doc = write_document(tmp_path)

    first = await run_cli("check", "--doc", str(doc), "--question", QUESTION)
    second = await run_cli("check", "--doc", str(doc), "--question", QUESTION)

    assert (first.exit_code, second.exit_code) == (0, 0), (first.output, second.output)
    (user,) = await global_rows(
        cli, "select id, disabled_at from users where email = 'cli@localhost'"
    )
    (tenant,) = await global_rows(cli, "select id, name from tenants")
    (membership,) = await global_rows(cli, "select tenant_id, user_id, role from memberships")
    assert tenant["name"] == "cli"
    assert user["disabled_at"] is not None
    assert (membership["tenant_id"], membership["user_id"], membership["role"]) == (
        tenant["id"],
        user["id"],
        "owner",
    )
    analyses = await cli_rows(cli, "select user_id, status from analyses")
    assert [(row["user_id"], row["status"]) for row in analyses] == [
        (user["id"], "succeeded")
    ] * 2
