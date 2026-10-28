import asyncio
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from backend.auth.service import Principal
from backend.config import Settings
from backend.services import CalibratorRejected, build_services

logger = logging.getLogger(__name__)

app = typer.Typer(name="halluciscope", help="Multi-signal hallucination detection for RAG QA")
console = Console()

# The user `check` runs as. It is created disabled and with a password hash that no password
# verifies against, so it can never log in.
CLI_EMAIL = "cli@localhost"
CLI_TENANT_NAME = "cli"
_LOCKED_PASSWORD_HASH = "!"


VERDICT_COLORS = {"supported": "green", "insufficient": "yellow", "contradicted": "red"}


async def _cli_principal(db: Any) -> Principal:
    """The principal `check` runs as: owner of the `cli` tenant, created on first use.

    Its `system` auth is never produced by HTTP authentication, and the user is disabled with
    no usable password, so nothing outside this command can act as it.
    """
    from backend.db.repos import tenants, users

    async with db.system_tx() as conn:
        user = await users.get_user_by_email(conn, CLI_EMAIL)
        if user is None:
            user = await users.insert_user(conn, CLI_EMAIL, _LOCKED_PASSWORD_HASH, "CLI")
            await conn.execute("update users set disabled_at = now() where id = %s", (user["id"],))
            tenant = await tenants.insert_tenant(conn, CLI_TENANT_NAME)
            await tenants.add_membership(conn, tenant["id"], user["id"], "owner")
        membership = (await tenants.list_memberships(conn, user["id"]))[0]
    return Principal(
        user_id=user["id"],
        email=user["email"],
        display_name=user["display_name"],
        tenant_id=membership["tenant_id"],
        role=membership["role"],
        auth="system",
    )


async def _top_evidence(
    conn: Any, evidence: list[dict[str, Any]]
) -> dict[UUID, tuple[dict[str, Any], str]]:
    """The best ranked evidence item of each claim, with the text of the span it rests on."""
    top: dict[UUID, dict[str, Any]] = {}
    for item in evidence:  # by claim, then rank
        top.setdefault(item["claim_id"], item)
    shown = {}
    for claim_id, item in top.items():
        cur = await conn.execute(
            "select substr(text, %s, %s) as span from document_versions where id = %s",
            (item["span_start"] + 1, item["span_end"] - item["span_start"], item["version_id"]),
        )
        shown[claim_id] = (item, (await cur.fetchone())["span"])
    return shown


def _evidence_cell(shown: tuple[dict[str, Any], str] | None) -> Text:
    if shown is None:
        return Text("-")
    item, span = shown
    where = f"{item['document_title']} v{item['version_number']}"
    if item["page_start"] is not None:
        where += f" p.{item['page_start']}"
    return Text(f"{where}\n{span}")


async def _print_result(db: Any, tenant_id: UUID, analysis_id: UUID, verbose: bool) -> None:
    from backend.db.repos import analyses

    async with db.tenant_tx(tenant_id) as conn:
        analysis = await analyses.get_analysis(conn, analysis_id)
        claims = await analyses.get_claims(conn, analysis_id)
        evidence = await analyses.get_evidence(conn, [claim["id"] for claim in claims])
        top = await _top_evidence(conn, evidence) if verbose else {}

    # Model output is untrusted text: Text objects keep Rich from reading it as markup.
    console.print()
    console.print(Text.assemble(("Question: ", "bold"), analysis["question"]))
    console.print(Text.assemble(("Answer: ", "bold"), analysis["answer"]))
    console.print()
    table = Table(title="Claim Analysis")
    table.add_column("Claim", style="white", max_width=60)
    table.add_column("Verdict", justify="center")
    table.add_column("P(supported)", justify="center")
    if verbose:
        table.add_column("Evidence", max_width=60)
    for claim in claims:
        row = [
            Text(claim["text"]),
            Text(claim["verdict"], style=VERDICT_COLORS[claim["verdict"]]),
            Text(f"{claim['p_supported']:.2f}"),
        ]
        if verbose:
            row.append(_evidence_cell(top.get(claim["id"])))
        table.add_row(*row)
    console.print(table)


async def _check(
    settings: Settings, filename: str, data: bytes, question: str, verbose: bool
) -> str | None:
    """Run a quick analysis of the document in this process and print it.

    Returns the error code when the check fails, else None.
    """
    import httpx

    from backend.analysis.quick import create_quick_analysis
    from backend.analysis.service import run_analysis
    from backend.api.errors import ApiError
    from backend.db.pool import Database
    from backend.jobs.worker import PermanentJobError

    db = Database(settings.database_url)
    await db.open()
    try:
        async with httpx.AsyncClient() as http:
            # The models load first, so a refused artifact stores nothing.
            try:
                services = await build_services(settings, db, http)
            except CalibratorRejected as exc:
                raise _reject_calibrator(exc.field) from None
            principal = await _cli_principal(db)
            try:
                analysis_id = await create_quick_analysis(
                    db,
                    principal,
                    data=data,
                    filename=filename,
                    question=question,
                    settings=settings,
                    enqueue=False,
                )
            except ApiError as exc:
                return exc.code
            try:
                await run_analysis(services, principal.tenant_id, analysis_id)
            except PermanentJobError as exc:
                # run_analysis has already marked the analysis failed with this code.
                return exc.code
        await _print_result(db, principal.tenant_id, analysis_id, verbose)
        return None
    finally:
        await db.close()


@app.command()
def check(
    doc: Path = typer.Option(
        ..., exists=True, dir_okay=False, readable=True, help="Path to the document (PDF, TXT, ...)"
    ),
    question: str = typer.Option(..., help="Question to ask"),
    verbose: bool = typer.Option(False, help="Show the top evidence of each claim"),
):
    """Answer a question from a document and check every claim of the answer against it."""
    error_code = asyncio.run(_check(Settings(), doc.name, doc.read_bytes(), question, verbose))
    if error_code is not None:
        typer.echo(f"error: {error_code}", err=True)
        raise typer.Exit(1)


@app.command()
def migrate():
    """Apply pending SQL migrations to the Postgres database."""
    from backend.db.migrate import migrate as run_migrations

    applied = asyncio.run(run_migrations(Settings().migration_database_url))
    console.print(f"applied: {applied}" if applied else "up to date")


def _reject_calibrator(field: str) -> typer.Exit:
    """Log which field made the calibrator artifact unusable, and stop the command.

    Only the field name is logged. The exception message can quote artifact content.
    """
    logger.error("calibrator artifact rejected: %s", field)
    return typer.Exit(1)


async def _run_worker(settings: Settings) -> None:
    import signal

    import httpx

    from backend.db.pool import Database
    from backend.jobs.registry import build_failure_handlers, build_handlers
    from backend.jobs.worker import Worker, default_worker_id
    from backend.obs.logging import configure_logging
    from backend.obs.metrics_server import start_metrics_server

    configure_logging(settings.log_level)
    # Each slot can hold a handler, a job heartbeat and its housekeeping at once.
    db = Database(settings.database_url, max_size=max(10, 4 * settings.worker_concurrency))
    metrics_server = None
    http_client: httpx.AsyncClient | None = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    try:
        await db.open()
        if settings.metrics_token and settings.metrics_token.strip():
            metrics_server = start_metrics_server(
                "127.0.0.1", settings.worker_metrics_port, settings.metrics_token
            )
        else:
            logger.info("worker metrics disabled")
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
        http_client = httpx.AsyncClient()
        try:
            services = await build_services(settings, db, http_client)
        except CalibratorRejected as exc:
            raise _reject_calibrator(exc.field) from None
        base_id = default_worker_id()
        slots = [
            asyncio.create_task(
                Worker(
                    db,
                    build_handlers(services),
                    failure_handlers=build_failure_handlers(services),
                    services=services,
                    # The analysis enforces its own limit; the job limit is only a backstop.
                    timeouts_s={"run_analysis": settings.analysis_timeout_s + 30},
                    worker_id=f"{base_id}:{slot}",
                ).run(stop)
            )
            for slot in range(settings.worker_concurrency)
        ]
        try:
            await asyncio.wait(slots, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            # A failed slot stops the others; wait for them so no job is cut off mid-flight
            # by the pool closing underneath it.
            stop.set()
            await asyncio.gather(*slots, return_exceptions=True)
        for slot_task in slots:
            slot_task.result()
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)
        if metrics_server is not None:
            metrics_server.shutdown()
            metrics_server.server_close()
        if http_client is not None:
            await http_client.aclose()
        await db.close()


@app.command()
def worker():
    """Run the background job worker until SIGTERM or SIGINT."""
    settings = Settings()
    if settings.worker_concurrency < 1:
        typer.echo("worker_concurrency must be at least 1", err=True)
        raise typer.Exit(2)
    asyncio.run(_run_worker(settings))


if __name__ == "__main__":
    app()
