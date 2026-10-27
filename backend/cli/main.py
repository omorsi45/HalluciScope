import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from backend.config import Settings
from backend.core.document import parse_document
from backend.core.ensemble import ConfidenceTier

app = typer.Typer(name="halluciscope", help="Multi-signal hallucination detection for RAG QA")
console = Console()


def _build_pipeline():
    """Build the full pipeline with real dependencies. Mocked in tests."""
    from backend.core.chunker import Chunker
    from backend.core.ensemble import EnsembleScorer
    from backend.core.pipeline import Pipeline
    from backend.core.verifiers.consistency import ConsistencyVerifier
    from backend.core.verifiers.nli import NLIVerifier
    from backend.core.verifiers.similarity import SimilarityVerifier
    from backend.models.loader import get_embedding_model, get_nli_model

    settings = Settings()
    embedding_model = get_embedding_model()
    tokenizer, nli_model = get_nli_model()

    return Pipeline(
        settings=settings,
        chunker=Chunker(
            embedding_model=embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k_chunks,
        ),
        nli_verifier=NLIVerifier(tokenizer=tokenizer, model=nli_model),
        similarity_verifier=SimilarityVerifier(embedding_model=embedding_model),
        consistency_verifier=ConsistencyVerifier(
            embedding_model=embedding_model,
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            n_samples=settings.consistency_samples,
            temperature=settings.consistency_temperature,
            similarity_threshold=settings.consistency_similarity_threshold,
        ),
        ensemble=EnsembleScorer(
            nli_weight=settings.nli_weight,
            consistency_weight=settings.consistency_weight,
            similarity_weight=settings.similarity_weight,
        ),
    )


TIER_COLORS = {
    ConfidenceTier.SUPPORTED: "green",
    ConfidenceTier.UNCERTAIN: "yellow",
    ConfidenceTier.HALLUCINATED: "red",
}


@app.command()
def check(
    doc: Path = typer.Option(..., help="Path to document (PDF or TXT)"),
    question: str = typer.Option(..., help="Question to ask"),
    verbose: bool = typer.Option(False, help="Show per-claim verifier breakdown"),
):
    """Analyze a document for hallucinations."""
    pipeline = _build_pipeline()
    document_text = parse_document(file_path=doc)

    result = asyncio.run(pipeline.analyze(document_text=document_text, question=question))

    console.print(f"\n[bold]Question:[/bold] {result.question}")
    console.print(f"[bold]Answer:[/bold] {result.answer}")
    console.print(f"[bold]Overall hallucination score:[/bold] {result.overall_score:.2f}\n")

    table = Table(title="Claim Analysis")
    table.add_column("Claim", style="white", max_width=60)
    table.add_column("Score", justify="center")
    table.add_column("Tier", justify="center")

    if verbose:
        table.add_column("NLI", justify="center")
        table.add_column("Consistency", justify="center")
        table.add_column("Similarity", justify="center")

    for sc in result.scored_claims:
        color = TIER_COLORS[sc.tier]
        row = [
            sc.claim,
            f"{sc.hallucination_score:.2f}",
            Text(sc.tier.value, style=color),
        ]
        if verbose:
            nli_details = sc.verifier_details.get("nli", {})
            cons_details = sc.verifier_details.get("consistency", {})
            sim_details = sc.verifier_details.get("similarity", {})
            row.append(
                f"e:{nli_details.get('entailment', 0):.2f} "
                f"n:{nli_details.get('neutral', 0):.2f} "
                f"c:{nli_details.get('contradiction', 0):.2f}"
            )
            row.append(
                f"{cons_details.get('appearances', '?')}/{cons_details.get('n_samples', '?')}"
            )
            row.append(f"{sim_details.get('max_similarity', 0):.2f}")
        table.add_row(*row)

    console.print(table)


@app.command()
def migrate():
    """Apply pending SQL migrations to the Postgres database."""
    from backend.db.migrate import migrate as run_migrations

    applied = asyncio.run(run_migrations(Settings().migration_database_url))
    console.print(f"applied: {applied}" if applied else "up to date")


async def _run_worker(settings: Settings) -> None:
    import signal

    from backend.db.pool import Database
    from backend.jobs.registry import build_handlers
    from backend.jobs.worker import Worker, default_worker_id
    from backend.obs.logging import configure_logging
    from backend.obs.metrics_server import start_metrics_server

    configure_logging(settings.log_level)
    # Each slot can hold a handler, a job heartbeat and its housekeeping at once.
    db = Database(settings.database_url, max_size=max(10, 4 * settings.worker_concurrency))
    await db.open()
    metrics_server = start_metrics_server(
        "127.0.0.1", settings.worker_metrics_port, settings.metrics_token
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    signals = (signal.SIGTERM, signal.SIGINT)
    for sig in signals:
        loop.add_signal_handler(sig, stop.set)
    try:
        services = None  # Plan B introduces the services container
        base_id = default_worker_id()
        workers = [
            Worker(db, build_handlers(services), services=services, worker_id=f"{base_id}:{slot}")
            for slot in range(settings.worker_concurrency)
        ]
        await asyncio.gather(*(worker.run(stop) for worker in workers))
    finally:
        for sig in signals:
            loop.remove_signal_handler(sig)
        metrics_server.shutdown()
        metrics_server.server_close()
        await db.close()


@app.command()
def worker():
    """Run the background job worker until SIGTERM or SIGINT."""
    asyncio.run(_run_worker(Settings()))


@app.command()
def evaluate(
    dataset: str = typer.Option(..., help="Dataset name (halueval, custom, adversarial)"),
    output: Path = typer.Option(Path("evaluation/results"), help="Output directory"),
    ablation: bool = typer.Option(False, help="Run ablation study"),
):
    """Run evaluation benchmark on a dataset."""
    if ablation:
        from evaluation.ablation import run_ablation
        asyncio.run(run_ablation(dataset=dataset, output_dir=str(output)))
    else:
        from evaluation.benchmarks import run_benchmark
        asyncio.run(run_benchmark(dataset=dataset, output_dir=str(output)))
