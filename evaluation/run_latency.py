"""End-to-end latency of ask analyses (spec section 20.4).

    python -m evaluation.run_latency --requests 50
    python -m evaluation.report latency evaluation/results/latency.json

The corpus is version 1 of the first ten families of the synthetic temporal corpus
(``evaluation.synthetic_temporal``), uploaded through ``intake_upload`` and ``process_upload``
into a bench tenant that is new for every run. Each request is an ask analysis created with
``create_analysis_request(enqueue=False)`` and run in this process with ``run_analysis``, one
after another, with the default options: the same stages and the same claim extraction through
the LLM as in the worker. The questions are fixed by the corpus (``build_questions``).

The latency of a stage is its entry in ``analyses.timings``, which ``run_analysis`` measures
itself, so the numbers are the ones a user's analysis records. Percentiles use the nearest-rank
method. A request that fails is counted under its error code and left out of every percentile.
A real run uses ``llama3.2:3b`` through Ollama, the real NLI, embedder and reranker, and the
trained calibrator; ``--fake`` swaps in the test fakes to smoke-test the runner, never to
produce results.
"""

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from backend.analysis.service import STAGES, create_analysis_request, run_analysis
from backend.calibration.artifact import Calibrator
from backend.config import Settings
from backend.db.pool import Database
from backend.db.repos import analyses
from backend.jobs.worker import PermanentJobError
from backend.llm.client import LLMClient, OllamaClient
from backend.llm.prompts import EXTRACT_SYSTEM
from backend.services import Services
from evaluation.extract import load_models
from evaluation.run_retrieval import DEFAULT_DATABASE_URL, collect_env, percentile
from evaluation.run_temporal import (
    BENCH_KINDS,
    _log,
    _make_tenant,
    _positive_int,
    _Tenant,
    _upload,
)
from evaluation.synthetic_temporal import FamilySpec, generate

FAMILIES = 10  # the documents of the benchmark: version 1 of each
STAGE_KEYS = (*STAGES, "total")
PROGRESS_EVERY = 10

_PASSAGE = re.compile(r"<passage>(.*?)</passage>", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# --- statistics --------------------------------------------------------------------------------


def summarize(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """(p50, p95 and mean seconds of each stage and of the total, failures by error code).

    ``rows`` are the finished analyses. Only the ones that succeeded contribute to the stage
    statistics; every other one is a failure, counted under its error code (or its status,
    for an analysis that is not final, which would be a bug). With no success there are no
    stages.
    """
    done = [row["timings"] for row in rows if row["status"] == "succeeded"]
    failures = Counter(
        row["error_code"] or row["status"] for row in rows if row["status"] != "succeeded"
    )
    stages: dict[str, dict[str, float]] = {}
    if done:
        for stage in STAGE_KEYS:
            values = [timings[stage] for timings in done]
            stages[stage] = {
                "p50_s": percentile(values, 50),
                "p95_s": percentile(values, 95),
                "mean_s": sum(values) / len(values),
            }
    return stages, dict(sorted(failures.items()))


# --- questions ---------------------------------------------------------------------------------


def build_questions(families: Sequence[FamilySpec], n: int) -> list[str]:
    """``n`` questions "What is the <fact phrase> for <entity>?", one per fact of version 1.

    The fact phrase is the fact key with spaces. The questions go round the families (the
    first fact of each, then the second of each, in key order) and start over when ``n``
    exceeds the number of facts.
    """
    per_family = [
        [
            f"What is the {key.replace('_', ' ')} for {family.entity}?"
            for key, value in sorted(family.versions[0].items())
            if value is not None
        ]
        for family in families
    ]
    pool = [
        question
        for questions in zip_longest(*per_family)
        for question in questions
        if question is not None
    ]
    return [pool[index % len(pool)] for index in range(n)]


# --- the fake LLM ------------------------------------------------------------------------------


def fake_reply(messages: Sequence[Mapping[str, str]]) -> str:
    """The reply of the scripted LLM of ``--fake``, fixed by the messages alone.

    Claim extraction gets valid JSON without claims, so every sentence stands as its own claim.
    An answer is the first sentence of the first context passage, cited as ``[P1]``.
    """
    if messages[0]["content"] == EXTRACT_SYSTEM:
        return json.dumps({"claims": []})
    passage = " ".join(_PASSAGE.findall(messages[-1]["content"])[0].split())
    return f"{_SENTENCE_END.split(passage)[0]} [P1]"


# --- the run -----------------------------------------------------------------------------------


async def ingest_corpus(services: Services, families: Sequence[FamilySpec], name: str) -> _Tenant:
    """A new bench tenant holding version 1 of each family, ingested in this process."""
    tenant = await _make_tenant(services.db, name)
    for family in families:
        await _upload(services, tenant, family, 0)
    return tenant


async def measure(
    services: Services, tenant: _Tenant, questions: Sequence[str]
) -> list[dict[str, Any]]:
    """Ask each question as an analysis, one after another; the finished analysis rows."""
    rows: list[dict[str, Any]] = []
    for number, question in enumerate(questions, start=1):
        analysis_id = await create_analysis_request(
            services.db,
            tenant.principal,
            kind="ask",
            question=question,
            answer=None,
            collection_ids=[tenant.collection_id],
            as_of=None,
            options={},
            settings=services.settings,
            collection_kinds=BENCH_KINDS,
            enqueue=False,
        )
        error_code: str | None = None
        try:
            await run_analysis(services, tenant.tenant_id, analysis_id)
        except PermanentJobError as exc:
            error_code = exc.code
        async with services.db.tenant_tx(tenant.tenant_id) as conn:
            row = await analyses.get_analysis(conn, analysis_id)
        if row is None or (error_code is not None and row["error_code"] != error_code):
            raise RuntimeError("a failed analysis does not carry the error code it failed with")
        rows.append(row)
        if number % PROGRESS_EVERY == 0:
            _log(f"requests done: {number} of {len(questions)}")
    return rows


def _fake_services(settings: Settings, db: Database) -> Services:
    # The deterministic stand-ins of the test suite: for smoke runs, never for results.
    from tests.fakes import ScriptedLLM, make_test_calibrator

    models = load_models("fake", "fake", settings)
    llm = ScriptedLLM()
    llm.default = fake_reply
    return Services(
        settings,
        db,
        models.embedder,
        models.reranker,
        llm=llm,
        nli=models.nli,
        calibrator=make_test_calibrator(),
    )


def _real_services(settings: Settings, db: Database, http: httpx.AsyncClient) -> Services:
    models = load_models("real", "real", settings)
    llm = OllamaClient(
        http, settings.ollama_base_url, settings.llm_model, timeout_s=settings.llm_timeout_s
    )
    return Services(
        settings,
        db,
        models.embedder,
        models.reranker,
        llm=llm,
        nli=models.nli,
        calibrator=Calibrator.load(settings.calibrator_path),
    )


def _model_ids(services: Services) -> dict[str, str]:
    llm: LLMClient | None = services.llm
    if llm is None or services.nli is None or services.reranker is None:
        raise RuntimeError("the benchmark services lack a model")
    return {
        "llm": llm.model_id,
        "nli": services.nli.model_id,
        "embedder": services.embedder.model_id,
        "reranker": services.reranker.model_id,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the benchmark on the database, write the results, return them."""
    settings = Settings()
    families = generate(n_families=FAMILIES).families
    questions = build_questions(families, args.requests)
    db = Database(args.database_url, max_size=4)
    await db.open()
    try:
        async with httpx.AsyncClient() as http:
            services = (
                _fake_services(settings, db) if args.fake else _real_services(settings, db, http)
            )
            _log("uploading the corpus")
            tenant = await ingest_corpus(services, families, f"latency-{uuid4().hex[:8]}")
            rows = await measure(services, tenant, questions)
            models = _model_ids(services)
        env = await collect_env(db)
    finally:
        await db.close()

    stages, failures = summarize(rows)
    result: dict[str, Any] = {
        "n": len(rows),
        "stages": stages,
        "models": models,
        "env": env,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


# --- command line ------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_latency")
    parser.add_argument("--requests", type=_positive_int, default=50)
    parser.add_argument(
        "--database-url",
        help="default: $HALLUCISCOPE_BENCH_DATABASE_URL, else " + DEFAULT_DATABASE_URL,
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="fake models, scripted LLM and test calibrator: a smoke run, never results",
    )
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/latency.json"))
    args = parser.parse_args(argv)
    if args.database_url is None:
        args.database_url = os.environ.get("HALLUCISCOPE_BENCH_DATABASE_URL", DEFAULT_DATABASE_URL)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run(args))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
