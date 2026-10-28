"""The end-to-end ask latency benchmark: statistics, questions, runner and report.

The statistics and the report are checked against hand-computed values. The runner tests use
the fake models, the scripted LLM and the test calibrator, so they assert the shape of the
output and which requests are counted, never how fast anything is.
"""

import json

import pytest

from backend.analysis.service import STAGES
from backend.llm.client import LLMUnavailable
from backend.llm.prompts import EXTRACT_SYSTEM
from evaluation import report, run_latency
from evaluation.synthetic_temporal import generate

pg = pytest.mark.pg

STAGE_KEYS = (*STAGES, "total")


def succeeded(**timings: float) -> dict:
    """An analysis row that finished: every stage of the run, `total` included."""
    return {
        "status": "succeeded",
        "error_code": None,
        "timings": {stage: timings.get(stage, 1.0) for stage in STAGE_KEYS},
    }


def failed(code: str | None, timings: dict | None = None) -> dict:
    return {"status": "failed", "error_code": code, "timings": timings or {}}


# --- statistics --------------------------------------------------------------------------------


def test_nearest_rank_percentiles():
    values = [1, 2, 3, 4, 5]

    assert run_latency.percentile(values, 50) == 3
    assert run_latency.percentile(values, 95) == 5
    # Sorted first, and the result is always an observed value (no interpolation).
    assert run_latency.percentile([0.4, 0.1, 0.3, 0.2], 50) == 0.2


def test_summarize_gives_p50_p95_and_mean_per_stage_and_total():
    rows = [succeeded(retrieving=value, total=value * 10) for value in (0.1, 0.2, 0.3, 0.4)]

    stages, failures = run_latency.summarize(rows)

    assert list(stages) == list(STAGE_KEYS)
    assert failures == {}
    # n = 4: p50 is rank 2, p95 is rank ceil(3.8) = 4.
    assert stages["retrieving"]["p50_s"] == 0.2
    assert stages["retrieving"]["p95_s"] == 0.4
    assert stages["retrieving"]["mean_s"] == pytest.approx(0.25)
    assert stages["total"]["p50_s"] == pytest.approx(2.0)
    assert stages["total"]["p95_s"] == pytest.approx(4.0)
    assert stages["generating"] == {"p50_s": 1.0, "p95_s": 1.0, "mean_s": 1.0}
    assert set(stages["verifying"]) == {"p50_s", "p95_s", "mean_s"}


def test_summarize_counts_failures_by_error_code_and_leaves_them_out_of_the_stages():
    # A failed analysis never carries timings in practice; give these some, to be sure that
    # nothing is read from them.
    slow = {stage: 99.0 for stage in STAGE_KEYS}
    rows = [
        succeeded(retrieving=0.1),
        failed("llm_timeout", slow),
        succeeded(retrieving=0.3),
        failed("retrieval_failed", slow),
        failed("llm_timeout", slow),
    ]

    stages, failures = run_latency.summarize(rows)

    assert failures == {"llm_timeout": 2, "retrieval_failed": 1}
    assert stages["retrieving"] == {"p50_s": 0.1, "p95_s": 0.3, "mean_s": pytest.approx(0.2)}
    assert stages["total"]["p95_s"] == 1.0


def test_summarize_without_a_successful_request_has_no_stages():
    stages, failures = run_latency.summarize([failed("llm_unavailable"), failed("llm_unavailable")])

    assert stages == {}
    assert failures == {"llm_unavailable": 2}


def test_summarize_does_not_pass_off_an_unfinished_analysis_as_a_success():
    unfinished = {"status": "running", "error_code": None, "timings": {}}

    stages, failures = run_latency.summarize([succeeded(), unfinished])

    assert failures == {"running": 1}
    assert stages["total"]["p95_s"] == 1.0


# --- questions ---------------------------------------------------------------------------------


def present_keys(family) -> list[str]:
    return sorted(key for key, value in family.versions[0].items() if value is not None)


def test_questions_follow_the_fact_phrases_round_robin_over_the_families():
    families = generate(n_families=10).families
    pool = sum(len(present_keys(family)) for family in families)

    questions = run_latency.build_questions(families, 12)

    assert run_latency.build_questions(families, 12) == questions
    # Round one asks about the first fact of each family in turn, then round two begins.
    first_round = [
        f"What is the {present_keys(f)[0].replace('_', ' ')} for {f.entity}?" for f in families
    ]
    assert questions[:10] == first_round
    assert questions[10] == (
        f"What is the {present_keys(families[0])[1].replace('_', ' ')} for {families[0].entity}?"
    )
    assert len(set(questions)) == 12
    # More requests than distinct questions: the sequence starts over.
    assert run_latency.build_questions(families, pool + 2)[pool:] == questions[:2]
    assert len(run_latency.build_questions(families, pool + 2)) == pool + 2


def test_questions_only_ask_about_facts_the_first_version_states():
    families = generate(n_families=10).families
    stated = {
        f"What is the {key.replace('_', ' ')} for {family.entity}?"
        for family in families
        for key in present_keys(family)
    }

    assert set(run_latency.build_questions(families, 50)) <= stated


# --- fake LLM ----------------------------------------------------------------------------------


def test_fake_reply_answers_from_the_first_passage_with_a_citation():
    context = (
        "[P1] Aurora Router product policy, v1\n<passage>Aurora Router product policy\n\n"
        "The Aurora Router can be returned within 14 days of delivery. A filler.</passage>\n\n"
        "[P2] Other, v1\n<passage>Something else entirely.</passage>"
    )
    messages = [
        {"role": "system", "content": "answer from the passages"},
        {"role": "user", "content": f"{context}\n\nQuestion: What is the return window?"},
    ]

    reply = run_latency.fake_reply(messages)

    assert reply.endswith(" [P1]")
    assert "14 days" in reply
    assert "Something else" not in reply
    assert run_latency.fake_reply(messages) == reply


def test_fake_reply_to_claim_extraction_is_json_the_extractor_accepts():
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": "0. Refunds take 30 days."},
    ]

    assert json.loads(run_latency.fake_reply(messages)) == {"claims": []}


# --- command line ------------------------------------------------------------------------------


def test_parse_args_defaults(monkeypatch):
    monkeypatch.delenv("HALLUCISCOPE_BENCH_DATABASE_URL", raising=False)

    args = run_latency.parse_args([])

    assert args.requests == 50
    assert args.fake is False
    assert args.database_url == "postgresql://eg_app@localhost:54329/evidencegraph_bench"
    assert str(args.out).replace("\\", "/") == "evaluation/results/latency.json"


def test_parse_args_reads_the_environment_and_flags(monkeypatch):
    monkeypatch.setenv("HALLUCISCOPE_BENCH_DATABASE_URL", "postgresql://x@h/db")

    assert run_latency.parse_args([]).database_url == "postgresql://x@h/db"
    args = run_latency.parse_args(
        ["--requests", "3", "--fake", "--database-url", "postgresql://y@h/db", "--out", "o.json"]
    )
    assert (args.requests, args.fake, args.database_url) == (3, True, "postgresql://y@h/db")
    assert str(args.out) == "o.json"
    with pytest.raises(SystemExit):
        run_latency.parse_args(["--requests", "0"])


# --- runner (Postgres, fake models) ------------------------------------------------------------


@pg
async def test_a_failed_request_is_counted_and_left_out_of_the_percentiles(db, services):
    services.llm.default = run_latency.fake_reply
    # The first answer fails the way an unreachable Ollama does; the other two are answered.
    services.llm.responses = [LLMUnavailable("down")]
    families = generate(n_families=2).families
    tenant = await run_latency.ingest_corpus(services, families, "latency-test")
    questions = run_latency.build_questions(families, 3)

    rows = await run_latency.measure(services, tenant, questions)

    assert [row["status"] for row in rows] == ["failed", "succeeded", "succeeded"]
    assert rows[0]["error_code"] == "llm_unavailable"
    # The answers were cited, and the citations resolved and were verified.
    for row in rows[1:]:
        assert row["summary"]["n_claims"] >= 1
        assert row["summary"]["n_citations"] >= 1
        assert row["summary"]["n_invalid_citations"] == 0
    stages, failures = run_latency.summarize(rows)
    assert failures == {"llm_unavailable": 1}
    assert stages["total"]["p95_s"] == max(row["timings"]["total"] for row in rows[1:])
    assert stages["total"]["p95_s"] > 0


@pg
async def test_latency_smoke_fake(db, pg_urls, tmp_path):
    out = tmp_path / "results" / "latency.json"
    args = run_latency.parse_args(
        ["--database-url", pg_urls["app"], "--fake", "--requests", "3", "--out", str(out)]
    )

    result = await run_latency.run(args)

    assert result["n"] == 3
    assert result["failures"] == {}
    assert list(result["stages"]) == list(STAGE_KEYS)
    for stats in result["stages"].values():
        assert set(stats) == {"p50_s", "p95_s", "mean_s"}
        assert 0 <= stats["p50_s"] <= stats["p95_s"]
        assert stats["mean_s"] >= 0
    assert result["stages"]["total"]["p95_s"] > 0
    assert result["models"] == {
        "nli": "fake-rule-nli",
        "embedder": "fake-hash-384",
        "reranker": "fake-keyword",
        "llm": "fake-llm",
    }
    assert {"git_commit", "cpu", "ram_gb", "python", "postgres", "pgvector"} <= set(result["env"])
    assert set(result) == {"n", "stages", "models", "env", "failures"}
    assert json.loads(out.read_text(encoding="utf-8")) == result


# --- report ------------------------------------------------------------------------------------


def latency_results(**overrides) -> dict:
    result = {
        "n": 50,
        "stages": {
            "retrieving": {"p50_s": 0.1234, "p95_s": 0.2, "mean_s": 0.15},
            "generating": {"p50_s": 4.5, "p95_s": 9.0, "mean_s": 5.0},
            "extracting": {"p50_s": 3.0, "p95_s": 6.25, "mean_s": 3.5},
            "verifying": {"p50_s": 2.0, "p95_s": 2.5, "mean_s": 2.1},
            "persisting": {"p50_s": 0.01, "p95_s": 0.02, "mean_s": 0.012},
            "total": {"p50_s": 10.0, "p95_s": 18.0, "mean_s": 11.0},
        },
        "models": {
            "llm": "llama3.2:3b",
            "nli": "nli-x",
            "embedder": "embed-x",
            "reranker": "rerank-x",
        },
        "env": {
            "git_commit": "0123456789abcdef0123456789abcdef01234567",
            "git_dirty": False,
            "cpu": "Example CPU 9000",
            "ram_gb": 16.0,
            "python": "3.14.9",
            "torch": "2.14.1+cpu",
            "postgres": "17.11",
            "pgvector": "0.8.7",
        },
        "failures": {},
    }
    result.update(overrides)
    return result


def test_render_latency_golden():
    text = report.render_latency(latency_results())

    assert text.startswith("# End-to-end latency benchmark\n")
    assert "nearest-rank" in text
    assert "Example CPU 9000" in text
    assert "0123456789abcdef0123456789abcdef01234567" in text
    assert "LLM llama3.2:3b" in text
    assert "NLI nli-x" in text
    assert "- Requests: 50, failed: 0" in text
    assert "| Stage | p50 s | p95 s | mean s |" in text
    assert "| --- | ---: | ---: | ---: |" in text
    assert "| retrieving | 0.123 | 0.200 | 0.150 |" in text
    assert "| generating | 4.500 | 9.000 | 5.000 |" in text
    assert "| persisting | 0.010 | 0.020 | 0.012 |" in text
    assert "| total | 10.000 | 18.000 | 11.000 |" in text
    # The rows come in the order of the results file.
    order = [text.index(f"| {name} |") for name in latency_results()["stages"]]
    assert order == sorted(order)
    assert "\u2014" not in text
    assert "None" not in text
    assert "\r" not in text


def test_render_latency_lists_failures_by_error_code():
    results = latency_results(n=50, failures={"llm_timeout": 2, "retrieval_failed": 1})

    text = report.render_latency(results)

    assert "- Requests: 50, failed: 3 (llm_timeout 2, retrieval_failed 1)" in text
    assert "excluded" in text


def test_render_latency_without_a_successful_request_shows_no_table():
    results = latency_results(n=4, stages={}, failures={"llm_unavailable": 4})
    results["env"] = {**results["env"], "git_commit": None, "git_dirty": None}

    text = report.render_latency(results)

    assert "- Requests: 4, failed: 4 (llm_unavailable 4)" in text
    assert "No request succeeded" in text
    assert "| Stage |" not in text
    assert "unknown" in text
    assert "None" not in text


def test_report_cli_writes_the_latency_file(tmp_path):
    source = tmp_path / "latency.json"
    source.write_text(json.dumps(latency_results()), encoding="utf-8")
    out = tmp_path / "docs" / "latency.md"

    exit_code = report.main(["latency", str(source), "--out", str(out)])

    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "| total | 10.000 | 18.000 | 11.000 |" in written
    assert "\r" not in written
