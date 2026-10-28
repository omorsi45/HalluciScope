"""Render benchmark result files as markdown for ``docs/results/``.

    python -m evaluation.report retrieval evaluation/results/retrieval_*.json \
        --out docs/results/retrieval.md
    python -m evaluation.report factuality evaluation/results/factuality_test.json \
        --ablations evaluation/results/factuality_test_ablations.json \
        --atomic evaluation/results/factuality_test_atomic.json \
        --out docs/results/factuality.md --figures docs/results/figures
    python -m evaluation.report temporal evaluation/results/temporal.json \
        --out docs/results/temporal.md
    python -m evaluation.report latency evaluation/results/latency.json \
        --out docs/results/latency.md

Every number in the output is read from a results file written by ``evaluation.run_retrieval``,
``evaluation.run_factuality``, ``evaluation.run_temporal`` or ``evaluation.run_latency``;
nothing is computed or filled in here beyond rounding.
"""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TABLE_HEADER = (
    "| System | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
)

_METHODOLOGY = (
    "Methodology: BM25 parameters were fixed before evaluation. Passages are sentence-aligned "
    "and every system collapses its passage ranking to documents with MaxP (a document ranks "
    "by its best passage). There was no tuning on test qrels."
)


def _text(value: Any, unit: str = "") -> str:
    return "unknown" if value is None else f"{value}{unit}"


def _commit_line(env: Mapping[str, Any]) -> str:
    dirty = env.get("git_dirty")
    state = "unknown" if dirty is None else ("dirty" if dirty else "clean")
    return f"- Commit: {_text(env.get('git_commit'))} (working tree: {state})"


def _section(result: Mapping[str, Any]) -> str:
    env = result["env"]
    params = result["params"]
    models = result["models"]
    checks = result["checks"]
    lines = [
        f"## {result['dataset']} ({result['split']} split)",
        "",
        f"- Queries: {result['n_queries']}, documents: {result['n_docs']}, "
        f"passages: {result['n_passages']}, documents skipped as empty: "
        f"{result['n_skipped_docs']}",
        f"- Run: {result['started_at']} to {result['finished_at']} "
        f"(ingest {result['ingest_seconds']:.1f} s, total {result['runtime_seconds']:.1f} s)",
        _commit_line(env),
        f"- CPU: {_text(env.get('cpu'))}, RAM: {_text(env.get('ram_gb'), ' GB')}",
        f"- Python: {_text(env.get('python'))}, torch: {_text(env.get('torch'))}, "
        f"PostgreSQL: {_text(env.get('postgres'))}, pgvector: {_text(env.get('pgvector'))}",
        f"- Models: embedder {models['embedder']}, reranker {models['reranker']}",
        f"- Parameters: BM25 k1 = {params['k1']}, b = {params['b']}; RRF k = {params['rrf_k']}; "
        f"candidates = {params['candidates']}; rerank depth = {params['rerank_depth']} documents; "
        f"HNSW ef_search = {params['ef_search']}; passage target = "
        f"{params['passage_target_chars']} characters",
        "",
        *_TABLE_HEADER,
    ]
    for name, metrics in result["systems"].items():
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.4f} | {metrics['recall@10']:.4f} "
            f"| {metrics['recall@100']:.4f} | {metrics['mrr@10']:.4f} "
            f"| {metrics['latency_ms_p50']:.1f} | {metrics['latency_ms_p95']:.1f} |"
        )
    lines += [
        "",
        "Checks:",
        "",
        f"- acl_leaks: {checks['acl_leaks']} (must be 0)",
        f"- bm25_parity_max_abs_diff: {checks['bm25_parity_max_abs_diff']:.3e} (must be < 1e-9)",
        "",
        _METHODOLOGY,
    ]
    return "\n".join(lines)


def render_retrieval(results: Sequence[Mapping[str, Any]]) -> str:
    """One section per results mapping, in the order given, as a markdown document."""
    sections = [_section(result) for result in results]
    return "# Retrieval benchmark\n\n" + "\n\n".join(sections) + "\n"


_FACTUALITY_COLUMNS = (
    "System",
    "Det. P",
    "Det. R",
    "Det. F1 [95% CI]",
    "Macro-F1",
    "False-supported [95% CI]",
    "Accepted risk (CP upper)",
    "Coverage",
)


def _header(columns: Sequence[str]) -> tuple[str, str]:
    return "| " + " | ".join(columns) + " |", "| --- |" + " ---: |" * (len(columns) - 1)


_FACTUALITY_HEADER = _header((*_FACTUALITY_COLUMNS, "AURC"))
_TASK_HEADER = _header(_FACTUALITY_COLUMNS)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _threshold(value: float | None) -> str:
    return "none" if value is None else f"{value}"


def _with_interval(point: float | None, interval: Sequence[float | None]) -> str:
    lo, hi = interval
    bounds = "n/a" if lo is None or hi is None else f"{lo:.4f}, {hi:.4f}"
    return f"{_num(point)} [{bounds}]"


def _system_row(name: str, block: Mapping[str, Any], aurc: float | None = None) -> str:
    detection, ci, risk = block["detection"], block["ci"], block["accepted_risk"]
    cells = [
        name,
        _num(detection["precision"]),
        _num(detection["recall"]),
        _with_interval(detection["f1"], ci["detection_f1"]),
        _num(block["macro_f1"]),
        _with_interval(block["false_supported_rate"], ci["false_supported_rate"]),
        f"{_num(risk['risk'])} ({_num(risk['cp_upper'])})",
        _num(block["supported_coverage"]),
    ]
    if aurc is not None:
        cells.append(_num(aurc))
    return "| " + " | ".join(cells) + " |"


def _run_section(results: Mapping[str, Any]) -> str:
    data, artifact, models = results["data"], results["artifact"], results["models"]
    evaluated, env, bootstrap = data["evaluated"], results["env"], results["bootstrap"]
    thresholds, logged = results["thresholds"], results["test_evaluations"]
    sizes = "; ".join(
        f"{role}: {size['sources']} sources"
        + (f", {size['units']} units" if "units" in size else "")
        for role, size in data["role_sizes"].items()
    )
    meaning = (
        "final evaluation"
        if results["split"] == "test"
        else "development run, the test role was not read"
    )
    lines = [
        "## Run",
        "",
        f"- Evaluated role: {results['split']} split ({meaning}); {evaluated['units']} units "
        f"from {evaluated['sources']} sources and {evaluated['responses']} responses",
        f"- Run: {results['created_at']}",
        _commit_line(env),
        f"- Python: {_text(env.get('python'))}, platform: {_text(env.get('platform'))}",
        f"- Dataset: {data['dataset']} at commit {data['dataset_commit']}; "
        f"manifest hash {data['manifest_hash']}; seed {data['seed']}",
        f"- Role sizes: {sizes}",
        f"- Models: NLI {models['nli']}, embedder {models['embedder']}, "
        f"reranker {models['reranker']}",
        f"- Artifact: {artifact['id']} ({artifact['premise_mode']} premises, evidence_k = "
        f"{artifact['evidence_k']}, feature set {_text(artifact['feature_set'])}, "
        f"temperature {artifact['temperature']:.4f})",
        f"- Decision thresholds: tau_s = {_threshold(artifact['tau_s'])}, "
        f"tau_c = {_threshold(artifact['tau_c'])} (alpha_s = {artifact['alpha_s']}, "
        f"alpha_c = {artifact['alpha_c']}, delta = {artifact['delta']})",
        f"- Baseline thresholds from the select role: NLI-only {thresholds['nli_only']}, "
        f"similarity-only {thresholds['similarity_only']}",
        f"- Test evaluations logged for this artifact: {logged['count']} (log: {logged['log']})",
        f"- Intervals: {bootstrap['ci']:.0%} cluster bootstrap by {bootstrap['by']}, "
        f"{bootstrap['n_resamples']} resamples, seed {bootstrap['seed']}",
    ]
    return "\n".join(lines)


def _systems_section(results: Mapping[str, Any], figures: str) -> str:
    systems = results["systems"]
    lines = ["## Systems", "", *_FACTUALITY_HEADER]
    for name, system in systems.items():
        lines.append(_system_row(name, system["overall"], system["risk_coverage"]["aurc"]))
    scores = "; ".join(f"{name}: {system['score']}" for name, system in systems.items())
    lines += [
        "",
        "Detection treats a not-supported unit as the positive class. False-supported is the "
        "share of unsupported units predicted supported. Accepted risk is the share of "
        "unsupported units among those predicted supported, with the one-sided "
        f"Clopper-Pearson upper bound at delta = {results['cp_delta']} in brackets. Coverage is "
        "the share of units predicted supported. The AURC is the area under the "
        f"risk-coverage curve of the support score ({scores}).",
        "",
        f"![Risk-coverage curves]({figures}/risk_coverage.png)",
    ]
    return "\n".join(lines)


def _task_section(results: Mapping[str, Any]) -> str:
    systems = results["systems"]
    tasks = sorted({task for system in systems.values() for task in system["by_task"]})
    lines = ["## By task type"]
    for task in tasks:
        lines += ["", f"### {task}", "", *_TASK_HEADER]
        lines += [
            _system_row(name, system["by_task"][task])
            for name, system in systems.items()
            if task in system["by_task"]
        ]
    return "\n".join(lines)


def _calibration_section(results: Mapping[str, Any], figures: str) -> str:
    calibration = results["systems"]["evidencegraph"]["calibration"]
    return "\n".join(
        [
            "## Calibration (evidencegraph)",
            "",
            "| ECE (top label) | ECE (classwise) | Brier | NLL |",
            "| ---: | ---: | ---: | ---: |",
            f"| {_num(calibration['ece'])} | {_num(calibration['classwise_ece'])} "
            f"| {_num(calibration['brier'])} | {_num(calibration['nll'])} |",
            "",
            f"![Reliability diagram]({figures}/reliability.png)",
        ]
    )


def _ablation_section(ablations: Mapping[str, Any]) -> str:
    lines = [
        "## Ablations",
        "",
        "Each row retrains the calibrator from the cache on the fit, select and risk roles "
        f"and scores it on the {ablations['split']} split.",
        "",
        "| Configuration | Det. F1 | Macro-F1 | False-supported | Accepted risk | NLL | ECE "
        "| tau_s | tau_c |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ablations["rows"]:
        lines.append(
            f"| {row['kind']}: {row['name']} | {_num(row['detection_f1'])} "
            f"| {_num(row['macro_f1'])} | {_num(row['false_supported_rate'])} "
            f"| {_num(row['accepted_risk'])} | {_num(row['nll'])} | {_num(row['ece'])} "
            f"| {_threshold(row['tau_s'])} | {_threshold(row['tau_c'])} |"
        )
    return "\n".join(lines)


def _atomic_section(atomic: Mapping[str, Any]) -> str:
    by_task = ", ".join(f"{task} {count}" for task, count in sorted(atomic["by_task"].items()))
    methods = ", ".join(f"{method}: {count}" for method, count in atomic["methods"].items())
    lines = [
        "## Atomic-claim subset",
        "",
        f"- Responses: {atomic['n_responses']} of {atomic['n_requested']} requested, sampled by "
        f"source with seed {atomic['seed']}, stratified by task type ({by_task}); "
        f"{atomic['n_units']} sentence units",
        f"- Atomic claims: {atomic['n_claims']} extracted by {atomic['llm_model']}, "
        f"{atomic['n_claims_unaligned']} not aligned to a sentence",
        f"- Extraction methods: {methods}",
        "",
        "| Level | Det. P | Det. R | Det. F1 | False-supported |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, level in (("Atomic claims", "atomic"), ("Sentence units", "sentence")):
        detection = atomic[level]["detection"]
        lines.append(
            f"| {label} | {_num(detection['precision'])} | {_num(detection['recall'])} "
            f"| {_num(detection['f1'])} | {_num(atomic[level]['false_supported_rate'])} |"
        )
    return "\n".join(lines)


def _factuality_methodology(results: Mapping[str, Any]) -> str:
    artifact = results["artifact"]
    return "\n".join(
        [
            "## Methodology",
            "",
            "- Sentence units are labeled by span overlap: a response sentence is contradicted "
            "when a conflict span overlaps it, insufficient when only a baseless-information "
            "span does, and supported otherwise.",
            "- The splits are by source: every response of a source sits in one role (fit, "
            "select, risk or test), so no source is shared between training and evaluation.",
            f"- The Learn-then-Test guarantee (alpha_s = {artifact['alpha_s']}, alpha_c = "
            f"{artifact['alpha_c']}, delta = {artifact['delta']}) is stated for sentence units "
            "and for the per-source false-supported rate: with probability at least 1 - delta "
            "over the draw of the risk sources, the mean over sources of the fraction of "
            "unsupported sentence units accepted as supported is at most alpha_s, and the same "
            "holds at alpha_c for the fraction of non-contradicted units accepted as "
            "contradicted. It is not a bound on the accepted risk, on a single run's interval, "
            "or on atomic claims.",
            "- The legacy baselines are scored on the same evidence as evidencegraph, the "
            "cached evidence signals of the same sentence units, so the comparison isolates the "
            "scoring method from the retrieval and unit changes.",
            "- Nothing was tuned on test: the single-signal thresholds come from the select "
            "role, the test role is read only by the final evaluation, and every such run is "
            "appended to the test log, whose count for this artifact is given above.",
        ]
    )


def render_factuality(
    results: Mapping[str, Any],
    ablations: Mapping[str, Any] | None,
    atomic: Mapping[str, Any] | None,
    *,
    figures: str = "figures",
) -> str:
    """The factuality report as markdown. ``figures`` is the folder of the PNGs relative to it."""
    sections = [
        "# Factuality benchmark (RAGTruth)",
        _run_section(results),
        _systems_section(results, figures),
        _task_section(results),
        _calibration_section(results, figures),
    ]
    if ablations is not None:
        sections.append(_ablation_section(ablations))
    if atomic is not None:
        sections.append(_atomic_section(atomic))
    sections.append(_factuality_methodology(results))
    return "\n\n".join(sections) + "\n"


def plot_factuality(results: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Write ``reliability.png`` and ``risk_coverage.png`` to ``out_dir`` and return the paths."""
    # matplotlib is an optional (eval) dependency and the Agg backend needs no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    systems = results["systems"]

    calibration = systems["evidencegraph"]["calibration"]
    bins = [each for each in calibration["reliability_bins"] if each["count"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(
        [each["mean_conf"] for each in bins],
        [each["accuracy"] for each in bins],
        marker="o",
        label=f"evidencegraph (ECE {calibration['ece']:.4f})",
    )
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability of the top verdict")
    ax.legend(loc="upper left")
    reliability = out_dir / "reliability.png"
    fig.savefig(reliability, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Systems that rank the units alike draw the same curve, so the line styles differ.
    styles = ["-", "--", "-.", ":", "-"]
    for index, (name, system) in enumerate(systems.items()):
        curve = system["risk_coverage"]
        ax.plot(
            curve["coverage"],
            curve["risk"],
            linestyle=styles[index % len(styles)],
            label=f"{name} (AURC {curve['aurc']:.4f})",
        )
    ax.set_xlabel("coverage (share of units accepted as supported)")
    ax.set_ylabel("risk (share of accepted units that are unsupported)")
    ax.set_xlim(0, 1)
    ax.set_title("Risk-coverage")
    ax.legend(loc="lower right", fontsize="small")
    risk_coverage = out_dir / "risk_coverage.png"
    fig.savefig(risk_coverage, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [reliability, risk_coverage]


_TEMPORAL_INTRO = (
    "This is an out-of-distribution check labeled by construction: every gold verdict follows "
    "from the generated document versions (a claim is supported when the fact has the claim's "
    "value at that version, contradicted when it has another value, insufficient when the fact "
    "is absent). The calibrator trained on RAGTruth is used unchanged; nothing was tuned on "
    "this data. The 30-case sample in [temporal-sample.md](temporal-sample.md) is for human "
    "review."
)

_TEMPORAL_DEFINITIONS = (
    "Definitions:",
    "",
    "- As-of analyses use each version's effective time plus one day; the live row is the "
    "analysis run before the new versions were uploaded (gold: version 1).",
    "- Temporal correctness: a claim counts when its predicted verdict changes between "
    "versions exactly where the gold verdict changes.",
    "- Supporting-span correctness: of the claims predicted supported, those whose first "
    "supporting evidence item is in the version valid at the as-of time and whose span text "
    "contains the claim's value.",
    "- Source change: gold changed means the gold verdict at the last version differs from "
    "version 1; predicted changed means the claim's latest verdict (re-check or original) "
    "differs from its original one.",
    "- A rate with no cases behind it is shown as n/a.",
)

_TEMPORAL_VERDICT_HEADER = (
    "| As-of | Claims | Accuracy | Macro-F1 | False-supported |",
    "| --- | ---: | ---: | ---: | ---: |",
)


def _count_of(count: int, total: int, rate: float | None) -> str:
    return f"{count} of {total} ({_num(rate)})"


def _verdict_row(label: str, scores: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {scores['n']} | {_num(scores['accuracy'])} | {_num(scores['macro_f1'])} "
        f"| {_num(scores['false_supported_rate'])} |"
    )


def render_temporal(results: Mapping[str, Any]) -> str:
    """The synthetic temporal and provenance benchmark as markdown."""
    env, corpus = results["env"], results["corpus"]
    models, calibrator = results["models"], results["calibrator"]
    validity, spans = results["evidence_validity"], results["span_correctness"]
    changes, temporal = results["source_change"], results["temporal_correctness"]
    by_kind = ", ".join(f"{kind} {count}" for kind, count in corpus["claims_by_kind"].items())
    links = _count_of(validity["n_valid_links"], validity["n_links"], validity["link_rate"])
    citations = _count_of(
        validity["n_valid_citations"], validity["n_citations"], validity["citation_rate"]
    )
    span_correct = _count_of(spans["n_correct"], spans["n_supported"], spans["rate"])

    sections = [
        "# Temporal and provenance benchmark (synthetic)\n\n" + _TEMPORAL_INTRO,
        "\n".join(
            [
                "## Environment",
                "",
                f"- Run: {results['started_at']} to {results['finished_at']} "
                f"(total {results['runtime_seconds']:.1f} s), seed {results['seed']}",
                _commit_line(env),
                f"- CPU: {_text(env.get('cpu'))}, RAM: {_text(env.get('ram_gb'), ' GB')}",
                f"- Python: {_text(env.get('python'))}, torch: {_text(env.get('torch'))}, "
                f"PostgreSQL: {_text(env.get('postgres'))}, pgvector: "
                f"{_text(env.get('pgvector'))}",
                f"- Models: NLI {models['nli']}, embedder {models['embedder']}, "
                f"reranker {models['reranker']}",
                f"- Calibrator: {calibrator['id']} ({calibrator['premise_mode']} premises, "
                f"evidence_k = {calibrator['evidence_k']})",
            ]
        ),
        "\n".join(
            [
                "## Corpus",
                "",
                f"- Families: {corpus['families']}, distractors: {corpus['distractors']}, "
                f"conflicting copies in the second tenant: {corpus['foreign']}",
                f"- Document versions: {corpus['versions']}; claims: {corpus['claims']} "
                f"({by_kind})",
                f"- Analyses: {corpus['analyses']} (one live per family, one as-of per "
                "family version)",
                f"- Re-verification jobs run: {results['reverify']['jobs']}",
            ]
        ),
        "\n".join(
            [
                "## Verdicts by as-of time",
                "",
                *_TEMPORAL_VERDICT_HEADER,
                _verdict_row("live (before new versions)", results["live"]),
                *(_verdict_row(f"v{row['version']}", row) for row in results["per_version"]),
                "",
                "False-supported is the share of claims whose gold verdict is not supported "
                "that were predicted supported.",
            ]
        ),
        "\n".join(
            [
                "## Temporal correctness",
                "",
                "| Claims | Correct | Rate |",
                "| ---: | ---: | ---: |",
                f"| {temporal['n_claims']} | {temporal['n_correct']} | {_num(temporal['rate'])} |",
            ]
        ),
        "\n".join(
            [
                "## Evidence and citation validity",
                "",
                f"- Evidence links valid: {links} over {validity['n_analyses']} analyses",
                f"- Citations valid: {citations}",
                f"- Supporting-span correctness: {span_correct}",
                f"- Supporting item in the valid version: "
                f"{spans['n_version_correct']} of {spans['n_supported']}; span contains the "
                f"claim's value: {spans['n_span_contains_value']} of {spans['n_supported']}",
            ]
        ),
        "\n".join(
            [
                "## Source-change detection",
                "",
                f"Claims checked at least once: {changes['n_checked']} of {changes['n_claims']}.",
                "",
                "| Gold changed | Predicted changed | True positives | Precision | Recall |",
                "| ---: | ---: | ---: | ---: | ---: |",
                f"| {changes['n_gold_changed']} | {changes['n_predicted_changed']} "
                f"| {changes['tp']} | {_num(changes['precision'])} | {_num(changes['recall'])} |",
            ]
        ),
        "\n".join(
            [
                "## Cross-tenant leakage",
                "",
                f"- cross_tenant_links: {results['cross_tenant_links']} (must be 0)",
            ]
        ),
        "\n".join(_TEMPORAL_DEFINITIONS),
    ]
    return "\n\n".join(sections) + "\n"


_LATENCY_INTRO = (
    "End-to-end latency of ask analyses over the documents of the synthetic temporal corpus. "
    "Each request retrieves passages, has the LLM write a cited answer, extracts its claims "
    "and verifies them, in one process and one request after another. A stage's time is its "
    "entry in the analysis's recorded timings, and the total covers the whole run. "
    "Percentiles are nearest-rank, so each one is an observed time."
)


def render_latency(results: Mapping[str, Any]) -> str:
    """The end-to-end latency benchmark as markdown."""
    env, models = results["env"], results["models"]
    failures = results["failures"]
    n_failed = sum(failures.values())
    by_code = ", ".join(f"{code} {count}" for code, count in failures.items())
    requests = f"- Requests: {results['n']}, failed: {n_failed}" + (
        f" ({by_code})" if failures else ""
    )
    if results["stages"]:
        table = [
            "| Stage | p50 s | p95 s | mean s |",
            "| --- | ---: | ---: | ---: |",
            *(
                f"| {stage} | {stats['p50_s']:.3f} | {stats['p95_s']:.3f} | {stats['mean_s']:.3f} |"
                for stage, stats in results["stages"].items()
            ),
        ]
        if failures:
            table += [
                "",
                "The statistics cover the requests that succeeded; failed requests are excluded.",
            ]
    else:
        table = ["No request succeeded, so no latency was measured."]

    sections = [
        "# End-to-end latency benchmark\n\n" + _LATENCY_INTRO,
        "\n".join(
            [
                "## Environment",
                "",
                _commit_line(env),
                f"- CPU: {_text(env.get('cpu'))}, RAM: {_text(env.get('ram_gb'), ' GB')}",
                f"- Python: {_text(env.get('python'))}, torch: {_text(env.get('torch'))}, "
                f"PostgreSQL: {_text(env.get('postgres'))}, pgvector: "
                f"{_text(env.get('pgvector'))}",
                f"- Models: LLM {models['llm']}, NLI {models['nli']}, "
                f"embedder {models['embedder']}, reranker {models['reranker']}",
            ]
        ),
        "\n".join(["## Latency by stage", "", requests, "", *table]),
    ]
    return "\n\n".join(sections) + "\n"


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the committed file uses LF whatever the platform.
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.report")
    commands = parser.add_subparsers(dest="command", required=True)
    retrieval = commands.add_parser("retrieval", help="render retrieval benchmark results")
    retrieval.add_argument("results", nargs="+", type=Path, help="results JSON files")
    retrieval.add_argument("--out", type=Path, default=Path("docs/results/retrieval.md"))
    factuality = commands.add_parser("factuality", help="render the factuality results")
    factuality.add_argument("results", type=Path, help="results JSON of run_factuality")
    factuality.add_argument("--ablations", type=Path, help="the ablations JSON of the same run")
    factuality.add_argument("--atomic", type=Path, help="the atomic-subset JSON of the same run")
    factuality.add_argument("--out", type=Path, default=Path("docs/results/factuality.md"))
    factuality.add_argument("--figures", type=Path, default=Path("docs/results/figures"))
    temporal = commands.add_parser("temporal", help="render the synthetic temporal results")
    temporal.add_argument("results", type=Path, help="results JSON of run_temporal")
    temporal.add_argument("--out", type=Path, default=Path("docs/results/temporal.md"))
    latency = commands.add_parser("latency", help="render the end-to-end latency results")
    latency.add_argument("results", type=Path, help="results JSON of run_latency")
    latency.add_argument("--out", type=Path, default=Path("docs/results/latency.md"))
    args = parser.parse_args(argv)

    if args.command == "retrieval":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        _write_markdown(args.out, render_retrieval(results))
        return 0

    if args.command == "temporal":
        temporal_results = json.loads(args.results.read_text(encoding="utf-8"))
        _write_markdown(args.out, render_temporal(temporal_results))
        return 0

    if args.command == "latency":
        latency_results = json.loads(args.results.read_text(encoding="utf-8"))
        _write_markdown(args.out, render_latency(latency_results))
        return 0

    def read(path: Path | None) -> Any:
        return None if path is None else json.loads(path.read_text(encoding="utf-8"))

    factuality_results = read(args.results)
    plot_factuality(factuality_results, args.figures)
    figures = Path(os.path.relpath(args.figures, args.out.parent)).as_posix()
    text = render_factuality(
        factuality_results, read(args.ablations), read(args.atomic), figures=figures
    )
    _write_markdown(args.out, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
