import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from backend.config import Settings
from backend.verify.nli import DebertaNLIScorer, label_indices
from backend.verify.types import NLIResult
from tests.fakes import RuleNLIScorer

REPO_ROOT = Path(__file__).resolve().parents[1]

# (entailment, contradiction, neutral) the fake model reports for pair i. Hand-picked so a
# scorer that mixes up the columns, or the order of the pairs, cannot return the same numbers.
PROBABILITIES = [
    (0.70, 0.20, 0.10),
    (0.10, 0.80, 0.10),
    (0.05, 0.05, 0.90),
    (0.60, 0.30, 0.10),
    (0.25, 0.25, 0.50),
]
# One more row for the six-pair length tests, distinct from the five above.
PROBABILITIES_6 = [*PROBABILITIES, (0.35, 0.45, 0.20)]
# Premise word counts of the six-pair length tests: long and short premises interleaved.
WORD_COUNTS = [1, 50, 2, 49, 3, 48]
# Two head layouts. The first is the DeBERTa config's; the second differs in every column.
LAYOUTS = [
    {"contradiction": 0, "entailment": 1, "neutral": 2},
    {"entailment": 0, "neutral": 1, "contradiction": 2},
]


class FakeTokenizer:
    """Encodes a pair as the number in its premise ("premise-3" -> 3) and records each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.calls.append({"premises": list(premises), "hypotheses": list(hypotheses), **kwargs})
        ids = torch.tensor([[int(premise.rsplit("-", 1)[1])] for premise in premises])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel:
    """Returns logits whose softmax is PROBABILITIES[pair], placed in the layout's columns."""

    def __init__(
        self, layout: dict[str, int], probabilities: list[tuple[float, float, float]] | None = None
    ) -> None:
        self.layout = layout
        self.probabilities = PROBABILITIES if probabilities is None else probabilities
        self.config = SimpleNamespace(id2label={index: label for label, index in layout.items()})
        self.device: str | None = None
        self.eval_called = False
        self.batch_sizes: list[int] = []
        self.inference_mode_flags: list[bool] = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.shape[0])
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        rows = []
        for pair_id in input_ids[:, 0].tolist():
            entail, contra, neutral = self.probabilities[pair_id]
            row = [0.0, 0.0, 0.0]
            row[self.layout["entailment"]] = math.log(entail)
            row[self.layout["contradiction"]] = math.log(contra)
            row[self.layout["neutral"]] = math.log(neutral)
            rows.append(row)
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def _pairs(count: int) -> list[tuple[str, str]]:
    return [(f"premise-{i}", f"hypothesis-{i}") for i in range(count)]


def _pairs_with_word_counts(word_counts: list[int]) -> list[tuple[str, str]]:
    """Pair i has a premise of word_counts[i] words, ending in "premise-i" so the fake finds i."""
    return [
        (" ".join(["word"] * (count - 1) + [f"premise-{i}"]), f"hypothesis-{i}")
        for i, count in enumerate(word_counts)
    ]


def _scorer(
    layout: dict[str, int],
    batch_size: int = 2,
    probabilities: list[tuple[float, float, float]] | None = None,
):
    tokenizer, model = FakeTokenizer(), FakeModel(layout, probabilities)
    scorer = DebertaNLIScorer(
        tokenizer, model, model_id="fake-nli", device="cpu", batch_size=batch_size
    )
    return scorer, tokenizer, model


def _as_tuple(result: NLIResult) -> tuple[float, float, float]:
    return (result.entail, result.contra, result.neutral)


def test_label_indices_from_config():
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    assert label_indices(id2label) == {"contradiction": 0, "entailment": 1, "neutral": 2}


def test_label_indices_accepts_string_keys_and_any_case():
    # A config round-tripped through JSON has string keys; checkpoints differ in label case.
    id2label = {"0": "ENTAILMENT", "1": "Neutral", "2": "contradiction"}

    assert label_indices(id2label) == {"entailment": 0, "neutral": 1, "contradiction": 2}


@pytest.mark.parametrize("missing", ["entailment", "contradiction", "neutral"])
def test_label_indices_missing_label_raises(missing):
    labels = ["contradiction", "entailment", "neutral"]
    id2label = {
        index: ("LABEL_9" if label == missing else label) for index, label in enumerate(labels)
    }

    with pytest.raises(ValueError, match=missing):
        label_indices(id2label)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_scorer_maps_labels_and_batches(layout):
    scorer, tokenizer, model = _scorer(layout, batch_size=2)

    results = scorer.score(_pairs(5))

    # 5 pairs at batch size 2 are three forward passes, split 2, 2, 1.
    assert model.batch_sizes == [2, 2, 1]
    assert [call["premises"] for call in tokenizer.calls] == [
        ["premise-0", "premise-1"],
        ["premise-2", "premise-3"],
        ["premise-4"],
    ]
    # Results are in input order and each class is read from the column the config names.
    assert len(results) == 5
    for result, expected in zip(results, PROBABILITIES, strict=True):
        assert _as_tuple(result) == pytest.approx(expected, abs=1e-5)
    assert all(type(value) is float for result in results for value in _as_tuple(result))


def test_scorer_passes_premise_first_and_hypothesis_second():
    scorer, tokenizer, _ = _scorer(LAYOUTS[0], batch_size=16)

    scorer.score(_pairs(2))

    assert tokenizer.calls[0]["premises"] == ["premise-0", "premise-1"]
    assert tokenizer.calls[0]["hypotheses"] == ["hypothesis-0", "hypothesis-1"]


def test_batches_group_similar_lengths():
    scorer, tokenizer, model = _scorer(LAYOUTS[0], batch_size=3, probabilities=PROBABILITIES_6)

    scorer.score(_pairs_with_word_counts(WORD_COUNTS))

    # Longest first: the three long premises share a batch, the three short ones the other,
    # so neither batch pads a short premise out to a long one.
    assert model.batch_sizes == [3, 3]
    assert [[len(premise.split()) for premise in call["premises"]] for call in tokenizer.calls] == [
        [50, 49, 48],
        [3, 2, 1],
    ]


def test_sorted_batching_keeps_input_order():
    scorer, _, _ = _scorer(LAYOUTS[0], batch_size=3, probabilities=PROBABILITIES_6)

    results = scorer.score(_pairs_with_word_counts(WORD_COUNTS))

    # The fake model's row depends on the pair, so a result left in batch order is visible.
    assert len(results) == 6
    for result, expected in zip(results, PROBABILITIES_6, strict=True):
        assert _as_tuple(result) == pytest.approx(expected, abs=1e-5)


def test_premise_only_truncation():
    scorer, tokenizer, _ = _scorer(LAYOUTS[0])

    scorer.score(_pairs(1))

    kwargs = tokenizer.calls[0]
    # Only the premise may be cut; the hypothesis is the claim being judged.
    assert kwargs["truncation"] == "only_first"
    assert kwargs["max_length"] == 512
    assert kwargs["padding"] is True
    assert kwargs["return_tensors"] == "pt"


def test_scorer_moves_model_to_device_and_runs_in_inference_mode():
    tokenizer, model = FakeTokenizer(), FakeModel(LAYOUTS[0])

    scorer = DebertaNLIScorer(tokenizer, model, model_id="fake-nli", device="cpu", batch_size=2)
    assert model.device == "cpu"
    assert model.eval_called
    assert scorer.model_id == "fake-nli"

    scorer.score(_pairs(3))
    assert model.inference_mode_flags == [True, True]


def test_empty_pairs():
    scorer, tokenizer, model = _scorer(LAYOUTS[0])

    assert scorer.score([]) == []
    assert tokenizer.calls == []
    assert model.batch_sizes == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_scorer_rejects_batch_size_below_one(batch_size):
    # A zero or negative size would otherwise score nothing and return [] for real pairs.
    with pytest.raises(ValueError, match="batch_size"):
        DebertaNLIScorer(
            FakeTokenizer(),
            FakeModel(LAYOUTS[0]),
            model_id="fake-nli",
            device="cpu",
            batch_size=batch_size,
        )


def test_scorer_rejects_model_without_nli_labels():
    model = FakeModel(LAYOUTS[0])
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

    with pytest.raises(ValueError, match="entailment"):
        DebertaNLIScorer(FakeTokenizer(), model, model_id="fake-nli", device="cpu")


def test_importing_scorer_does_not_import_ml_libraries():
    code = (
        "import json, sys; import backend.verify.nli; "
        "print(json.dumps([m for m in ('torch', 'sentence_transformers', 'transformers') "
        "if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    assert json.loads(result.stdout) == []


def test_load_nli_uses_settings_and_the_loaded_models_labels(monkeypatch):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from backend.models.loader import load_nli

    loaded: list[tuple[str, str]] = []
    tokenizer, model = FakeTokenizer(), FakeModel(LAYOUTS[1])

    def fake_tokenizer(name):
        loaded.append(("tokenizer", name))
        return tokenizer

    def fake_model(name):
        loaded.append(("model", name))
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", fake_model)

    scorer = load_nli(Settings(nli_model="org/nli-test", nli_batch_size=4, model_device="cpu"))
    results = scorer.score(_pairs(5))

    assert sorted(loaded) == [("model", "org/nli-test"), ("tokenizer", "org/nli-test")]
    assert scorer.model_id == "org/nli-test"
    assert model.device == "cpu"
    assert model.batch_sizes == [4, 1]
    assert _as_tuple(results[1]) == pytest.approx(PROBABILITIES[1], abs=1e-5)


@pytest.mark.parametrize(
    ("premise", "hypothesis", "expected"),
    [
        # Same claim, different number: contradiction.
        ("Refunds take 30 days.", "Refunds take 14 days.", "contra"),
        ("Refunds take 30 days.", "Refunds take 30 days.", "entail"),
        ("Refunds take 30 days.", "Shipping is free.", "neutral"),
        # A premise with more words still entails a hypothesis it covers.
        ("Refunds take 30 days after purchase.", "Refunds take 30 days.", "entail"),
        # Fixtures share "take" and "days" across documents; a different subject is no conflict.
        ("Exchanges take 14 days.", "Refunds take 30 days.", "neutral"),
        ("Refunds take 30 days.", "Exchanges take 14 days.", "neutral"),
        # One hypothesis number does occur in the premise, so the numbers do not conflict.
        ("Refunds take 30 days.", "Refunds take 14 or 30 days.", "neutral"),
        # The premise states no number to contradict.
        ("Refunds are processed quickly.", "Refunds take 30 days.", "neutral"),
        # Contradiction needs every other hypothesis word to be present in the premise.
        ("Refunds take 30 days.", "Refunds take 14 business days.", "neutral"),
    ],
)
def test_rule_nli_cases(premise, hypothesis, expected):
    (result,) = RuleNLIScorer().score([(premise, hypothesis)])

    others = {"entail", "contra", "neutral"} - {expected}
    assert getattr(result, expected) == 0.9
    assert all(getattr(result, other) == 0.05 for other in others)


def test_rule_nli_keeps_input_order_and_handles_empty_input():
    scorer = RuleNLIScorer()
    pairs = [
        ("Refunds take 30 days.", "Shipping is free."),
        ("Refunds take 30 days.", "Refunds take 14 days."),
        ("Refunds take 30 days.", "Refunds take 30 days."),
    ]

    results = scorer.score(pairs)

    assert [_as_tuple(result) for result in results] == [
        (0.05, 0.05, 0.9),
        (0.05, 0.9, 0.05),
        (0.9, 0.05, 0.05),
    ]
    assert scorer.model_id == "fake-rule-nli"
    assert scorer.score([]) == []


@pytest.mark.models
def test_real_deberta_obvious_pairs():
    from backend.models.loader import load_nli

    settings = Settings(model_device="cpu")
    scorer = load_nli(settings)
    pairs = [
        ("A soccer game with multiple males playing.", "Some men are playing a sport."),
        ("The store opens at 9 am.", "The store never opens."),
    ]

    entailed, contradicted = scorer.score(pairs)

    assert scorer.model_id == settings.nli_model == "cross-encoder/nli-deberta-v3-large"
    assert entailed.entail > 0.9
    assert contradicted.contra > 0.9
    for result in (entailed, contradicted):
        assert math.isclose(result.entail + result.contra + result.neutral, 1.0, abs_tol=1e-4)
    # Scores come back in input order whatever the batch size.
    swapped = scorer.score(list(reversed(pairs)))
    assert swapped[0].contra == pytest.approx(contradicted.contra, abs=1e-3)
    assert swapped[1].entail == pytest.approx(entailed.entail, abs=1e-3)
    assert scorer.score([]) == []
