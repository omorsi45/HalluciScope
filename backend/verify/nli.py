"""NLI scoring of (premise, hypothesis) pairs with a DeBERTa cross-encoder.

NLI checkpoints disagree on the order of their output columns, and the legacy pipeline once
read the wrong one. The column of each class is therefore always looked up in the loaded
model's `config.id2label`, never assumed.

torch is imported when scoring, not at import time, so importing this module is cheap.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from backend.verify.types import NLIResult

MAX_LENGTH = 512
_CLASSES = ("entailment", "contradiction", "neutral")


def label_indices(id2label: Mapping[Any, str]) -> dict[str, int]:
    """Output column of each NLI class, from a model's `id2label`.

    Keys may be ints or numeric strings (a config read back from JSON) and labels are matched
    case-insensitively. A config without all three classes is not an NLI head: ValueError.
    """
    by_label = {str(label).lower(): int(index) for index, label in id2label.items()}
    missing = [name for name in _CLASSES if name not in by_label]
    if missing:
        raise ValueError(f"NLI model labels {sorted(by_label)} are missing: {', '.join(missing)}")
    return {name: by_label[name] for name in _CLASSES}


class DebertaNLIScorer:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_id: str,
        device: str,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._columns = label_indices(model.config.id2label)
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self.model_id = model_id

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        """Score (premise, hypothesis) pairs, `batch_size` at a time; results in input order.

        Pairs are batched longest first (by characters, a cheap proxy for token count; stable on
        ties) so a batch pads to similar lengths instead of to its one longest pair.
        """
        if len(pairs) == 0:
            return []
        import torch

        order = sorted(
            range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]), reverse=True
        )
        scored: dict[int, NLIResult] = {}
        with torch.inference_mode():
            for start in range(0, len(order), self._batch_size):
                indices = order[start : start + self._batch_size]
                batch = [pairs[i] for i in indices]
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    # Only the premise may be cut: the hypothesis is the claim being judged.
                    truncation="only_first",
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {name: tensor.to(self._device) for name, tensor in encoded.items()}
                logits = self._model(**inputs).logits
                rows = logits.float().softmax(dim=-1).cpu().tolist()
                for index, row in zip(indices, rows, strict=True):
                    scored[index] = NLIResult(
                        entail=row[self._columns["entailment"]],
                        contra=row[self._columns["contradiction"]],
                        neutral=row[self._columns["neutral"]],
                    )
        return [scored[i] for i in range(len(pairs))]
