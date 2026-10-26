"""Minimal Prometheus metrics registry (text exposition format 0.0.4), stdlib only."""

import itertools
import math
import threading
from collections.abc import Sequence

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    return repr(float(value))


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label_block(names: Sequence[str], values: Sequence[str], extra: str = "") -> str:
    pairs = [f'{n}="{_escape_label(v)}"' for n, v in zip(names, values, strict=True)]
    if extra:
        pairs.append(extra)
    return "{" + ",".join(pairs) + "}" if pairs else ""


class _CounterChild:
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        with self._lock:
            self.value += amount


class _GaugeChild:
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.value = 0.0

    def set(self, value: float) -> None:
        with self._lock:
            self.value = float(value)


class _HistogramChild:
    def __init__(self, lock: threading.Lock, buckets: tuple[float, ...]) -> None:
        self._lock = lock
        self._buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        with self._lock:
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self.counts[i] += 1
                    break
            self.total += 1
            self.sum += value


class _Metric:
    TYPE = ""

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()
        self._children: dict[tuple[str, ...], object] = {}
        if not self.labelnames:
            self._children[()] = self._new_child()

    def _new_child(self) -> object:
        raise NotImplementedError

    def labels(self, **kw: object) -> object:
        if not self.labelnames:
            raise ValueError(f"{self.name} has no labels")
        if set(kw) != set(self.labelnames):
            raise ValueError(f"{self.name} requires exactly labels {list(self.labelnames)}")
        key = tuple(str(kw[n]) for n in self.labelnames)
        with self._lock:
            child = self._children.get(key)
            if child is None:
                child = self._children[key] = self._new_child()
        return child

    def _unlabeled(self) -> object:
        if self.labelnames:
            raise ValueError(f"{self.name} requires labels {list(self.labelnames)}")
        return self._children[()]

    def _sample_lines(self) -> list[str]:
        with self._lock:
            return [
                f"{self.name}{_label_block(self.labelnames, key)} {_fmt(child.value)}"
                for key, child in sorted(self._children.items())
            ]

    def render(self) -> str:
        lines = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.TYPE}"]
        lines.extend(self._sample_lines())
        return "\n".join(lines) + "\n"


class Counter(_Metric):
    TYPE = "counter"

    def _new_child(self) -> _CounterChild:
        return _CounterChild(self._lock)

    def inc(self, amount: float = 1.0) -> None:
        self._unlabeled().inc(amount)


class Gauge(_Metric):
    TYPE = "gauge"

    def _new_child(self) -> _GaugeChild:
        return _GaugeChild(self._lock)

    def set(self, value: float) -> None:
        self._unlabeled().set(value)


class Histogram(_Metric):
    TYPE = "histogram"

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        bounds = tuple(float(b) for b in buckets)
        if not bounds or any(not math.isfinite(b) for b in bounds):
            raise ValueError("buckets must be non-empty and finite")
        if any(a >= b for a, b in itertools.pairwise(bounds)):
            raise ValueError("buckets must be strictly increasing")
        self._bounds = bounds
        super().__init__(name, help, labelnames)

    def _new_child(self) -> _HistogramChild:
        return _HistogramChild(self._lock, self._bounds)

    def observe(self, value: float) -> None:
        self._unlabeled().observe(value)

    def _sample_lines(self) -> list[str]:
        lines: list[str] = []
        with self._lock:
            for key, child in sorted(self._children.items()):
                cumulative = 0
                for bound, count in zip(self._bounds, child.counts, strict=True):
                    cumulative += count
                    block = _label_block(self.labelnames, key, f'le="{_fmt(bound)}"')
                    lines.append(f"{self.name}_bucket{block} {_fmt(cumulative)}")
                block = _label_block(self.labelnames, key, 'le="+Inf"')
                lines.append(f"{self.name}_bucket{block} {_fmt(child.total)}")
                plain = _label_block(self.labelnames, key)
                lines.append(f"{self.name}_sum{plain} {_fmt(child.sum)}")
                lines.append(f"{self.name}_count{plain} {_fmt(child.total)}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, _Metric] = {}

    def register(self, metric: _Metric) -> _Metric:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric {metric.name} already registered")
            self._metrics[metric.name] = metric
        return metric

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics.values())
        return "".join(metric.render() for metric in metrics)


REGISTRY = Registry()

HTTP_REQUESTS = REGISTRY.register(
    Counter("http_requests_total", "HTTP requests handled.", ("route", "method", "status"))
)
HTTP_DURATION = REGISTRY.register(
    Histogram("http_request_duration_seconds", "HTTP request latency.", ("route", "method"))
)
JOBS = REGISTRY.register(Counter("jobs_total", "Background jobs finished.", ("kind", "outcome")))
JOB_DURATION = REGISTRY.register(
    Histogram("job_duration_seconds", "Background job duration.", ("kind",))
)
STAGE_SECONDS = REGISTRY.register(
    Histogram("pipeline_stage_seconds", "Pipeline stage duration.", ("stage",))
)
INGEST_PASSAGES = REGISTRY.register(
    Counter("ingest_passages_total", "Passages processed during ingest.", ("outcome",))
)
JOBS_QUEUED = REGISTRY.register(Gauge("jobs_queued", "Jobs waiting in the queue."))
