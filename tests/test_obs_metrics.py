import threading
import urllib.error
import urllib.request

import pytest

from backend.obs import metrics
from backend.obs.metrics import Counter, Gauge, Histogram, Registry
from backend.obs.metrics_server import start_metrics_server


def test_counter_render():
    registry = Registry()
    counter = Counter("http_requests_total", "Requests.", ("route", "method", "status"))
    registry.register(counter)
    child = counter.labels(route="/x", method="GET", status="200")
    child.inc()
    child.inc()
    text = registry.render()
    assert "# HELP http_requests_total Requests.\n" in text
    assert "# TYPE http_requests_total counter\n" in text
    assert 'http_requests_total{route="/x",method="GET",status="200"} 2.0\n' in text


def test_unlabeled_counter_and_gauge_render():
    registry = Registry()
    counter = Counter("c_total", "C.")
    gauge = Gauge("g", "G.")
    registry.register(counter)
    registry.register(gauge)
    counter.inc(2.5)
    gauge.set(7)
    gauge.set(4)
    text = registry.render()
    assert "c_total 2.5\n" in text
    assert "# TYPE g gauge\n" in text
    assert "g 4.0\n" in text


def test_histogram_buckets_cumulative():
    registry = Registry()
    histogram = Histogram("d_seconds", "D.", ("route",))
    registry.register(histogram)
    child = histogram.labels(route="/x")
    child.observe(0.003)
    child.observe(0.2)
    text = registry.render()
    assert "# TYPE d_seconds histogram\n" in text
    assert 'd_seconds_bucket{route="/x",le="0.005"} 1.0\n' in text
    assert 'd_seconds_bucket{route="/x",le="0.1"} 1.0\n' in text
    assert 'd_seconds_bucket{route="/x",le="0.25"} 2.0\n' in text
    assert 'd_seconds_bucket{route="/x",le="1.0"} 2.0\n' in text
    assert 'd_seconds_bucket{route="/x",le="+Inf"} 2.0\n' in text
    assert 'd_seconds_count{route="/x"} 2.0\n' in text
    assert 'd_seconds_sum{route="/x"} 0.203\n' in text


def test_unlabeled_histogram_observe():
    registry = Registry()
    histogram = Histogram("h", "H.", buckets=(1, 2))
    registry.register(histogram)
    histogram.observe(1.5)
    text = registry.render()
    assert 'h_bucket{le="1.0"} 0.0\n' in text
    assert 'h_bucket{le="2.0"} 1.0\n' in text
    assert 'h_bucket{le="+Inf"} 1.0\n' in text
    assert "h_count 1.0\n" in text


def test_label_escaping():
    registry = Registry()
    counter = Counter("e_total", "E.", ("v",))
    registry.register(counter)
    counter.labels(v='a"b\nc\\d').inc()
    text = registry.render()
    assert 'e_total{v="a\\"b\\nc\\\\d"} 1.0\n' in text


def test_help_escaping():
    registry = Registry()
    registry.register(Counter("h_total", "line1\nline2 \\ end"))
    assert "# HELP h_total line1\\nline2 \\\\ end\n" in registry.render()


def test_empty_metrics_render_headers_only():
    registry = Registry()
    registry.register(Counter("plain_total", "P."))
    registry.register(Counter("lab_total", "L.", ("a",)))
    text = registry.render()
    assert "plain_total 0.0\n" in text
    assert "lab_total{" not in text
    assert "# TYPE lab_total counter\n" in text


def test_unknown_label_raises():
    counter = Counter("x_total", "X.", ("a", "b"))
    with pytest.raises(ValueError):
        counter.labels(a="1", c="2")
    with pytest.raises(ValueError):
        counter.labels(a="1")
    with pytest.raises(ValueError):
        counter.labels(a="1", b="2", c="3")


def test_labels_on_unlabeled_metric_raises():
    with pytest.raises(ValueError):
        Counter("y_total", "Y.").labels()


def test_labeled_metric_direct_mutation_raises():
    counter = Counter("z_total", "Z.", ("a",))
    with pytest.raises(ValueError):
        counter.inc()


def test_counter_rejects_negative_increment():
    with pytest.raises(ValueError):
        Counter("n_total", "N.").inc(-1)


def test_duplicate_registration_raises():
    registry = Registry()
    registry.register(Counter("dup_total", "D."))
    with pytest.raises(ValueError):
        registry.register(Counter("dup_total", "D."))


def test_counter_is_thread_safe():
    counter = Counter("t_total", "T.")

    def work():
        for _ in range(1000):
            counter.inc()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    registry = Registry()
    registry.register(counter)
    assert "t_total 8000.0\n" in registry.render()


def test_module_level_registry_has_named_metrics():
    text = metrics.REGISTRY.render()
    for name, kind in (
        ("http_requests_total", "counter"),
        ("http_request_duration_seconds", "histogram"),
        ("jobs_total", "counter"),
        ("job_duration_seconds", "histogram"),
        ("pipeline_stage_seconds", "histogram"),
        ("ingest_passages_total", "counter"),
        ("jobs_queued", "gauge"),
    ):
        assert f"# TYPE {name} {kind}\n" in text
    assert metrics.DEFAULT_BUCKETS[0] == 0.005
    assert metrics.DEFAULT_BUCKETS[-1] == 300


def _get(url, token=None):
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode(), response.headers
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read().decode(), error.headers


def _serve(token):
    registry = Registry()
    registry.register(Counter("srv_total", "S."))
    server = start_metrics_server("127.0.0.1", 0, token, registry)
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def test_metrics_server_auth():
    server, base = _serve(None)
    try:
        assert _get(f"{base}/metrics")[0] == 404
        assert _get(f"{base}/metrics", token="anything")[0] == 404
    finally:
        server.shutdown()
        server.server_close()

    server, base = _serve("s3cret")
    try:
        assert _get(f"{base}/metrics")[0] == 401
        assert _get(f"{base}/metrics", token="wrong")[0] == 401
        status, body, headers = _get(f"{base}/metrics", token="s3cret")
        assert status == 200
        assert "# TYPE" in body
        assert headers["Content-Type"] == "text/plain; version=0.0.4"
        assert _get(f"{base}/other", token="s3cret")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
