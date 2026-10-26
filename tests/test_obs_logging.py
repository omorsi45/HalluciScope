import io
import json
import logging

import pytest

from backend.obs.logging import JsonFormatter, bind, request_id_var


@pytest.fixture
def capture():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_obs_logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    yield logger, stream
    logger.handlers = []


def test_formatter_drops_unlisted_extra(capture):
    logger, stream = capture
    logger.info("analysis finished", extra={"n_claims": 3, "question": "SECRET-Q"})
    line = stream.getvalue()
    data = json.loads(line)
    assert data["n_claims"] == 3
    assert "question" not in data
    assert "SECRET-Q" not in line
    assert data["msg"] == "analysis finished"
    assert data["level"] == "INFO"
    assert data["logger"] == "test_obs_logging"
    assert data["ts"].endswith("+00:00") or data["ts"].endswith("Z")


def test_one_json_object_per_line(capture):
    logger, stream = capture
    logger.info("first")
    logger.info("second")
    lines = stream.getvalue().splitlines()
    assert [json.loads(x)["msg"] for x in lines] == ["first", "second"]


def test_bound_context_appears(capture):
    logger, stream = capture
    with bind(request_id="r1"):
        logger.info("inside")
    logger.info("outside")
    inside, outside = (json.loads(x) for x in stream.getvalue().splitlines())
    assert inside["request_id"] == "r1"
    assert "request_id" not in outside
    assert request_id_var.get() is None


def test_bind_unknown_key_raises():
    with pytest.raises(KeyError), bind(nonsense="x"):
        pass


def test_exception_message_not_logged(capture):
    logger, stream = capture
    try:
        raise ValueError("SECRET-DOC")
    except ValueError:
        logger.exception("job failed")
    line = stream.getvalue()
    data = json.loads(line)
    assert data["exc_type"] == "ValueError"
    assert "SECRET-DOC" not in line
    assert data["exc_frames"]
    assert all(" in " in frame and ":" in frame for frame in data["exc_frames"])


def test_configure_logging_installs_json_formatter():
    from backend.obs.logging import configure_logging

    root = logging.getLogger()
    uvicorn = logging.getLogger("uvicorn")
    access = logging.getLogger("uvicorn.access")
    saved = [
        (lg, list(lg.handlers), lg.level, lg.propagate, lg.disabled)
        for lg in (root, uvicorn, access)
    ]
    try:
        configure_logging("WARNING")
        assert root.level == logging.WARNING
        assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
        assert any(isinstance(h.formatter, JsonFormatter) for h in uvicorn.handlers)
        assert uvicorn.propagate is False
        assert access.disabled is True
    finally:
        for lg, handlers, level, propagate, disabled in saved:
            lg.handlers = handlers
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled


def test_non_string_message_logs_type_name_only(capture):
    logger, stream = capture
    try:
        raise ValueError("SECRET-DOC")
    except ValueError as error:
        logger.exception(error)
    line = stream.getvalue()
    assert "SECRET-DOC" not in line
    assert json.loads(line)["msg"] == "ValueError"


def _reject_constant(name):
    raise AssertionError(f"non-standard JSON constant emitted: {name}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_emitted_as_null(capture, value):
    logger, stream = capture
    logger.info("timed", extra={"duration_ms": value})
    data = json.loads(stream.getvalue(), parse_constant=_reject_constant)
    assert data["duration_ms"] is None


def test_arbitrary_object_in_whitelisted_key_emitted_as_type_name(capture):
    class Sneaky:
        def __str__(self):
            return "SECRET-OBJ"

        __repr__ = __str__

    logger, stream = capture
    logger.info("odd", extra={"route": Sneaky(), "n_claims": 2, "stage": None, "bytes": True})
    line = stream.getvalue()
    data = json.loads(line)
    assert "SECRET-OBJ" not in line
    assert data["route"] == "Sneaky"
    assert data["n_claims"] == 2
    assert data["stage"] is None
    assert data["bytes"] is True
