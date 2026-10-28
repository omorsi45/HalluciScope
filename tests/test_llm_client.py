import json

import httpx
import pytest

from backend.llm.client import LLMError, LLMFailure, LLMTimeout, LLMUnavailable, OllamaClient
from tests.fakes import ScriptedLLM

BASE_URL = "http://ollama.test:11434"
MODEL = "llama3.2:3b"
MESSAGES = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is the refund window?"},
]
SECRET = "SECRET-PROMPT"


@pytest.fixture
async def make_client():
    """Build an OllamaClient over a MockTransport handler; no test touches the network."""
    http_clients: list[httpx.AsyncClient] = []

    def factory(handler, **kwargs) -> OllamaClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http_clients.append(http)
        return OllamaClient(http, BASE_URL, MODEL, **kwargs)

    yield factory
    for http in http_clients:
        await http.aclose()


def _ok(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


async def test_chat_payload_and_content(make_client):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok("Refunds take 30 days.")

    client = make_client(handler)

    answer = await client.chat(MESSAGES, temperature=0.7)

    assert answer == "Refunds take 30 days."
    assert client.model_id == MODEL
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"{BASE_URL}/api/chat"
    # Exact equality: with no schema there must be no "format" key.
    assert json.loads(seen[0].content) == {
        "model": MODEL,
        "messages": MESSAGES,
        "stream": False,
        "options": {"temperature": 0.7},
    }


async def test_default_temperature_is_zero(make_client):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()

    await make_client(handler).chat(MESSAGES)

    assert json.loads(seen[0].content)["options"] == {"temperature": 0.0}


async def test_json_schema_sent_as_format(make_client):
    schema = {
        "type": "object",
        "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
        "required": ["claims"],
    }
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok('{"claims": []}')

    answer = await make_client(handler).chat(MESSAGES, json_schema=schema)

    assert answer == '{"claims": []}'
    body = json.loads(seen[0].content)
    assert body["format"] == schema
    assert body["stream"] is False


async def test_connect_error_unavailable(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    with pytest.raises(LLMUnavailable) as excinfo:
        await make_client(handler).chat(MESSAGES)

    assert isinstance(excinfo.value, LLMFailure)
    assert excinfo.value.code == "llm_unavailable"
    assert "ollama serve" in str(excinfo.value)


@pytest.mark.parametrize("timeout_exc", [httpx.ReadTimeout, httpx.ConnectTimeout])
async def test_timeout(make_client, timeout_exc):
    def handler(request: httpx.Request) -> httpx.Response:
        raise timeout_exc("timed out", request=request)

    with pytest.raises(LLMTimeout) as excinfo:
        await make_client(handler).chat(MESSAGES)

    assert isinstance(excinfo.value, LLMFailure)
    assert excinfo.value.code == "llm_timeout"


async def test_timeout_uses_client_default_and_per_call_override(make_client):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return _ok()

    client = make_client(handler, timeout_s=45.0)

    await client.chat(MESSAGES)
    await client.chat(MESSAGES, timeout_s=7.5)

    assert seen[0]["read"] == 45.0
    assert seen[1]["read"] == 7.5


@pytest.mark.parametrize("status", [404, 500, 503])
async def test_http_error_hides_body(make_client, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=SECRET)

    with pytest.raises(LLMError) as excinfo:
        await make_client(handler).chat(MESSAGES)

    assert excinfo.value.code == "llm_error"
    assert str(status) in str(excinfo.value)
    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


@pytest.mark.parametrize(
    "raw_body",
    [
        pytest.param(f"{SECRET} is not json".encode(), id="not-json"),
        pytest.param(json.dumps([SECRET]).encode(), id="json-list"),
        pytest.param(json.dumps({"note": SECRET}).encode(), id="no-message"),
        pytest.param(json.dumps({"message": {"note": SECRET}}).encode(), id="no-content"),
        pytest.param(json.dumps({"message": SECRET}).encode(), id="message-not-object"),
        pytest.param(json.dumps({"message": {"content": [SECRET]}}).encode(), id="content-not-str"),
        pytest.param(json.dumps({"message": {"content": None, "n": SECRET}}).encode(), id="null"),
    ],
)
async def test_malformed_body_llm_error(make_client, raw_body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_body)

    with pytest.raises(LLMError) as excinfo:
        await make_client(handler).chat(MESSAGES)

    assert excinfo.value.code == "llm_error"
    assert SECRET not in str(excinfo.value)


async def test_other_transport_failure_is_llm_error(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    with pytest.raises(LLMError) as excinfo:
        await make_client(handler).chat(MESSAGES)

    assert excinfo.value.code == "llm_error"


async def test_scripted_llm_records_calls():
    schema = {"type": "object"}
    llm = ScriptedLLM()
    llm.responses = ["first", LLMTimeout("scripted timeout")]
    llm.default = lambda messages: f"default:{messages[-1]['content']}"

    assert llm.model_id == "fake-llm"
    assert await llm.chat(MESSAGES, temperature=0.7, json_schema=schema) == "first"
    with pytest.raises(LLMTimeout):
        await llm.chat(MESSAGES)
    assert await llm.chat([{"role": "user", "content": "q"}]) == "default:q"

    assert llm.responses == []
    assert llm.calls == [
        {"messages": MESSAGES, "temperature": 0.7, "json_schema": schema},
        {"messages": MESSAGES, "temperature": 0.0, "json_schema": None},
        {"messages": [{"role": "user", "content": "q"}], "temperature": 0.0, "json_schema": None},
    ]
    assert ScriptedLLM("other").model_id == "other"
