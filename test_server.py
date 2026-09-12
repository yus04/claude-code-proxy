"""Tests for the Claude Code <-> GPT proxy.

These tests do not call any external API: LiteLLM is replaced by fakes so the
translation logic can be verified offline.

Run with: uv run pytest
"""

import json

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client():
    return TestClient(server.app)


# --------------------------------------------------------------------------- #
# Model mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "claude_model,expected",
    [
        ("claude-opus-5", "openai/gpt-5"),
        ("claude-sonnet-5", "openai/gpt-5"),
        ("claude-haiku-4-5", "openai/gpt-5-mini"),
        ("claude-haiku-4-5-20251001", "openai/gpt-5-mini"),
        ("claude-sonnet-4-5-20250929", "openai/gpt-5"),
        ("anthropic/claude-opus-5", "openai/gpt-5"),
        ("Claude-Sonnet-5", "openai/gpt-5"),
        # Unknown models are forwarded so a GPT model can be selected directly.
        ("gpt-5-nano", "openai/gpt-5-nano"),
    ],
)
def test_map_model_name(claude_model, expected):
    assert server.map_model_name(claude_model) == expected


def test_map_model_name_uses_azure_prefix(monkeypatch):
    monkeypatch.setattr(server, "USE_AZURE", True)
    assert server.map_model_name("claude-sonnet-5") == "azure/gpt-5"


def test_is_reasoning_model():
    assert server.is_reasoning_model("openai/gpt-5")
    assert server.is_reasoning_model("azure/gpt-5-mini")
    assert not server.is_reasoning_model("openai/gpt-5-chat-latest")
    assert not server.is_reasoning_model("openai/gpt-4.1")


@pytest.mark.parametrize(
    "thinking,expected",
    [
        (None, "medium"),
        ({"type": "disabled"}, "medium"),
        ({"type": "enabled", "budget_tokens": 1024}, "low"),
        ({"type": "enabled", "budget_tokens": 10000}, "medium"),
        ({"type": "enabled", "budget_tokens": 30000}, "high"),
    ],
)
def test_reasoning_effort_for(thinking, expected):
    config = server.ThinkingConfig(**thinking) if thinking else None
    assert server.reasoning_effort_for(config) == expected


# --------------------------------------------------------------------------- #
# Anthropic -> OpenAI conversion
# --------------------------------------------------------------------------- #


def test_system_prompt_and_plain_messages():
    request = server.MessagesRequest(
        model="claude-sonnet-5",
        max_tokens=100,
        system="You are helpful",
        messages=[{"role": "user", "content": "Hello"}],
    )
    payload = server.convert_anthropic_to_litellm(request)

    assert payload["model"] == "openai/gpt-5"
    assert payload["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]
    assert payload["max_tokens"] == 100
    assert payload["reasoning_effort"] == "medium"


def test_system_prompt_as_content_blocks():
    messages = server.convert_messages(
        [server.Message(role="user", content="hi")],
        system=[
            server.SystemContent(type="text", text="line one"),
            server.SystemContent(type="text", text="line two"),
        ],
    )
    assert messages[0] == {"role": "system", "content": "line one\n\nline two"}


def test_embedded_system_message_is_converted():
    messages = server.convert_messages(
        [
            server.Message(
                role="system",
                content=[{"type": "text", "text": "You are helpful"}],
            ),
            server.Message(role="user", content="hi"),
        ]
    )

    assert messages == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_tool_use_is_converted_to_tool_calls():
    messages = server.convert_messages(
        [
            server.Message(role="user", content="What is 2+2?"),
            server.Message(
                role="assistant",
                content=[
                    {"type": "text", "text": "Let me calculate"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
            ),
            server.Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "4"}],
                    }
                ],
            ),
        ]
    )

    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me calculate"
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "expression": "2+2"
    }

    assert messages[2] == {"role": "tool", "tool_call_id": "toolu_1", "content": "4"}


def test_tool_result_precedes_following_user_text():
    messages = server.convert_messages(
        [
            server.Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "42",
                    },
                    {"type": "text", "text": "thanks"},
                ],
            )
        ]
    )
    assert [message["role"] for message in messages] == ["tool", "user"]
    assert messages[1]["content"] == "thanks"


def test_thinking_and_unknown_blocks_are_ignored():
    messages = server.convert_messages(
        [
            server.Message(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web"},
                ],
            )
        ]
    )
    assert messages == [{"role": "assistant", "content": "answer"}]


def test_image_block_is_converted_to_image_url():
    messages = server.convert_messages(
        [
            server.Message(
                role="user",
                content=[
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGk=",
                        },
                    },
                ],
            )
        ]
    )
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,aGk="


def test_tools_and_tool_choice_conversion():
    request = server.MessagesRequest(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "name": "calculator",
                "description": "Evaluate expressions",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            }
        ],
        tool_choice={"type": "any"},
    )
    payload = server.convert_anthropic_to_litellm(request)

    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "calculator"
    assert payload["tool_choice"] == "required"


def test_tool_choice_specific_tool():
    assert server.convert_tool_choice({"type": "tool", "name": "weather"}) == {
        "type": "function",
        "function": {"name": "weather"},
    }


def test_max_tokens_is_capped_and_streaming_usage_requested():
    request = server.MessagesRequest(
        model="claude-sonnet-5",
        max_tokens=200000,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    payload = server.convert_anthropic_to_litellm(request)

    assert payload["max_tokens"] == server.MAX_OUTPUT_TOKENS
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_non_reasoning_model_has_no_reasoning_effort():
    request = server.MessagesRequest(
        model="gpt-4.1",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    payload = server.convert_anthropic_to_litellm(request)
    assert "reasoning_effort" not in payload


def test_azure_credentials_are_attached(monkeypatch):
    monkeypatch.setattr(server, "USE_AZURE", True)
    monkeypatch.setattr(
        server, "AZURE_API_BASE", "https://example.services.ai.azure.com"
    )
    monkeypatch.setattr(server, "AZURE_API_KEY", "test-key")
    monkeypatch.setattr(server, "AZURE_API_VERSION", "preview")

    payload = server.convert_anthropic_to_litellm(
        server.MessagesRequest(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert payload["model"] == "azure/gpt-5-mini"
    assert payload["api_base"] == "https://example.services.ai.azure.com"
    assert payload["api_key"] == "test-key"
    assert payload["api_version"] == "preview"


def test_azure_apim_openai_base_uses_deployment_path(monkeypatch):
    monkeypatch.setattr(server, "USE_AZURE", True)
    monkeypatch.setattr(
        server,
        "AZURE_API_BASE",
        "https://jpe-apim-platform.azure-api.net/foundry/openai",
    )
    monkeypatch.setattr(server, "AZURE_API_KEY", "test-key")
    monkeypatch.setattr(server, "AZURE_API_VERSION", "preview")
    monkeypatch.setattr(server, "AZURE_DEPLOYMENT_API_VERSION", "2025-02-01-preview")

    payload = server.convert_anthropic_to_litellm(
        server.MessagesRequest(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert payload["model"] == "azure/gpt-5"
    assert payload["api_base"] == "https://jpe-apim-platform.azure-api.net/foundry"
    assert payload["api_version"] == "2025-02-01-preview"


def test_azure_openai_base_with_dated_version_is_not_normalized(monkeypatch):
    monkeypatch.setattr(server, "AZURE_DEPLOYMENT_API_VERSION", "2025-02-01-preview")

    api_base, api_version = server.normalize_azure_api_settings(
        "https://jpe-apim-platform.azure-api.net/foundry/openai",
        "2026-01-01-preview",
    )

    assert api_base == "https://jpe-apim-platform.azure-api.net/foundry/openai"
    assert api_version == "2026-01-01-preview"


def test_azure_base_without_openai_suffix_is_not_normalized(monkeypatch):
    monkeypatch.setattr(server, "AZURE_DEPLOYMENT_API_VERSION", "2025-02-01-preview")

    api_base, api_version = server.normalize_azure_api_settings(
        "https://jpe-apim-platform.azure-api.net/foundry",
        "preview",
    )

    assert api_base == "https://jpe-apim-platform.azure-api.net/foundry"
    assert api_version == "preview"


def test_non_apim_openai_suffix_is_not_normalized(monkeypatch):
    monkeypatch.setattr(server, "AZURE_DEPLOYMENT_API_VERSION", "2025-02-01-preview")

    api_base, api_version = server.normalize_azure_api_settings(
        "https://example.services.ai.azure.com/proxy/openai",
        "preview",
    )

    assert api_base == "https://example.services.ai.azure.com/proxy/openai"
    assert api_version == "preview"


# --------------------------------------------------------------------------- #
# OpenAI -> Anthropic conversion
# --------------------------------------------------------------------------- #


def make_request(**kwargs):
    defaults = {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    defaults.update(kwargs)
    return server.MessagesRequest(**defaults)


def test_response_conversion_text():
    completion = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    response = server.convert_litellm_to_anthropic(completion, make_request())

    assert response.content == [{"type": "text", "text": "Hello!"}]
    assert response.stop_reason == "end_turn"
    assert response.model == "claude-sonnet-5"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 3


def test_response_conversion_tool_calls():
    completion = {
        "id": "chatcmpl-2",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2+2"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    response = server.convert_litellm_to_anthropic(completion, make_request())

    assert response.stop_reason == "tool_use"
    assert response.content[0]["type"] == "tool_use"
    assert response.content[0]["name"] == "calculator"
    assert response.content[0]["input"] == {"expression": "2+2"}


def test_response_conversion_length_and_empty_content():
    completion = {
        "id": "chatcmpl-3",
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {},
    }
    response = server.convert_litellm_to_anthropic(completion, make_request())

    assert response.stop_reason == "max_tokens"
    assert response.content == [{"type": "text", "text": ""}]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


def test_health_and_models(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["models"]["haiku"] == server.SMALL_MODEL

    models = client.get("/v1/models").json()
    assert {model["id"] for model in models["data"]} == set(server.CLAUDE_MODEL_ALIASES)


def test_count_tokens(client):
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "Hello world"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_create_message(monkeypatch, client):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "id": "chatcmpl-4",
            "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1},
        }

    monkeypatch.setattr(server.litellm, "acompletion", fake_acompletion)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "thinking": {"type": "adaptive"},
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Answer concisely."}],
                },
                {"role": "user", "content": "Capital of France?"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "Paris"}]
    assert body["model"] == "claude-sonnet-5"
    assert captured["model"] == "openai/gpt-5"
    assert captured["messages"][0] == {
        "role": "system",
        "content": "Answer concisely.",
    }


def test_create_message_streaming(monkeypatch, client):
    async def fake_acompletion(**kwargs):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ' "2+2"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 9},
            },
        ]

        async def generator():
            for chunk in chunks:
                yield chunk

        return generator()

    monkeypatch.setattr(server.litellm, "acompletion", fake_acompletion)

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [
        line[len("event: ") :] for line in body.splitlines() if line.startswith("event: ")
    ]
    payloads = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]

    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert "content_block_start" in events

    text = "".join(
        payload["delta"]["text"]
        for payload in payloads
        if payload.get("type") == "content_block_delta"
        and payload["delta"]["type"] == "text_delta"
    )
    assert text == "Hello world"

    arguments = "".join(
        payload["delta"]["partial_json"]
        for payload in payloads
        if payload.get("type") == "content_block_delta"
        and payload["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(arguments) == {"expression": "2+2"}

    message_delta = next(
        payload for payload in payloads if payload.get("type") == "message_delta"
    )
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"] == {"input_tokens": 4, "output_tokens": 9}

    # The text block and the tool block must use different indexes.
    starts = [
        payload for payload in payloads if payload.get("type") == "content_block_start"
    ]
    assert [start["index"] for start in starts] == [0, 1]
    assert starts[1]["content_block"]["name"] == "calculator"


def test_upstream_errors_use_anthropic_error_format(monkeypatch, client):
    class UpstreamError(Exception):
        status_code = 429

    async def fake_acompletion(**kwargs):
        raise UpstreamError("rate limited")

    monkeypatch.setattr(server.litellm, "acompletion", fake_acompletion)

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 429
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    # Upstream error details stay in the proxy log.
    assert "rate limited" not in body["error"]["message"]
