"""Anthropic Messages API <-> OpenAI (GPT-5) proxy.

Claude Code talks to this server using the Anthropic Messages API. The proxy
translates the requests into OpenAI Chat Completions calls (via LiteLLM) that
are served by GPT-5 class models hosted on Microsoft Foundry or on OpenAI, and
translates the responses back into the Anthropic format.
"""

import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


class MessageFilter(logging.Filter):
    """Block noisy third party log messages."""

    BLOCKED_PHRASES = [
        "LiteLLM completion()",
        "HTTP Request:",
        "selected model name for cost calculation",
        "utils.py",
        "cost_calculator",
    ]

    def filter(self, record):
        if isinstance(getattr(record, "msg", None), str):
            return not any(phrase in record.msg for phrase in self.BLOCKED_PHRASES)
        return True


logging.getLogger().addFilter(MessageFilter())

# LiteLLM silently removes parameters that the target model does not accept
# (e.g. top_p / stop / temperature != 1 for the GPT-5 reasoning models).
litellm.drop_params = True

app = FastAPI(title="Claude Code to GPT proxy")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Microsoft Foundry endpoint, e.g. https://<resource>.services.ai.azure.com
# (without the /openai/v1 suffix, LiteLLM appends it).
AZURE_API_BASE = (os.environ.get("AZURE_API_BASE") or "").rstrip("/")

# API key of the Foundry resource (key based authentication).
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")

# "preview", "v1" and "latest" select the Foundry/Azure OpenAI v1 API surface.
# A dated value such as 2025-02-01-preview selects the legacy deployment API.
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "preview")

# Fallback: plain OpenAI (or any other OpenAI compatible endpoint).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Microsoft Foundry is used when an endpoint is configured for it.
USE_AZURE = bool(AZURE_API_BASE)

# Claude -> GPT model mapping. On Microsoft Foundry these are deployment names.
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-5")  # claude opus
MIDDLE_MODEL = os.environ.get("MIDDLE_MODEL", "gpt-5")  # claude sonnet
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-5-mini")  # claude haiku

# Default reasoning effort used for GPT-5 class models when the client does not
# request extended thinking.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium").lower()

VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
if REASONING_EFFORT not in VALID_REASONING_EFFORTS:
    logger.warning(
        f"Unknown REASONING_EFFORT '{REASONING_EFFORT}', falling back to 'medium'."
    )
    REASONING_EFFORT = "medium"

# Port used when running `python server.py`.
PORT = int(os.environ.get("PORT", "8082"))

# Safety net: Claude Code asks for very large max_tokens values, larger than
# what most GPT deployments accept.
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32768"))


def is_reasoning_model(model: str) -> bool:
    """GPT-5/GPT-6 class models expose the `reasoning_effort` parameter."""
    name = model.split("/")[-1].lower()
    if name.startswith("gpt-5-chat") or name.startswith("gpt-5.1-chat"):
        return False
    return name.startswith("gpt-5") or name.startswith("gpt-6")


# --------------------------------------------------------------------------- #
# Anthropic API schema
# --------------------------------------------------------------------------- #


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Any], Dict[str, Any], None] = None
    is_error: Optional[bool] = None


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str = ""
    signature: Optional[str] = None


class ContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: Optional[str] = None


ContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
    # Forward compatibility: unknown block types are accepted and ignored
    # instead of failing the whole request.
    Dict[str, Any],
]

BLOCK_TYPES = {
    "text": ContentBlockText,
    "image": ContentBlockImage,
    "tool_use": ContentBlockToolUse,
    "tool_result": ContentBlockToolResult,
    "thinking": ContentBlockThinking,
    "redacted_thinking": ContentBlockRedactedThinking,
}


def normalize_block(block: Any) -> Optional[BaseModel]:
    """Return a typed content block, or None for unsupported block types."""
    if isinstance(block, BaseModel):
        return block
    if isinstance(block, dict):
        block_class = BLOCK_TYPES.get(block.get("type", ""))
        if block_class is not None:
            try:
                return block_class(**block)
            except Exception:
                logger.warning(f"Ignoring malformed content block: {block.get('type')}")
                return None
    logger.warning(f"Ignoring unsupported content block: {block}")
    return None


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ThinkingConfig(BaseModel):
    type: Optional[Literal["enabled", "disabled", "adaptive"]] = "enabled"
    budget_tokens: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.type != "disabled"


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Dict[str, Any]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# --------------------------------------------------------------------------- #
# Model mapping
# --------------------------------------------------------------------------- #


def map_model_name(model: str) -> str:
    """Map a Claude model name to the configured GPT model (LiteLLM format).

    Claude Code currently sends Claude Sonnet 5, Claude Haiku 4.5 and Claude
    Opus 5 model ids. They are mapped by family:

    * ``*opus*``   -> ``BIG_MODEL``
    * ``*sonnet*`` -> ``MIDDLE_MODEL``
    * ``*haiku*``  -> ``SMALL_MODEL``

    Any other model name is forwarded as-is, so a specific GPT model can be
    selected directly (e.g. ``gpt-5-mini`` or ``/model gpt-5-nano``).
    """
    clean = model.split("/")[-1] if "/" in model else model
    lowered = clean.lower()

    if "opus" in lowered:
        target = BIG_MODEL
    elif "sonnet" in lowered:
        target = MIDDLE_MODEL
    elif "haiku" in lowered:
        target = SMALL_MODEL
    else:
        target = clean

    prefix = "azure/" if USE_AZURE else "openai/"
    return f"{prefix}{target}"


def reasoning_effort_for(request_thinking: Optional[ThinkingConfig]) -> str:
    """Translate Anthropic extended thinking into an OpenAI reasoning effort."""
    if request_thinking is None or not request_thinking.enabled:
        return REASONING_EFFORT

    budget = request_thinking.budget_tokens
    if budget is None:
        return REASONING_EFFORT
    if budget <= 2048:
        return "low"
    if budget <= 16384:
        return "medium"
    return "high"


# --------------------------------------------------------------------------- #
# Anthropic -> OpenAI conversion
# --------------------------------------------------------------------------- #


def parse_tool_result_content(content: Any) -> str:
    """Normalize the content of a tool_result block into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def convert_image_block(block: ContentBlockImage) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic image block into an OpenAI image_url part."""
    source = block.source or {}
    source_type = source.get("type")

    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }

    if source_type == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}

    logger.warning(f"Unsupported image source type: {source_type}")
    return None


def convert_messages(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic messages into OpenAI chat messages."""
    openai_messages: List[Dict[str, Any]] = []

    if system:
        if isinstance(system, str):
            system_text = system
        else:
            system_text = "\n\n".join(
                block.text if isinstance(block, SystemContent) else block.get("text", "")
                for block in system
            ).strip()
        if system_text:
            openai_messages.append({"role": "system", "content": system_text})

    for message in messages:
        content = message.content

        if isinstance(content, str):
            if content:
                openai_messages.append({"role": message.role, "content": content})
            continue

        text_parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_messages: List[Dict[str, Any]] = []

        for raw_block in content:
            block = normalize_block(raw_block)
            if block is None:
                continue
            block_type = getattr(block, "type", None)

            if block_type == "text":
                if block.text:
                    text_parts.append({"type": "text", "text": block.text})
            elif block_type == "image":
                image_part = convert_image_block(block)
                if image_part:
                    text_parts.append(image_part)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input or {}, ensure_ascii=False),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": parse_tool_result_content(block.content) or "(empty)",
                    }
                )
            elif block_type in ("thinking", "redacted_thinking"):
                # Reasoning content is not replayed to the OpenAI API.
                continue

        # Tool results must directly follow the assistant message that
        # requested them.
        openai_messages.extend(tool_messages)

        if message.role == "assistant":
            if text_parts or tool_calls:
                assistant_message: Dict[str, Any] = {"role": "assistant"}
                assistant_text = "\n".join(
                    part["text"] for part in text_parts if part["type"] == "text"
                )
                assistant_message["content"] = assistant_text or None
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                openai_messages.append(assistant_message)
        elif message.role in ("user", "system") and text_parts:
            only_text = all(part["type"] == "text" for part in text_parts)
            openai_messages.append(
                {
                    "role": message.role,
                    "content": "\n".join(part["text"] for part in text_parts)
                    if only_text
                    else text_parts,
                }
            )

    return openai_messages


def convert_tools(tools: Optional[List[Tool]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return openai_tools


def convert_tool_choice(tool_choice: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def convert_anthropic_to_litellm(request: MessagesRequest) -> Dict[str, Any]:
    """Build the LiteLLM (OpenAI compatible) request payload."""
    model = map_model_name(request.model)

    litellm_request: Dict[str, Any] = {
        "model": model,
        "messages": convert_messages(request.messages, request.system),
        "max_tokens": min(request.max_tokens, MAX_OUTPUT_TOKENS),
        "stream": bool(request.stream),
    }

    if request.temperature is not None:
        litellm_request["temperature"] = request.temperature
    if request.top_p is not None:
        litellm_request["top_p"] = request.top_p
    if request.stop_sequences:
        litellm_request["stop"] = request.stop_sequences

    if is_reasoning_model(model):
        litellm_request["reasoning_effort"] = reasoning_effort_for(request.thinking)

    tools = convert_tools(request.tools)
    if tools:
        litellm_request["tools"] = tools
        tool_choice = convert_tool_choice(request.tool_choice)
        if tool_choice:
            litellm_request["tool_choice"] = tool_choice

    if request.stream:
        litellm_request["stream_options"] = {"include_usage": True}

    litellm_request["api_key"] = AZURE_API_KEY if USE_AZURE else OPENAI_API_KEY
    if USE_AZURE:
        litellm_request["api_base"] = AZURE_API_BASE
        litellm_request["api_version"] = AZURE_API_VERSION
    elif OPENAI_BASE_URL:
        litellm_request["api_base"] = OPENAI_BASE_URL

    return litellm_request


# --------------------------------------------------------------------------- #
# OpenAI -> Anthropic conversion
# --------------------------------------------------------------------------- #

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:  # pragma: no cover - defensive
                continue
    return getattr(value, "__dict__", {}) or {}


def convert_litellm_to_anthropic(
    litellm_response: Any, request: MessagesRequest
) -> MessagesResponse:
    """Convert an OpenAI chat completion into an Anthropic message."""
    response = _as_dict(litellm_response)
    choices = response.get("choices") or [{}]
    choice = _as_dict(choices[0])
    message = _as_dict(choice.get("message", {}))
    usage = _as_dict(response.get("usage", {}))

    content: List[Dict[str, Any]] = []

    text = message.get("content") or ""
    if text:
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        tool_call = _as_dict(tool_call)
        function = _as_dict(tool_call.get("function", {}))
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Tool arguments are not valid JSON: {arguments}")
                arguments = {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "input": arguments,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    return MessagesResponse(
        id=response.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        model=request.model,
        content=content,
        stop_reason=STOP_REASON_MAP.get(choice.get("finish_reason") or "stop", "end_turn"),
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
        ),
    )


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def handle_streaming(response_generator, request: MessagesRequest):
    """Translate an OpenAI stream into Anthropic server sent events."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield sse("ping", {"type": "ping"})

    text_block_open = False
    next_index = 0
    text_index = 0
    # Maps the OpenAI tool call index to the Anthropic content block index.
    tool_block_indexes: Dict[int, int] = {}
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"

    try:
        async for chunk in response_generator:
            chunk = _as_dict(chunk)

            usage = _as_dict(chunk.get("usage") or {})
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens) or input_tokens
                output_tokens = (
                    usage.get("completion_tokens", output_tokens) or output_tokens
                )

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = _as_dict(choices[0])
            delta = _as_dict(choice.get("delta") or {})

            delta_text = delta.get("content")
            if delta_text:
                if not text_block_open:
                    text_index = next_index
                    next_index += 1
                    text_block_open = True
                    yield sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": text_index,
                        "delta": {"type": "text_delta", "text": delta_text},
                    },
                )

            for tool_call in delta.get("tool_calls") or []:
                tool_call = _as_dict(tool_call)
                openai_index = tool_call.get("index", 0) or 0
                function = _as_dict(tool_call.get("function") or {})

                if openai_index not in tool_block_indexes:
                    if text_block_open:
                        text_block_open = False
                        yield sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": text_index},
                        )
                    block_index = next_index
                    next_index += 1
                    tool_block_indexes[openai_index] = block_index
                    yield sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_call.get("id")
                                or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": function.get("name") or "",
                                "input": {},
                            },
                        },
                    )

                arguments = function.get("arguments")
                if arguments:
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_block_indexes[openai_index],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        },
                    )

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        if text_block_open:
            yield sse("content_block_stop", {"type": "content_block_stop", "index": text_index})
        for block_index in tool_block_indexes.values():
            yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})

        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
        )
        yield sse("message_stop", {"type": "message_stop"})

    except Exception:  # pragma: no cover - network failures
        logger.exception("Error while streaming response")
        yield sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "The upstream stream failed. See the proxy logs for details.",
                },
            },
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


def error_response(exc: Exception) -> JSONResponse:
    """Return an Anthropic style error payload.

    The upstream exception is only written to the proxy log: error messages
    coming from the backend may contain internal details, so the client gets a
    generic message with the status code preserved.
    """
    status_code = int(getattr(exc, "status_code", 500) or 500)
    if status_code < 400 or status_code > 599:
        status_code = 500

    error_types = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
    }
    error_type = error_types.get(status_code, "api_error")

    logger.error(f"Request failed ({status_code}): {exc}", exc_info=True)

    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": (
                    f"The upstream request failed with status {status_code}. "
                    "See the proxy logs for details."
                ),
            },
        },
    )


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    litellm_request = convert_anthropic_to_litellm(request)

    log_request(
        "POST",
        raw_request.url.path,
        request.model,
        litellm_request["model"],
        len(litellm_request["messages"]),
        len(request.tools or []),
    )

    try:
        if request.stream:
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )

        start_time = time.time()
        litellm_response = await litellm.acompletion(**litellm_request)
        logger.debug(
            f"Response received from {litellm_request['model']} "
            f"in {time.time() - start_time:.2f}s"
        )
        return convert_litellm_to_anthropic(litellm_response, request)
    except Exception as exc:
        return error_response(exc)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    model = map_model_name(request.model)
    messages = convert_messages(request.messages, request.system)

    log_request(
        "POST",
        raw_request.url.path,
        request.model,
        model,
        len(messages),
        len(request.tools or []),
    )

    try:
        token_count = litellm.token_counter(model=model, messages=messages)
        return TokenCountResponse(input_tokens=token_count)
    except Exception as exc:
        logger.warning(f"Falling back to an approximate token count: {exc}")
        characters = sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)
        return TokenCountResponse(input_tokens=max(1, characters // 4))


CLAUDE_MODEL_ALIASES = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]


@app.get("/v1/models")
async def list_models():
    """Claude Code probes this endpoint when it talks to an LLM gateway."""
    return {
        "data": [
            {
                "type": "model",
                "id": alias,
                "display_name": alias,
                "created_at": "2025-01-01T00:00:00Z",
            }
            for alias in CLAUDE_MODEL_ALIASES
        ],
        "has_more": False,
        "first_id": CLAUDE_MODEL_ALIASES[0],
        "last_id": CLAUDE_MODEL_ALIASES[-1],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": "microsoft-foundry" if USE_AZURE else "openai",
        "endpoint": (AZURE_API_BASE if USE_AZURE else OPENAI_BASE_URL)
        or "https://api.openai.com/v1",
        "api_key_configured": bool(AZURE_API_KEY if USE_AZURE else OPENAI_API_KEY),
        "models": {
            "opus": BIG_MODEL,
            "sonnet": MIDDLE_MODEL,
            "haiku": SMALL_MODEL,
        },
        "reasoning_effort": REASONING_EFFORT,
    }


@app.get("/")
async def root():
    return {
        "message": "Claude Code to GPT proxy",
        "endpoints": [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/models",
            "/health",
        ],
    }


# --------------------------------------------------------------------------- #
# Pretty logging
# --------------------------------------------------------------------------- #


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log_request(method, path, claude_model, gpt_model, num_messages, num_tools):
    """Log the Claude -> GPT mapping of every request."""
    gpt_display = gpt_model.split("/")[-1]
    print(f"{Colors.BOLD}{method} {path}{Colors.RESET}")
    print(
        f"{Colors.CYAN}{claude_model}{Colors.RESET} → "
        f"{Colors.GREEN}{gpt_display}{Colors.RESET} "
        f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET} "
        f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    if not (AZURE_API_KEY if USE_AZURE else OPENAI_API_KEY):
        key_name = "AZURE_API_KEY" if USE_AZURE else "OPENAI_API_KEY"
        print(f"{key_name} is not set. Configure it in .env before starting.")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="error")
