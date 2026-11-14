import os
from dataclasses import dataclass, field
from typing import Literal, Protocol
from urllib.parse import urlparse

from openai import APIError, OpenAI
from pydantic import Field, SecretStr

from clarifysafe.shared.schemas import Contract


class ModelError(ValueError):
    """Sanitized error safe to persist. Never include raw provider error bodies."""


class ModelConfig(Contract):
    model: str = Field(min_length=1)
    api_key: SecretStr
    base_url: str = "https://api.openai.com/v1"
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    max_output_tokens: int = Field(default=800, ge=64, le=8192)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_calls_per_run: int = Field(default=8, ge=1, le=20)

    @classmethod
    def from_env(cls):
        model = os.getenv("CLARIFYSAFE_MODEL", "").strip()
        key = os.getenv("CLARIFYSAFE_API_KEY", "").strip()
        if not model or not key:
            raise ModelError("Set CLARIFYSAFE_MODEL and CLARIFYSAFE_API_KEY for --mode llm")
        try:
            config = cls(
                model=model,
                api_key=SecretStr(key),
                base_url=os.getenv("CLARIFYSAFE_BASE_URL", "https://api.openai.com/v1"),
                response_format=os.getenv("CLARIFYSAFE_RESPONSE_FORMAT", "json_schema"),
                token_parameter=os.getenv("CLARIFYSAFE_TOKEN_PARAMETER", "max_completion_tokens"),
                max_output_tokens=int(os.getenv("CLARIFYSAFE_MAX_OUTPUT_TOKENS", "800")),
                timeout_seconds=float(os.getenv("CLARIFYSAFE_TIMEOUT_SECONDS", "30")),
                max_calls_per_run=int(os.getenv("CLARIFYSAFE_MAX_CALLS", "8")),
            )
            parsed = urlparse(config.base_url)
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.scheme not in {"https", "http"}
            ):
                raise ValueError("Invalid URL")
            if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Use HTTPS for remote model endpoints")
            return config
        except ValueError:
            raise ModelError("Invalid CLARIFYSAFE model configuration; see .env.example") from None


@dataclass
class ModelReply:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    response_id: str | None = None
    resolved_model: str | None = None


class ModelClient(Protocol):
    def metadata(self) -> dict[str, str]: ...
    def generate(self, messages: list[dict[str, str]], output_schema: dict) -> ModelReply: ...


class OpenAICompatibleClient:
    """Chat Completions for HKU/other explicitly configured compatible gateways.

    SDK retries disabled: every HTTP attempt is accounted for by the harness.
    No fallback to another provider, format, model or tools endpoint.
    """

    def __init__(self, config: ModelConfig, http_client=None):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    def metadata(self):
        return {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "response_format": self.config.response_format,
            "max_output_tokens": str(self.config.max_output_tokens),
            "token_parameter": self.config.token_parameter,
            "timeout_seconds": str(self.config.timeout_seconds),
        }

    def generate(self, messages, output_schema):
        response_format = {"type": self.config.response_format}
        if self.config.response_format == "json_schema":
            response_format["json_schema"] = {
                "name": output_schema["title"],
                "strict": True,
                "schema": output_schema,
            }
        try:
            reply = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                response_format=response_format,
                **{self.config.token_parameter: self.config.max_output_tokens},
            )
        except APIError as error:
            raise ModelError(
                f"Provider request failed ({type(error).__name__}); no automatic retry"
            ) from None
        if not reply.choices:
            raise ModelError("Provider returned no choices")
        choice = reply.choices[0]
        if choice.message.refusal:
            raise ModelError("Provider refused the request")
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ModelError("Provider returned incomplete or empty output")
        usage = {}
        if reply.usage is not None:
            usage = {
                "input_tokens": reply.usage.prompt_tokens,
                "output_tokens": reply.usage.completion_tokens,
                "total_tokens": reply.usage.total_tokens,
            }
        return ModelReply(choice.message.content, usage, reply.id, reply.model)

    def close(self):
        self.client.close()
