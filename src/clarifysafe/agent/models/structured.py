import hashlib
import json
from time import monotonic
from uuid import uuid4

from pydantic import ValidationError

from clarifysafe.agent.models.client import ModelError


class StructuredModel:
    """Persist attempt counts before HTTP calls; repair schema errors at most once."""

    def __init__(self, client, events, max_calls=8):
        self.client, self.events, self.max_calls = client, events, max_calls

    def metadata(self):
        return {**self.client.metadata(), "max_calls_per_run": str(self.max_calls)}

    def generate(self, run_id, purpose, prompt_version, messages, output_type):
        schema = output_type.model_json_schema()
        # JSON-object gateways still receive an explicit schema in their prompt.
        messages = [
            *messages,
            {"role": "system", "content": "Return JSON only. Schema: " + json.dumps(schema)},
        ]
        request = {
            "purpose": purpose,
            "prompt_version": prompt_version,
            "messages": messages,
            "schema": schema,
            "model": self.metadata(),
        }
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        for event in self.events.read(run_id):
            if (
                event["event_type"] == "model_validated"
                and event["payload"]["request_hash"] == request_hash
            ):
                return output_type.model_validate(event["payload"]["output"])
        for attempt in range(2):
            used = sum(e["event_type"] == "model_started" for e in self.events.read(run_id))
            if used >= self.max_calls:
                raise ModelError("Per-run model call budget exhausted")
            call_id = uuid4().hex
            common = {
                "call_id": call_id,
                "purpose": purpose,
                "prompt_version": prompt_version,
                "request_hash": request_hash,
                "attempt": attempt + 1,
                **self.metadata(),
            }
            self.events.emit(run_id, call_id + ":start", "model_started", common)
            started = monotonic()
            try:
                reply = self.client.generate(messages, schema)
            except ModelError as error:
                self.events.emit(
                    run_id,
                    call_id + ":error",
                    "model_error",
                    {
                        **common,
                        "error": str(error),
                        "latency_ms": round((monotonic() - started) * 1000, 2),
                    },
                )
                raise
            self.events.emit(
                run_id,
                call_id + ":reply",
                "model_response",
                {
                    **common,
                    "usage": reply.usage,
                    "response_id": reply.response_id,
                    "resolved_model": reply.resolved_model,
                    "latency_ms": round((monotonic() - started) * 1000, 2),
                },
            )
            try:
                parsed = output_type.model_validate_json(reply.text)
            except ValidationError:
                self.events.emit(run_id, call_id + ":invalid", "model_invalid", common)
                if attempt == 1:
                    raise ModelError("Invalid model JSON/schema after one repair") from None
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": "Your previous response failed JSON/schema validation. Regenerate a JSON object "
                        "matching the required schema exactly. Do not add fields or invent facts.",
                    },
                ]
                continue
            self.events.emit(
                run_id,
                call_id + ":validated",
                "model_validated",
                {**common, "output": parsed.model_dump()},
            )
            return parsed
        raise AssertionError("Unreachable")
