"""Deterministic policy decision point, independent of model prose."""

import hashlib
import json
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import Field

from clarifysafe.shared.schemas import Action, Contract, RiskAssessment, ToolSpec


class PolicyConfig(Contract):
    allowed_write_tools: list[str] = Field(default_factory=lambda: ["send_message"])
    granted_permissions: list[str] = Field(default_factory=lambda: ["messages.send"])
    confirm_external: bool = True
    confirm_irreversible: bool = True
    confirm_object_count: int = Field(default=2, ge=1)
    max_affected_objects: int = Field(default=10, ge=1)
    approval_ttl_seconds: int = Field(default=300, ge=1, le=3600)

    @classmethod
    def from_file(cls, path: str | Path):
        with Path(path).open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def digest(self, tools: list[ToolSpec]) -> str:
        payload = {
            "version": "p3-policy-v1",
            "config": self.config.model_dump(),
            "tools": [t.model_dump() for t in sorted(tools, key=lambda t: t.name)],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def assess(self, action: Action | None, tools: list[ToolSpec]) -> RiskAssessment:
        def result(decision, reasons, risk=None):
            return RiskAssessment(decision=decision, reasons=reasons, risk=risk)

        if action is None:
            return result("ask", ["No grounded action; resolve missing information first"])
        matches = [t for t in tools if t.name == action.tool]
        if len(matches) != 1:
            return result("decline", ["Unknown or duplicate tool registration"])
        tool = matches[0]
        try:
            Draft202012Validator.check_schema(tool.parameters)
            Draft202012Validator(tool.parameters).validate(action.arguments)
        except (SchemaError, ValidationError):
            return result("decline", ["Tool arguments or schema are invalid"])
        if tool.effect == "read":
            return result("act", ["Read-only tool"])
        risk = tool.risk
        if action.tool not in self.config.allowed_write_tools or risk is None:
            return result("decline", ["Write tool not allowed or missing trusted risk metadata"])
        if not set(risk.required_permissions) <= set(self.config.granted_permissions):
            return result("decline", ["Required permission is unavailable"], risk)
        if risk.affected_objects > self.config.max_affected_objects:
            return result("decline", ["Operation exceeds allowed object count"], risk)
        reasons = []
        if risk.externally_visible and self.config.confirm_external:
            reasons.append("Operation is externally visible")
        if not risk.reversible and self.config.confirm_irreversible:
            reasons.append("Operation cannot be reversed")
        if risk.affected_objects >= self.config.confirm_object_count:
            reasons.append("Operation affects multiple objects")
        return result("confirm" if reasons else "act", reasons or ["Within policy limits"], risk)
