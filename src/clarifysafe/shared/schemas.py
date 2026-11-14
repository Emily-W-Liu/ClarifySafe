from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentInput(Contract):
    instruction: str = Field(min_length=1, max_length=10000)


class UserKnowledge(Contract):
    """Private: held by the user simulator, never passed into the runtime."""

    facts: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(Contract):
    """Private: held by the offline evaluator."""

    expected_changes: list[dict[str, Any]] = Field(default_factory=list)
    required_interactions: list[str] = Field(default_factory=list)


class ToolRisk(Contract):
    """Trusted adapter metadata for a bounded tool, never supplied by the model."""

    externally_visible: bool
    reversible: bool
    affected_objects: int = Field(ge=1)
    required_permissions: list[str]


class RiskAssessment(Contract):
    decision: Literal["act", "ask", "confirm", "decline"]
    reasons: list[str]
    risk: ToolRisk | None = None


class ToolSpec(Contract):
    name: str
    description: str
    parameters: dict[str, Any]
    effect: Literal["read", "write"]
    risk: ToolRisk | None = None


class Candidate(Contract):
    object_id: str
    label: str
    source: Literal["user", "application", "model"]


class Action(Contract):
    tool: str
    arguments: dict[str, JsonValue]


class Approval(Contract):
    action_fingerprint: str
    approved: bool
    source: Literal["user"] = "user"
    approval_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None
    policy_digest: str | None = None
    expires_at: float | None = None


class VerificationResult(Contract):
    passed: bool
    checks: dict[str, bool]


class TaskState(Contract):
    run_id: str
    instruction: str
    mode: Literal["scripted_mock", "llm_mock", "llm_appworld"] = "scripted_mock"
    status: Literal["running", "waiting_for_user", "declined", "stopped", "completed"] = "running"
    decision: Literal["act", "ask", "confirm", "decline"] | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    action: Action | None = None
    approval: Approval | None = None
    before: list[dict[str, Any]] = Field(default_factory=list)
    verification: VerificationResult | None = None
    reason: str | None = None
    user_answers: list[str] = Field(default_factory=list)
    clarification_round: int = 0
    question: str | None = None
    minutes_content: str | None = None
    answer: str | None = None
    model_metadata: dict[str, str] = Field(default_factory=dict)
    policy_digest: str | None = None
    risk_assessment: RiskAssessment | None = None
    step_index: int = Field(default=1, ge=1)


class UserResponse(Contract):
    intervention_id: str
    choice: str | None = None
    approved: bool | None = None
    text: str | None = Field(default=None, min_length=1, max_length=2000)


class AgentEvent(Contract):
    sequence: int
    run_id: str
    event_type: str
    payload: dict[str, Any]


class AgentResult(Contract):
    run_id: str
    status: str
    mode: Literal["scripted_mock", "llm_mock", "llm_appworld"] = "scripted_mock"
    pending: dict[str, Any] | None = None
    state: TaskState
