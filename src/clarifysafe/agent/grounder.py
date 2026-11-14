import json
from typing import Literal

from pydantic import Field

from clarifysafe.agent.models.client import ModelError
from clarifysafe.agent.prompts import (
    INTENT_PROMPT,
    INTENT_VERSION,
    QUESTION_PROMPT,
    QUESTION_VERSION,
)
from clarifysafe.shared.schemas import Action, Candidate, Contract, VerificationResult


class GroundedIntent(Contract):
    intent: Literal["send_minutes", "read_minutes", "unsupported"]
    recipient_query: str | None
    recipient_quote: str | None
    reason: str


class ClarifyingQuestion(Contract):
    question: str = Field(min_length=1, max_length=500)


class Grounder:
    """P2 bounded domain: model extracts intent, code queries and resolves candidates."""

    def __init__(self, model, events):
        self.model, self.events = model, events

    def run(self, state, env):
        metadata = {
            **self.model.metadata(),
            "intent_prompt": INTENT_VERSION,
            "question_prompt": QUESTION_VERSION,
        }
        if state.model_metadata and state.model_metadata != metadata:
            raise ModelError("Model or prompt configuration changed; start a new run")
        state.model_metadata = metadata
        if state.clarification_round >= 3:
            state.status, state.reason = "stopped", "Clarification budget exhausted"
            return state
        tools = env.get_tool_schemas()
        required = {"search_contacts": "read", "read_minutes": "read", "send_message": "write"}
        actual = {tool.name: tool.effect for tool in tools}
        if any(actual.get(name) != effect for name, effect in required.items()):
            raise ModelError("Adapter does not provide the required typed minutes tools")
        inputs = {
            "instruction": state.instruction,
            "clarification_answers": state.user_answers,
            "tool_schemas": [t.model_dump() for t in tools],
        }
        intent = self.model.generate(
            state.run_id,
            "intent",
            INTENT_VERSION,
            [
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
            ],
            GroundedIntent,
        )
        self.events.emit(
            state.run_id,
            f"intent:{state.clarification_round}",
            "intent_grounded",
            {"round": state.clarification_round, **intent.model_dump()},
        )
        if intent.intent == "unsupported":
            state.status, state.decision = "declined", "decline"
            state.reason = "Outside the P2 minutes-app scope: " + intent.reason
            return state
        state.before = env.observe_state()
        state.minutes_content = env.call_read_tool("read_minutes", {})
        self.events.emit(
            state.run_id,
            f"minutes:{state.clarification_round}",
            "read_tool",
            {"tool": "read_minutes", "result": state.minutes_content},
        )
        if intent.intent == "read_minutes":
            if intent.recipient_query is not None or intent.recipient_quote is not None:
                raise ModelError("Unexpected recipient in read-only intent")
            state.answer = state.minutes_content
            unchanged = env.observe_state() == state.before
            state.verification = VerificationResult(
                passed=unchanged, checks={"no_state_changes": unchanged}
            )
            state.status = "completed" if unchanged else "stopped"
            state.decision = "act"
            state.reason = None if unchanged else "Unexpected modification during read-only task"
            return state
        query = intent.recipient_query
        if (query is None) != (intent.recipient_quote is None):
            raise ModelError("Recipient query and source quote must both be present or absent")
        if query is not None:
            if (
                not query.strip()
                or query != intent.recipient_quote
                or not any(query in value for value in [state.instruction, *state.user_answers])
            ):
                raise ModelError("Recipient query is not supported by an exact user quote")
            contacts = env.call_read_tool("search_contacts", {"query": query})
            # Prefer a full-name match over broad substring matches, without inventing an ID.
            exact = [c for c in contacts if c["name"].casefold() == query.casefold()]
            contacts = exact or contacts
        else:
            contacts = []
        state.candidates = [
            Candidate(object_id=c["id"], label=c["name"], source="application") for c in contacts
        ]
        self.events.emit(
            state.run_id,
            f"contacts:{state.clarification_round}",
            "read_evidence",
            {
                "tool": "search_contacts",
                "query": query,
                "performed": query is not None,
                "candidates": [c.model_dump() for c in state.candidates],
            },
        )
        if len(state.candidates) == 1:
            state.action = Action(
                tool="send_message",
                arguments={
                    "recipient_id": state.candidates[0].object_id,
                    "content": state.minutes_content,
                },
            )
            state.status, state.decision = "waiting_for_user", "confirm"
            return state
        evidence = {
            "instruction": state.instruction,
            "missing_field": "recipient",
            "candidates": [c.model_dump() for c in state.candidates],
        }
        question = self.model.generate(
            state.run_id,
            "question",
            QUESTION_VERSION,
            [
                {"role": "system", "content": QUESTION_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
            ],
            ClarifyingQuestion,
        )
        state.question = question.question
        state.status, state.decision = "waiting_for_user", "ask"
        return state
