"""Policy enforcement point and a durable ledger around state-changing calls."""

import json
import time
from uuid import uuid4

from clarifysafe.agent.safety import fingerprint
from clarifysafe.shared.schemas import Approval


def operation_id(state):
    return f"{state.run_id}:step-{state.step_index}:{fingerprint(state.action)}"


def issue_approval(state, policy, approved, now=None):
    now = time.time() if now is None else now
    return Approval(
        approved=approved,
        action_fingerprint=fingerprint(state.action),
        approval_id=uuid4().hex,
        run_id=state.run_id,
        operation_id=operation_id(state),
        policy_digest=state.policy_digest,
        expires_at=float(now + policy.config.approval_ttl_seconds),
    )


class GuardedExecutor:
    def __init__(self, db, policy, events, clock=time.time):
        self.db, self.policy, self.events, self.clock = db, policy, events, clock
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS operations ("
            "operation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
            "status TEXT NOT NULL, result TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS consumed_approvals ("
            "approval_id TEXT PRIMARY KEY, operation_id TEXT UNIQUE NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS batch_runs (run_id TEXT PRIMARY KEY, plan_digest TEXT NOT NULL)"
        )
        self.db.commit()

    def check_authorization(self, state):
        approval = state.approval
        if (
            approval is None
            or not approval.approved
            or not approval.approval_id
            or approval.run_id != state.run_id
            or approval.operation_id != operation_id(state)
            or approval.action_fingerprint != fingerprint(state.action)
            or approval.policy_digest != state.policy_digest
            or approval.expires_at is None
            or approval.expires_at <= self.clock()
        ):
            raise ValueError("Missing, expired or mismatched operation-scoped approval")

    def check_preconditions(self, state, env):
        if getattr(env, "supports_batch_operations", False):
            if env.observe_operation(state.action) != state.before:
                raise ValueError("Operation target changed before execution")
            return
        if state.action.tool == "venmo.create_transaction":
            if env.observe_state() != state.before:
                raise ValueError("Payment evidence changed before execution")
            return
        if state.action.tool == "phone.send_text_message":
            from clarifysafe.agent.phone_contract import check_phone_preconditions

            check_phone_preconditions(state, env)
            return
        if state.action.tool == "simple_note.update_note":
            from clarifysafe.agent.note_contract import check_note_preconditions

            check_note_preconditions(state, env)
            return
        # Application-specific contract: only this bounded write is implemented.
        if state.action.tool != "send_message":
            raise ValueError("No execution contract registered for this tool")
        args = state.action.arguments
        if set(args) != {"recipient_id", "content"} or not args["content"].strip():
            raise ValueError("Invalid message contract")
        candidate = next(
            (
                c
                for c in state.candidates
                if c.object_id == args["recipient_id"] and c.source in {"application", "user"}
            ),
            None,
        )
        if candidate is None:
            raise ValueError("Recipient is not supported by evidence")
        contacts = env.call_read_tool("search_contacts", {"query": candidate.label})
        if not any(
            c["id"] == candidate.object_id and c["name"] == candidate.label for c in contacts
        ):
            raise ValueError("Recipient evidence changed; start a new task")
        if env.call_read_tool("read_minutes", {}) != args["content"]:
            raise ValueError("Source content changed; start a new task")
        if env.observe_state() != state.before:
            raise ValueError("Application state changed; start a new task")

    def execute(self, state, env, *, require_confirmation=False):
        tools = env.get_tool_schemas()
        if state.policy_digest != self.policy.digest(tools):
            raise ValueError("Policy or tool registration changed; start a new task")
        assessment = self.policy.assess(state.action, tools)
        if state.action is None or assessment.decision in {"ask", "decline"}:
            raise ValueError("Policy denied execution: " + "; ".join(assessment.reasons))
        if next(t for t in tools if t.name == state.action.tool).effect != "write":
            raise ValueError("Write executor cannot dispatch a read tool")
        needs_approval = require_confirmation or assessment.decision == "confirm"
        if needs_approval:
            self.check_authorization(state)
        op_id = operation_id(state)
        existing = self.db.execute(
            "SELECT status,result FROM operations WHERE operation_id=?", (op_id,)
        ).fetchone()
        if existing:
            if existing[0] == "returned":
                return json.loads(existing[1])
            raise ValueError("Operation outcome is unknown; reconciliation required, no retry")
        self.check_preconditions(state, env)
        if needs_approval:
            self.check_authorization(state)
        self.events.emit(
            state.run_id,
            f"guard_passed:{op_id}",
            "guard_passed",
            {
                "operation_id": op_id,
                "policy_digest": state.policy_digest,
                "assessment": assessment.model_dump(),
                "checks": {
                    "schema": True,
                    "permission": True,
                    "target_evidence": True,
                    "source_content_current": True,
                    "pre_state_unchanged": True,
                    "approval_required": needs_approval,
                },
            },
        )
        # Claim and authorization consumption are atomic locally, not with an external API.
        with self.db:
            self.db.execute(
                "INSERT INTO operations VALUES(?,?,?,NULL)", (op_id, state.run_id, "in_flight")
            )
            if needs_approval:
                self.db.execute(
                    "INSERT INTO consumed_approvals VALUES(?,?)",
                    (state.approval.approval_id, op_id),
                )
        self.events.emit(
            state.run_id,
            f"tool_started:{op_id}",
            "tool_started",
            {"operation_id": op_id, "action": state.action.model_dump()},
        )
        try:
            result = env.call_write_tool(state.action, op_id)
            encoded = json.dumps(result)
            with self.db:
                self.db.execute(
                    "UPDATE operations SET status='returned',result=? WHERE operation_id=?",
                    (encoded, op_id),
                )
        except Exception:
            with self.db:
                self.db.execute(
                    "UPDATE operations SET status='unknown' WHERE operation_id=?", (op_id,)
                )
            raise
        return result
