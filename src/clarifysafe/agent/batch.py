"""Approve one exact plan and execute its writes sequentially through the durable ledger."""

import sqlite3
from hashlib import sha256
from typing import Literal

from pydantic import Field

from clarifysafe.agent.executor import issue_approval
from clarifysafe.agent.safety import fingerprint
from clarifysafe.shared.schemas import Action, Contract, TaskState


class BatchPlan(Contract):
    actions: list[Action] = Field(min_length=1, max_length=32)

    def digest(self):
        return sha256(
            "\n".join(fingerprint(action) for action in self.actions).encode()
        ).hexdigest()


class BatchResult(Contract):
    status: Literal["completed", "declined", "stopped"]
    completed_steps: int = 0
    acknowledged_steps: int = 0
    reason: str | None = None


def execute_batch(
    run_id,
    instruction,
    plan,
    environment,
    executor,
    confirm,
    *,
    require_confirmation=False,
    prepared_state=None,
):
    """The environment independently checks each operation and observes its effect."""
    events, policy = executor.events, executor.policy
    approved_plan = plan.model_copy(deep=True)
    actions = approved_plan.actions
    tools = environment.get_tool_schemas()
    assessments = [policy.assess(action, tools) for action in actions]
    denied = next((risk for risk in assessments if risk.decision in {"decline", "ask"}), None)
    if denied:
        return BatchResult(status="stopped", reason="; ".join(denied.reasons))
    if len({fingerprint(action) for action in actions}) != len(actions):
        return BatchResult(status="stopped", reason="The plan repeats an operation")
    plan_digest = approved_plan.digest()
    events.emit(
        run_id, "batch_plan", "batch_plan", {"digest": plan_digest, **approved_plan.model_dump()}
    )
    # A resumed caller must use a new run and reconcile the prior ledger before any new write.
    # This applies even if the prior process exited before completing the plan.
    with executor.db:
        claim = executor.db.execute(
            "INSERT OR IGNORE INTO batch_runs (run_id, plan_digest) VALUES (?, ?)",
            (run_id, plan_digest),
        )
    if not claim.rowcount:
        return BatchResult(
            status="stopped", reason="This batch run already started. Inspect its ledger."
        )
    policy_digest = policy.digest(tools)
    needs_confirmation = require_confirmation or any(
        risk.decision == "confirm" for risk in assessments
    )
    try:
        observed_plan = environment.observe_plan(actions)
        if prepared_state is not None and observed_plan != prepared_state:
            raise ValueError("Application evidence changed before confirmation")
    except (KeyError, TypeError):
        return BatchResult(status="stopped", reason="Unexpected application data format")
    except (ValueError, OSError, TimeoutError) as error:
        return BatchResult(status="stopped", reason=str(error))
    approved_at = executor.clock()
    if needs_confirmation:
        preview = approved_plan.model_copy(deep=True)
        answer = confirm(preview.actions)
        if answer is not True:
            events.emit(run_id, "batch_denied", "batch_denied", {"digest": plan_digest})
            return BatchResult(status="declined", reason="User declined this exact plan")
        if preview.digest() != plan_digest:
            return BatchResult(
                status="stopped", reason="The action preview changed during confirmation"
            )
        approved_at = executor.clock()
        events.emit(run_id, "batch_approved", "batch_approved", {"digest": plan_digest})
    if plan.digest() != plan_digest or approved_plan.digest() != plan_digest:
        return BatchResult(status="stopped", reason="Plan changed after preview")

    completed = acknowledged = 0
    for index, action in enumerate(actions, start=1):
        try:
            approved_action = action.model_copy(deep=True)
            current = environment.observe_plan(actions)
            if current != observed_plan:
                raise ValueError("Application evidence changed after the plan was prepared")
            if plan.digest() != plan_digest or approved_plan.digest() != plan_digest:
                raise ValueError("Plan changed after approval")
            state = TaskState(
                run_id=run_id,
                instruction=instruction,
                mode="llm_appworld",
                action=approved_action,
                step_index=index,
                policy_digest=policy_digest,
                before=environment.operation_snapshot(approved_action, current),
            )
            if needs_confirmation:
                state.approval = issue_approval(state, policy, True, now=approved_at)
                if state.approval.expires_at <= executor.clock():
                    raise ValueError("Batch approval expired")
            result = executor.execute(state, environment, require_confirmation=needs_confirmation)
            verification = environment.verify_operation(state, result)
            expected = environment.advance_plan(observed_plan, approved_action)
            if hasattr(environment, "verify_plan_state") and not environment.verify_plan_state(
                actions, expected
            ):
                verification.checks["plan_scope_unchanged_except_expected_write"] = False
                verification.passed = False
            events.emit(
                run_id,
                f"batch_step:{index}",
                "batch_step",
                {"step": index, **verification.model_dump()},
            )
            if not verification.passed:
                raise ValueError("Operation result did not satisfy its execution contract")
            completed += 1
            acknowledged += int(verification.checks.get("receipt_only", False))
            observed_plan = expected
        except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
            reason = (
                "Unexpected application data format"
                if isinstance(error, (KeyError, TypeError))
                else str(error)
            )
            events.emit(
                run_id,
                "batch_stopped",
                "batch_stopped",
                {"step": index, "completed_steps": completed, "reason": reason},
            )
            return BatchResult(
                status="stopped",
                completed_steps=completed,
                acknowledged_steps=acknowledged,
                reason=reason,
            )
    return BatchResult(
        status="completed", completed_steps=completed, acknowledged_steps=acknowledged
    )
