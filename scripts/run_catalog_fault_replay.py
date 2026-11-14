"""Inject write failures at the current AppWorld catalog executor boundary."""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from clarifysafe.agent.batch import BatchPlan, execute_batch
from clarifysafe.agent.executor import GuardedExecutor
from clarifysafe.environment.mock import EventStore
from clarifysafe.evaluation.interactive_dataset import Case, apply_setup
from clarifysafe.experiments.catalog_ablation import CatalogAblationTools
from clarifysafe.shared.schemas import Action

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ("acknowledged_without_write", "committed_with_timeout")


class FaultyAppWorldTools(CatalogAblationTools):
    def __init__(self, world, docs, fault, controller):
        super().__init__(world, docs, controller)
        self.fault = fault
        self.attempts = 0

    def call_write_tool(self, action, operation_id):
        self.attempts += 1
        if self.attempts == 1:
            if self.fault == "acknowledged_without_write":
                return {"message": "Alarm updated successfully."}
            super().call_write_tool(action, operation_id)
            raise TimeoutError("Response timed out after the write")
        return super().call_write_tool(action, operation_id)


def run_fault(world, docs, case, fault, *, controller="catalog", single_write=False):
    apply_setup(world.models, case.setup)
    env = FaultyAppWorldTools(world, docs, fault, controller)
    actions = [Action.model_validate(action) for action in case.approved_actions]
    assert len(actions) > 1 and all(action.tool == "phone.update_alarm" for action in actions)
    if single_write:
        actions = actions[:1]
    plan = BatchPlan(actions=actions)
    before = env.observe_plan(actions)
    expected = env.advance_plan(before, actions[0]) if fault == "committed_with_timeout" else before
    with closing(sqlite3.connect(":memory:")) as db:
        events = EventStore(db)
        executor = GuardedExecutor(db, env.policy(), events)
        run_id = uuid4().hex
        outcomes = [
            execute_batch(
                run_id,
                case.agent_input.instruction,
                plan,
                env,
                executor,
                lambda _: True,
                require_confirmation=True,
                prepared_state=before,
            )
            for _ in range(2)
        ]
        after = env.observe_plan(actions)
        states = [row[0] for row in db.execute("SELECT status FROM operations")]
        trace = [
            {"event": event["event_type"], "step": event["payload"].get("step")}
            for event in events.read(run_id)
        ]
        correctly_handled = (
            env.attempts == 1
            and after == expected
            and all(outcome.status == "stopped" for outcome in outcomes)
            and all(outcome.completed_steps == 0 for outcome in outcomes)
            and states == ["unknown" if fault == "committed_with_timeout" else "returned"]
        )
        return {
            "case_id": case.case_id,
            "fault": fault,
            "controller": controller,
            "plan_size": len(actions),
            "confirmed": any(event["event"] == "batch_approved" for event in trace),
            "writes_sent": env.attempts,
            "first_write_observed": after != before,
            "states_match_fault_contract": after == expected,
            "operation_ledger": states,
            "first_result": outcomes[0].model_dump(),
            "same_run_retry_result": outcomes[1].model_dump(),
            "correctly_handled": correctly_handled,
            "trace": trace,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / ".appworld/world"
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / ".cache"))
    from appworld import AppWorld

    dataset = ROOT / ".appworld/clarifysafe-interactive-v2/held_out.jsonl"
    cases = [Case.model_validate_json(line) for line in dataset.read_text().splitlines()]
    cases = [case for case in cases if case.family == "alarms" and case.variant == "clear"]
    assert len(cases) == 3
    rows = []
    controllers = ("catalog", "no-verifier")
    for fault in FAULTS:
        for case in cases:
            for controller in controllers:
                with AppWorld(
                    case.base_task_id,
                    experiment_name=f"catalog-fault-{uuid4().hex}",
                    load_ground_truth=False,
                ) as world:
                    row = run_fault(
                        world,
                        root / "data/api_docs",
                        case,
                        fault,
                        controller=controller,
                        single_write=True,
                    )
                    rows.append(row)
                    print(case.case_id, fault, controller, row["correctly_handled"], flush=True)
    expected = {
        ("catalog", "acknowledged_without_write"): ("stopped", True),
        ("no-verifier", "acknowledged_without_write"): ("completed", False),
        ("catalog", "committed_with_timeout"): ("stopped", True),
        ("no-verifier", "committed_with_timeout"): ("stopped", True),
    }
    for row in rows:
        assert row["writes_sent"] == 1 and row["states_match_fault_contract"]
        assert row["same_run_retry_result"]["status"] == "stopped"
        assert (row["first_result"]["status"], row["correctly_handled"]) == expected[
            row["controller"], row["fault"]
        ]
    summary = {
        fault: {
            controller: {
                "cases": len(matched),
                "handled": sum(row["correctly_handled"] for row in matched),
                "claimed_completion": sum(
                    row["first_result"]["status"] == "completed" for row in matched
                ),
            }
            for controller in controllers
            if (
                matched := [
                    row
                    for row in rows
                    if row["fault"] == fault and row["controller"] == controller
                ]
            )
        }
        for fault in FAULTS
    }
    result = {
        "fixture": "Fixed approved plan, real local AppWorld state, no language model calls",
        "records": rows,
        "summary": summary,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
