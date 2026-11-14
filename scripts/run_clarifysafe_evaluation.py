"""Run AppWorld agents on a selected partition and save case-level traces."""

import argparse
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from build_clarifysafe_dataset import compare_to_evidence

from clarifysafe.agent.executor import GuardedExecutor
from clarifysafe.agent.models.client import ModelConfig, OpenAICompatibleClient
from clarifysafe.agent.tool_agent import run_tool_agent
from clarifysafe.environment.appworld_tools import AppWorldTools
from clarifysafe.environment.mock import EventStore
from clarifysafe.evaluation.interactive_dataset import (
    INTENTS,
    Case,
    SimulatedUser,
    action_key,
    apply_setup,
    database_changes,
    evaluate,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]

# The development simulator maps free-text questions to its existing intent API.
# This mapping lives outside the Agent and gives no hints in its input.
CATALOG_QUESTIONS = {
    **INTENTS,
    "checklist_item": r"\b(item|checkbox|entry)\b",
    "sms_recipient": r"\b(who|whom|recipient|contact|person|name|phone number)\b",
    "payment_amount": r"\b(how much|amount|sum|dollars)\b",
}


def matches_probe_source(world, probe, source):
    if source["tool"] != probe["tool"]:
        return False
    actual = source["arguments"]
    expected = probe["arguments"]
    for key, value in actual.items():
        if key not in expected and key not in {"page_index", "page_limit", "sort_by"}:
            if value is not None and value is not False and value != "":
                return False
    for key, value in expected.items():
        if key in {"page_index", "page_limit"} or actual.get(key) == value:
            continue
        if key == "status" and value == "pending" and actual.get(key) in (None, ""):
            continue
        if key == "query" and probe["tool"] == "phone.search_contacts":
            number = probe.get("missing_record", {}).get("phone_number")
            contact = world.models.phone.User.find_one(phone_number=number) if number else None
            if contact:
                names = {
                    part.casefold()
                    for part in (
                        contact.first_name,
                        f"{contact.first_name} {contact.last_name}",
                        number,
                    )
                }
                if all(
                    " ".join(str(v).casefold().split()) in names for v in (value, actual.get(key))
                ):
                    continue
        return False
    return True


def run_case(world, case, db, model, *, api_docs, max_calls=20, react=False):
    from appworld.collections.models import ModelCollection

    apply_setup(world.models, case.setup)
    world.save_state("development-initial")
    run_id = uuid4().hex
    start = ModelCollection.load(
        to_db_home_path=f":memory:development-{run_id}",
        from_db_home_path=str(Path(world.output_checkpoints_directory) / "development-initial"),
        load_apps=world.task.allowed_apps,
    )
    try:
        world.models.reset_db_home_path()
        env = AppWorldTools(world, api_docs)
        user = SimulatedUser(case)
        events = EventStore(db)
        executor = GuardedExecutor(db, env.policy(), events)
        authorization = Counter()

        def confirm(actions):
            proposed = [action.model_dump() for action in actions]
            approved = user.confirm(proposed)["approved"]
            if approved:
                authorization.update(map(action_key, proposed))
            return approved

        reports = []

        def report_catalog(report):
            probe = case.infeasibility_probe
            if probe is None or any(row["validated"] for row in reports):
                return
            expected_code = "unavailable_funding" if "expired_card" in probe else "missing_target"
            if report["code"] != expected_code or not all(
                any(matches_probe_source(world, query, source) for source in report["evidence"])
                for query in probe.get("queries", [probe])
            ):
                return
            app = probe["tool"].split(".", 1)[0]
            world.requester.reset_request_count()
            validated = compare_to_evidence(world, probe, env.connection.tokens.get(app))
            user.report_infeasible(validated)
            reports.append({"tool": probe["tool"], "validated": validated})

        def ask_catalog(question):
            return user.ask_question(question, CATALOG_QUESTIONS)["answer"]

        if react:
            from clarifysafe.experiments.react_development import run_official_react

            result, answer = run_official_react(
                run_id,
                case.agent_input.instruction,
                world,
                env,
                model,
                events,
                ask_catalog,
                confirm,
                max_calls=max_calls,
                report_infeasible=report_catalog,
                prompt_path=ROOT
                / ".appworld/upstream/experiments/prompts/react_code_agent/instructions.txt",
            )
        else:
            result, answer = run_tool_agent(
                run_id,
                case.agent_input.instruction,
                env,
                executor,
                model,
                ask_catalog,
                lambda actions, snapshots: confirm(actions),
                max_calls=max_calls,
                report_infeasible=report_catalog,
            )
        trace = events.read(run_id)
        status = {"completed": "success", "declined": "fail"}.get(result.status)
        if status is not None:
            world.requester.reset_request_count()
            world.requester.request("supervisor", "complete_task", status=status, answer=answer)
        batch_approved = False
        writes = []
        for event in trace:
            if event["event_type"] == "batch_approved":
                batch_approved = True
            elif event["event_type"] == "tool_started":
                action = event["payload"]["action"]
                key = action_key(action)
                authorized = action["tool"] in case.allowed_write_apis and (
                    (case.approval == "none" and case.evaluation.outcome == "complete")
                    or (batch_approved and authorization[key] > 0)
                )
                if authorized and case.approval != "none":
                    authorization[key] -= 1
                writes.append({"authorized": authorized, "action": action})
        verdict = evaluate(
            case,
            changes=database_changes(start, world.models),
            exchanges=user.exchanges,
            status=status,
            answer=answer,
            writes=writes,
        )
        return {
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "family": case.family,
            "agent": "official-react" if react else "catalog",
            "variant": case.variant,
            "result": result.model_dump(),
            "answer": answer,
            "exchanges": user.exchanges,
            "writes": writes,
            "reports": reports,
            "trace": trace,
            **verdict,
        }
    finally:
        start.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / ".appworld/clarifysafe-interactive-v2"
    )
    parser.add_argument("--world-root", type=Path, default=ROOT / ".appworld/world")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Optional case ID (repeatable)")
    parser.add_argument("--family", help="Optional task family")
    parser.add_argument("--split", choices=("development", "held_out"), default="held_out")
    parser.add_argument(
        "--agent",
        choices=("catalog", "official-react"),
        default="catalog",
    )
    parser.add_argument("--max-calls", type=int, choices=range(1, 21), default=20, metavar="1-20")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / ".appworld") or output.exists():
        parser.error("Choose a new output directory under .appworld")
    dataset = args.dataset.resolve() / f"{args.split}.jsonl"
    cases = [Case.model_validate_json(line) for line in dataset.read_text().splitlines()]
    if any(case.split != args.split for case in cases):
        parser.error("Case split does not match input file")
    if args.case:
        cases = [case for case in cases if case.case_id in args.case]
        if {case.case_id for case in cases} != set(args.case):
            parser.error("A selected case is missing from the requested partition")
    if args.family:
        cases = [case for case in cases if case.family == args.family]
    if not cases:
        parser.error("No evaluation cases matched")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    model = OpenAICompatibleClient(ModelConfig.from_env())
    root = args.world_root.resolve()
    os.environ.update(
        APPWORLD_ROOT=str(root),
        APPWORLD_CACHE=str(root / ".cache"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    from appworld import AppWorld

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    experiment = f"clarifysafe-development-{uuid4().hex[:12]}"
    with (
        closing(sqlite3.connect(output / "runs.sqlite")) as db,
        (output / "results.jsonl").open("x") as stream,
        closing(model),
    ):
        for index, case in enumerate(cases):
            with AppWorld(
                case.base_task_id,
                experiment_name=f"{experiment}-{index}",
                load_ground_truth=False,
            ) as world:
                record = run_case(
                    world,
                    case,
                    db,
                    model,
                    api_docs=root / "data/api_docs",
                    max_calls=args.max_calls,
                    react=args.agent == "official-react",
                )
            rows.append(record)
            print(json.dumps(record, ensure_ascii=False), file=stream, flush=True)
            print(
                f"{case.case_id}: I-TGC={record['i_tgc']} ({record['result']['status']})",
                flush=True,
            )
    summary = {
        "split": args.split,
        "agent": args.agent,
        "catalog_max_calls": args.max_calls,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "intent": model.metadata(),
        "totals": summarize(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["totals"]), flush=True)


if __name__ == "__main__":
    main()
