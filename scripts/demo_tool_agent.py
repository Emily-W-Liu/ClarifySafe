"""Run the tool-catalog agent on a local AppWorld instance with a human user."""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from clarifysafe.agent.executor import GuardedExecutor
from clarifysafe.agent.models.client import ModelConfig, OpenAICompatibleClient
from clarifysafe.agent.tool_agent import run_tool_agent
from clarifysafe.environment.appworld_tools import AppWorldTools
from clarifysafe.environment.mock import EventStore

ROOT = Path(__file__).resolve().parents[1]


def confirm(actions, snapshots):
    print("待批准的操作")
    for index, (action, snapshot) in enumerate(zip(actions, snapshots, strict=True), 1):
        resource = snapshot["resource"]
        if action.tool.split(".")[1].startswith("delete_"):
            fields = ("title", "label", "sender", "receiver", "message")
        else:
            fields = (*action.arguments.keys(), "title", "label", "name")
        current = {key: resource[key] for key in fields if key in resource}
        print(f"{index}. {action.model_dump_json()}")
        print("   当前", json.dumps(current, ensure_ascii=False))
    return input("输入 yes 批准，其他回答拒绝\n> ").strip().casefold() == "yes"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path, default=ROOT / ".appworld/world")
    parser.add_argument("--task-id", default="cf6abd2_1", help="AppWorld initial state to load")
    parser.add_argument("--instruction", help="Your task, in your own words")
    args = parser.parse_args()
    root = args.world_root.resolve()
    if not (root / "data/api_docs/openapi").is_dir():
        parser.error("AppWorld API docs are missing. Install AppWorld data or pass --world-root.")
    os.environ.update(
        APPWORLD_ROOT=str(root),
        APPWORLD_CACHE=str(root / ".cache"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    from appworld import AppWorld
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    config = ModelConfig.from_env()
    instruction = args.instruction or input("你希望 Agent 做什么？\n> ")
    run_id = uuid4().hex
    output = ROOT / ".appworld" / "results" / f"tool-agent-{run_id}"
    output.mkdir(parents=True)
    with (
        AppWorld(
            args.task_id, experiment_name=f"clarifysafe-tools-{run_id}", load_ground_truth=False
        ) as world,
        closing(OpenAICompatibleClient(config)) as model,
        closing(sqlite3.connect(output / "runs.sqlite")) as db,
    ):
        env = AppWorldTools(world, root / "data" / "api_docs")
        events = EventStore(db)
        try:
            result, answer = run_tool_agent(
                run_id,
                instruction,
                env,
                GuardedExecutor(db, env.policy(), events),
                model,
                lambda question: input(f"{question}\n> "),
                confirm,
                max_calls=config.max_calls_per_run,
                report_infeasible=lambda report: print(report["reason"]),
            )
            print(result.model_dump_json())
            if answer is not None:
                print(answer)
        except (EOFError, KeyboardInterrupt):
            print("\n交互已中断。")
        finally:
            (output / "trace.json").write_text(
                json.dumps(events.read(run_id), ensure_ascii=False, indent=2) + "\n"
            )
            print("审计记录", output / "trace.json")


if __name__ == "__main__":
    main()
