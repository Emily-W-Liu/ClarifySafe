import argparse
import json

from clarifysafe.agent.policy import PolicyConfig
from clarifysafe.agent.runtime import open_runtime
from clarifysafe.shared.schemas import AgentInput, UserResponse


def main():
    parser = argparse.ArgumentParser(
        description="ClarifySafe P3: policy-guarded reasoning over local mock tools"
    )
    parser.add_argument("--data-dir", default=".clarifysafe")
    parser.add_argument("--policy", help="Trusted TOML policy file; omit for built-in defaults")
    parser.add_argument("--mode", choices=["scripted", "llm"], default="scripted")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("instruction")
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("intervention_id")
    group = resume.add_mutually_exclusive_group(required=True)
    group.add_argument("--choice")
    group.add_argument("--text")
    group.add_argument("--approve", action="store_true")
    group.add_argument("--deny", action="store_true")
    for name in ("status", "events"):
        sub.add_parser(name).add_argument("run_id")
    sub.add_parser("demo")
    serve = sub.add_parser("serve", help="Start the local UI integration API")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="Allowed UI origin, e.g. http://localhost:5173; repeat if needed",
    )
    args = parser.parse_args()
    if args.command == "demo" and args.mode == "llm":
        parser.error("demo is scripted only; use start/resume for explicit live model interaction")
    try:
        config = PolicyConfig.from_file(args.policy) if args.policy else None
        if args.command == "serve":
            import uvicorn

            from clarifysafe.api import create_app

            app = create_app(
                args.data_dir,
                mode="llm_mock" if args.mode == "llm" else "scripted_mock",
                policy_config=config,
                allowed_origins=args.allow_origin,
            )
            uvicorn.run(app, host="127.0.0.1", port=args.port)
            return
        with open_runtime(args.data_dir, policy_config=config) as runtime:
            if args.command in {"start", "demo"}:
                result = runtime.start(
                    AgentInput(
                        instruction=args.instruction
                        if args.command == "start"
                        else "把会议纪要发给 Alex"
                    ),
                    mode="llm_mock" if args.mode == "llm" else "scripted_mock",
                )
                if args.command == "demo":
                    result = runtime.resume(
                        result.run_id,
                        UserResponse(
                            intervention_id=result.pending["intervention_id"], choice="alex-a"
                        ),
                    )
                    result = runtime.resume(
                        result.run_id,
                        UserResponse(
                            intervention_id=result.pending["intervention_id"], approved=True
                        ),
                    )
            elif args.command == "resume":
                result = runtime.resume(
                    args.run_id,
                    UserResponse(
                        intervention_id=args.intervention_id,
                        choice=args.choice,
                        text=args.text,
                        approved=True if args.approve else False if args.deny else None,
                    ),
                )
            elif args.command == "status":
                result = runtime.result(args.run_id)
            else:
                print(json.dumps(runtime.events.read(args.run_id), ensure_ascii=False, indent=2))
                return
            print(result.model_dump_json(indent=2))
    except ValueError as error:
        parser.exit(2, f"Invalid request: {error}\n")


if __name__ == "__main__":
    main()
