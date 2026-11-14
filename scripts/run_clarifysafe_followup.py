"""Run official Function Calling or a single catalog ablation on the shared evaluator."""

import argparse
import json
import sys
from pathlib import Path

import run_clarifysafe_evaluation as runner

from clarifysafe.experiments.catalog_ablation import CatalogAblationTools
from clarifysafe.experiments.function_calling_development import run_official_function_calling


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiment-mode", choices=("official-funcall", "no-verifier", "no-risk"), required=True
    )
    args, delegated_arguments = parser.parse_known_args()
    if "--agent" in delegated_arguments:
        parser.error("The follow-up selects its own runner")
    mode = args.experiment_mode
    production_agent = runner.run_tool_agent
    if mode != "official-funcall":
        from clarifysafe.agent import tool_agent

        production_executor = tool_agent.execute_batch

        def experiment_environment(world, api_docs):
            return CatalogAblationTools(world, api_docs, mode)

        runner.AppWorldTools = experiment_environment

        if mode == "no-risk":

            def execute_without_risk(
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
                return production_executor(
                    run_id,
                    instruction,
                    plan,
                    environment,
                    executor,
                    confirm,
                    require_confirmation=bool(tool_agent.EXPLICIT_APPROVAL.search(instruction)),
                    prepared_state=prepared_state,
                )

            tool_agent.execute_batch = execute_without_risk

    def experiment_agent(
        run_id,
        instruction,
        env,
        executor,
        model,
        ask,
        confirm,
        *,
        max_calls,
        report_infeasible,
    ):
        if mode != "official-funcall":
            return production_agent(
                run_id,
                instruction,
                env,
                executor,
                model,
                ask,
                confirm,
                max_calls=max_calls,
                report_infeasible=report_infeasible,
            )
        return run_official_function_calling(
            run_id,
            instruction,
            env.connection.world,
            env,
            model,
            executor.events,
            ask,
            lambda actions: confirm(actions, None),
            max_calls=max_calls,
            report_infeasible=report_infeasible,
        )

    runner.run_tool_agent = experiment_agent
    production_run_case = runner.run_case

    def labeled_run_case(*args, **kwargs):
        record = production_run_case(*args, **kwargs)
        record["agent"] = mode
        return record

    runner.run_case = labeled_run_case
    sys.argv = [sys.argv[0], *delegated_arguments, "--agent", "catalog"]
    runner.main()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    destination, _ = output_parser.parse_known_args(delegated_arguments)
    summary_path = destination.output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["agent"] = mode
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
