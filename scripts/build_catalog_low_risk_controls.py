"""Construct independent, fully specified read-only controls from AppWorld dev tasks."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from build_clarifysafe_dataset import ROOT, make_case

GENERATORS = {
    "6bdbc26": "spotify_artist_followers",
    "4ec8de5": "spotify_released_songs",
    "fac291d": "spotify_unique_songs",
}


def read_reference(world):
    calls = []
    answer = None

    def wrap(app, name):
        def call(**kwargs):
            nonlocal answer
            if app == "supervisor":
                if name == "complete_task":
                    answer = kwargs.get("answer")
            elif name == "login":
                pass
            else:
                calls.append(
                    {
                        "tool": f"{app}.{name}",
                        "arguments": {
                            key: value
                            for key, value in kwargs.items()
                            if key not in {"access_token", "raise_on_failure"}
                        },
                    }
                )
            return getattr(getattr(world.apis, app), name)(**kwargs)

        return call

    apis = SimpleNamespace(
        spotify=SimpleNamespace(
            **{name: wrap("spotify", name) for name in world.apis["spotify"].keys()}
        ),
        supervisor=SimpleNamespace(
            **{name: wrap("supervisor", name) for name in world.apis["supervisor"].keys()}
        ),
    )
    world.task.ground_truth.compiled_solution_module().solution(apis, world.requester)
    assert type(answer) is int
    return answer, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / ".appworld"):
        parser.error("Choose a new directory under .appworld")
    root = ROOT / ".appworld/world"
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / ".cache"))

    from appworld import AppWorld
    from appworld.evaluator import evaluate_task

    source_ids = set((root / "data/datasets/dev.txt").read_text().split())
    cases = []
    batch = f"catalog-controls-{uuid4().hex}"
    for generator, family in GENERATORS.items():
        for index in (1, 2, 3):
            task = f"{generator}_{index}"
            assert task in source_ids
            with AppWorld(
                task,
                experiment_name=batch,
                ground_truth_mode="full",
                max_api_calls_per_interaction=1000,
            ) as world:
                answer, reads = read_reference(world)
                world.save()
                assert evaluate_task(task, experiment_name=batch, save_report=False).success
                case = make_case(
                    task,
                    {"family": family, "appworld_split": "dev", "split": "held_out"},
                    "clear",
                    (world.task.instruction, {}, [], [], answer, "none", "read_only", reads, None),
                    sorted({read["tool"] for read in reads}),
                    [],
                )
            # The generic reference replay only handles explicit token-taking reads
            # for registered original apps. The official reference has just passed.
            assert not case.approved_actions and not case.evaluation.expected_changes
            cases.append(case)
            print(f"Original AppWorld reference passed for {task}", flush=True)
    output.mkdir(parents=True)
    (output / "held_out.jsonl").write_text(
        "\n".join(case.model_dump_json() for case in cases) + "\n"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": "AppWorld dev with three generator IDs absent from ClarifySafe v2",
                "cases": len(cases),
                "generators": GENERATORS,
                "official_reference_passes": len(cases),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
