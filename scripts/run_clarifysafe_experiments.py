"""Run or summarize matched ClarifySafe follow-up experiments, saving every shard."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from clarifysafe.agent.models.client import ModelConfig
from clarifysafe.evaluation.interactive_dataset import summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".appworld/results"
PLANS = {
    "main-20": ("clarifysafe-interactive-v2", ("catalog", "official-react"), "v2-formal-budget-20"),
    "budget-10": ("clarifysafe-interactive-v2", ("catalog", "official-react"), "v2-formal-main-10"),
    "funcall": ("clarifysafe-interactive-v2", ("official-funcall",), "v2-official-funcall-20"),
    "verifier": ("clarifysafe-interactive-v2", ("no-verifier",), "v2-verifier-ablation-20"),
    "risk": ("clarifysafe-risk-controls", ("catalog", "no-risk"), "risk-controls-20"),
    "low-risk": (
        "clarifysafe-low-risk-controls",
        ("catalog", "official-react"),
        "low-risk-controls-20",
    ),
}
FOLLOWUPS = {"official-funcall", "no-risk", "no-verifier"}


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launcher(system):
    script = (
        "run_clarifysafe_followup.py" if system in FOLLOWUPS else "run_clarifysafe_evaluation.py"
    )
    flag = "--experiment-mode" if system in FOLLOWUPS else "--agent"
    return [sys.executable, str(ROOT / "scripts" / script), flag, system]


def run(batch, dataset, manifest):
    batch.mkdir()
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONWARNINGS": "ignore",
        **manifest["model_settings"],
    }
    workers = {}
    for system in manifest["systems"]:
        for family in manifest["family_counts"]:
            name = f"{system}-{family}"
            command = launcher(system) + [
                "--split",
                "held_out",
                "--dataset",
                str(dataset),
                "--output",
                str(batch / name),
                "--family",
                family,
                "--max-calls",
                str(manifest["max_model_calls_per_case"]),
            ]
            with (batch / f"{name}.log").open("x") as log:
                workers[name] = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            print(f"Started {name}", flush=True)
    codes = {}
    for name, worker in workers.items():
        codes[name] = worker.wait()
        print(f"Finished {name}: exit={codes[name]}", flush=True)
    (batch / "worker_exit_codes.json").write_text(json.dumps(codes, indent=2) + "\n")
    if any(codes.values()):
        raise SystemExit("At least one shard failed; inspect the worker logs.")


def counts(rows):
    def score(part):
        values = summarize(part)
        values["cases"] = values.pop("instances")
        return values

    events = [event for row in rows for event in row["trace"]]
    return {
        "totals": {
            **score(rows),
            "unauthorized_writes": sum(row["unauthorized_write_count"] for row in rows),
            "cases_with_unauthorized_writes": sum(
                bool(row["unauthorized_write_count"]) for row in rows
            ),
            "unnecessary_confirmation": sum(row["unnecessary_confirmation"] for row in rows),
        },
        "by_family": {
            group: score([row for row in rows if row["family"] == group])
            for group in sorted({row["family"] for row in rows})
        },
        "by_variant": {
            group: score([row for row in rows if row["variant"] == group])
            for group in sorted({row["variant"] for row in rows})
        },
        "model_calls": sum(event["event_type"] == "model_started" for event in events),
        "stopped_reasons": dict(
            Counter(row["result"]["reason"] for row in rows if row["result"]["status"] == "stopped")
        ),
        "provider_errors": dict(
            Counter(
                event["payload"]["error"]
                for event in events
                if event["event_type"] == "model_error"
            )
        ),
    }


def finish(batch, manifest, cases):
    systems = {}
    for system in manifest["systems"]:
        rows = []
        for family, expected in manifest["family_counts"].items():
            shard = batch / f"{system}-{family}"
            meta = json.loads((shard / "summary.json").read_text())
            assert meta["split"] == "held_out" and meta["agent"] == system
            assert meta["catalog_max_calls"] == manifest["max_model_calls_per_case"]
            assert meta["dataset_sha256"] == manifest["dataset_sha256"]
            assert meta["intent"]["model"] == manifest["model_settings"]["CLARIFYSAFE_MODEL"]
            subset = lines(shard / "results.jsonl")
            assert len(subset) == expected
            assert all(row["agent"] == system and row["family"] == family for row in subset)
            rows.extend(subset)
        assert len(rows) == len(cases)
        assert {row["case_id"] for row in rows} == {case["case_id"] for case in cases}
        (batch / f"{system}-results.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        )
        systems[system] = counts(rows)
    summary = {
        "split": "held_out",
        "model": manifest["model_settings"]["CLARIFYSAFE_MODEL"],
        "dataset_sha256": manifest["dataset_sha256"],
        "max_model_calls_per_case": manifest["max_model_calls_per_case"],
        "systems": systems,
    }
    (batch / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("plan", choices=PLANS)
    cli.add_argument("--summarize-only", action="store_true")
    args = cli.parse_args()
    dataset_name, systems, batch_name = PLANS[args.plan]
    dataset = ROOT / ".appworld" / dataset_name
    batch = BASE / batch_name
    case_file = dataset / "held_out.jsonl"
    cases = lines(case_file)
    if args.summarize_only:
        manifest = json.loads((batch / "manifest.json").read_text())
    else:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        model = ModelConfig.from_env()
        files = sorted((ROOT / "src" / "clarifysafe").rglob("*.py"))
        files.extend(sorted((ROOT / "scripts").glob("*.py")))
        files.append(ROOT / "config/clarifysafe-interactive-dataset-v2.json")
        upstream = ROOT / ".appworld/upstream/experiments"
        if any(system.startswith("official-") for system in systems):
            files.extend(
                upstream / relative for relative in (
                    "code/simplified/react_code_agent.py",
                    "code/simplified/function_calling_agent.py",
                    "prompts/react_code_agent/instructions.txt",
                    "prompts/function_calling_agent/instructions.txt",
                    "prompts/api_predictor.txt",
                )
            )
        manifest = {
            "systems": systems,
            "family_counts": dict(Counter(case["family"] for case in cases)),
            "case_ids": [case["case_id"] for case in cases],
            "dataset_sha256": sha(case_file),
            "max_model_calls_per_case": 10 if args.plan == "budget-10" else 20,
            "model_settings": {
                "CLARIFYSAFE_MODEL": model.model,
                "CLARIFYSAFE_BASE_URL": model.base_url,
                "CLARIFYSAFE_RESPONSE_FORMAT": model.response_format,
                "CLARIFYSAFE_TOKEN_PARAMETER": model.token_parameter,
                "CLARIFYSAFE_MAX_OUTPUT_TOKENS": str(model.max_output_tokens),
                "CLARIFYSAFE_TIMEOUT_SECONDS": str(model.timeout_seconds),
            },
            "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        }
        BASE.mkdir(parents=True, exist_ok=True)
        run(batch, dataset, manifest)
    assert sha(case_file) == manifest["dataset_sha256"]
    assert all(sha(ROOT / path) == digest for path, digest in manifest["source_sha256"].items())
    finish(batch, manifest, cases)


if __name__ == "__main__":
    main()
