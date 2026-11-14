"""Recover provider-rate-limited cases from fresh AppWorld state and retain original logs."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

from run_clarifysafe_experiments import BASE, ROOT, finish, launcher, lines, sha

DATASETS = {
    "v2-formal-main-10": "clarifysafe-interactive-v2",
    "v2-formal-budget-20": "clarifysafe-interactive-v2",
    "v2-official-funcall-20": "clarifysafe-interactive-v2",
    "v2-verifier-ablation-20": "clarifysafe-interactive-v2",
    "risk-controls-20": "clarifysafe-risk-controls",
    "low-risk-controls-20": "clarifysafe-low-risk-controls",
}


def was_rate_limited(row):
    return row["result"]["status"] == "stopped" and any(
        "RateLimitError" in event["payload"].get("error", "")
        for event in row["trace"]
        if event["event_type"] == "model_error"
    )


def redo(job):
    batch, dataset, manifest, system, case, max_attempts = job
    case_id = case["case_id"]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONWARNINGS": "ignore",
        **manifest["model_settings"],
    }
    attempts_folder = batch / "rate-limit-recovery" / system / case_id
    attempts_folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, max_attempts + 1):
        attempt = attempts_folder / f"attempt-{index}"
        path = attempt / "results.jsonl"
        ran_attempt = not path.is_file() or not path.stat().st_size
        if ran_attempt:
            if attempt.exists():
                preserved = attempt.with_name(f"{attempt.name}-incomplete-{os.getpid()}")
                attempt.rename(preserved)
            command = launcher(system) + [
                "--split",
                "held_out",
                "--dataset",
                str(dataset),
                "--output",
                str(attempt),
                "--max-calls",
                str(manifest["max_model_calls_per_case"]),
                "--case",
                case_id,
            ]
            with attempt.with_suffix(".log").open("a") as log:
                print(f"Retry {system} {case_id} ({index}/{max_attempts})", flush=True)
                subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        (row,) = lines(path)
        assert row["case_id"] == case_id and row["agent"] == system
        if not was_rate_limited(row):
            return system, case_id, row
        if index < max_attempts and ran_attempt:
            sleep(min(25 * index, 60))
    return system, case_id, None


def atomic_write_rows(path, rows):
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        temp = Path(handle.name)
    temp.replace(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("batches", choices=DATASETS, nargs="+")
    cli.add_argument("--workers", type=int, choices=(1, 2), default=2)
    cli.add_argument("--attempts", type=int, choices=range(1, 10), default=3)
    args = cli.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    for name in args.batches:
        batch = BASE / name
        dataset = ROOT / ".appworld" / DATASETS[name]
        manifest = json.loads((batch / "manifest.json").read_text())
        codes = json.loads((batch / "worker_exit_codes.json").read_text())
        assert not any(codes.values()), codes
        case_file = dataset / "held_out.jsonl"
        assert sha(case_file) == manifest["dataset_sha256"]
        assert all(sha(ROOT / path) == digest for path, digest in manifest["source_sha256"].items())
        cases = lines(case_file)
        by_case = {case["case_id"]: case for case in cases}
        raw_paths = {}
        jobs = []
        for system in manifest["systems"]:
            for family, expected in manifest["family_counts"].items():
                shard = batch / f"{system}-{family}"
                assert (shard / "summary.json").is_file()
                path = shard / "results.jsonl"
                rows = lines(path)
                assert len(rows) == expected
                raw_paths[system, family] = path
                jobs.extend(
                    (batch, dataset, manifest, system, by_case[row["case_id"]], args.attempts)
                    for row in rows
                    if was_rate_limited(row)
                )
        print(f"{name}: {len(jobs)} provider-limited records to retry.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            replacements = {}
            failed = []
            for system, case_id, row in pool.map(redo, jobs):
                if row is None:
                    failed.append((system, case_id))
                else:
                    replacements[system, case_id] = row
        for (system, _), path in raw_paths.items():
            rows = lines(path)
            if not any((system, row["case_id"]) in replacements for row in rows):
                continue
            backup = path.with_name("results.original.jsonl")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            atomic_write_rows(
                path, [replacements.get((system, row["case_id"]), row) for row in rows]
            )
        recovery = batch / "rate-limit-recovery.json"
        recovery.write_text(
            json.dumps(
                {
                    "rate_limited_records": len(jobs),
                    "recovered_records": len(replacements),
                    "unresolved": failed,
                    "retry_rule": "Only terminal RateLimitError; fresh AppWorld state; first unaffected result.",
                },
                indent=2,
            )
            + "\n"
        )
        if failed:
            raise SystemExit(f"{name}: {len(failed)} cases still rate limited.")
        if name == "v2-formal-budget-20":
            subprocess.run(
                [sys.executable, str(batch / "summarize.py")],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                check=True,
            )
        else:
            finish(batch, manifest, cases)


if __name__ == "__main__":
    main()
