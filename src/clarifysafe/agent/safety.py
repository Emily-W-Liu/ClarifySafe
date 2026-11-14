import hashlib
import json

from clarifysafe.shared.schemas import Action, TaskState, VerificationResult


def fingerprint(action: Action) -> str:
    canonical = json.dumps(action.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_message(state: TaskState, after: list[dict[str, str]]) -> VerificationResult:
    assert state.action is not None
    old = {r["operation_id"]: r for r in state.before}
    new = {r["operation_id"]: r for r in after}
    added = [row for key, row in new.items() if key not in old]
    checks = {
        "prior_records_unchanged": all(new.get(key) == value for key, value in old.items()),
        "exactly_one_new_record": len(added) == 1,
        "correct_target_and_content": len(added) == 1
        and all(added[0][key] == value for key, value in state.action.arguments.items()),
    }
    return VerificationResult(passed=all(checks.values()), checks=checks)
