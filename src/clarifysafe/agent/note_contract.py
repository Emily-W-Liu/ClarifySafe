"""Contract for changing exactly one existing checklist marker."""

import re

from clarifysafe.shared.schemas import VerificationResult


def change_marker(content, item, done):
    lines = content.splitlines(keepends=True)
    matching = [
        i
        for i, line in enumerate(lines)
        if re.sub(r"^\[[ xX]\]\s*", "", line.strip()) == item and re.match(r"^\[[ xX]\]", line)
    ]
    if len(matching) != 1:
        raise ValueError("Checklist item must identify exactly one existing line")
    index = matching[0]
    lines[index] = ("[x]" if done else "[ ]") + lines[index][3:]
    return "".join(lines)


def check_note_preconditions(state, env):
    args = state.action.arguments
    if set(args) != {"note_id", "content"}:
        raise ValueError("Invalid note arguments")
    if str(args["note_id"]) not in {
        c.object_id for c in state.candidates if c.source == "application"
    }:
        raise ValueError("Note ID lacks application evidence")
    before = next((n for n in state.before if n["note_id"] == args["note_id"]), None)
    if before is None:
        raise ValueError("Missing note pre-state")
    old_lines, new_lines = (
        before["content"].splitlines(keepends=True),
        args["content"].splitlines(keepends=True),
    )
    if len(old_lines) != len(new_lines):
        raise ValueError("Note contract allows only a checklist marker change")
    differences = [(old, new) for old, new in zip(old_lines, new_lines) if old != new]
    if len(differences) != 1:
        raise ValueError("Exactly one checklist marker must change")
    old, new = differences[0]
    if not re.match(r"^\[[ xX]\]", old) or not re.match(r"^\[[ xX]\]", new) or old[3:] != new[3:]:
        raise ValueError("Change exceeds checklist marker scope")
    if env.observe_state() != state.before:
        raise ValueError("Note state changed before execution")


def verify_note(state, after, *, target_only=False):
    target = state.action.arguments["note_id"]
    expected = {note["note_id"]: dict(note) for note in state.before}
    expected[target]["content"] = state.action.arguments["content"]
    actual = {note["note_id"]: dict(note) for note in after}
    # Updating a note may legitimately change its updated_at field.
    if target in actual:
        actual[target].pop("updated_at", None)
        expected[target].pop("updated_at", None)
    if target_only:
        checks = {"target_note_matches_approved_edit": actual.get(target) == expected[target]}
    else:
        checks = {"expected_content_and_no_other_note_changes": actual == expected}
    return VerificationResult(passed=all(checks.values()), checks=checks)
