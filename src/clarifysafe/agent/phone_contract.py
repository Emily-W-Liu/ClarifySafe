"""A bounded movie-recommendation SMS contract with application evidence."""

from pydantic import Field

from clarifysafe.shared.schemas import Action, TaskState, VerificationResult


class PhoneTaskState(TaskState):
    request_id: int | None = None
    director: str | None = None
    selected_titles: list[str] = Field(default_factory=list)


def movie_catalog(snapshot):
    notes = [n for n in snapshot["notes"] if n["title"].casefold() == "movie recommendations"]
    if len(notes) != 1:
        raise ValueError("Requires one evidenced Movie Recommendations note")
    movies = {}
    for block in notes[0]["content"].split("\n\n"):
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith("#"):
            continue
        metadata = {}
        for line in lines[1:]:
            key, separator, value = line.strip().removeprefix("- ").partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
        if "director" not in metadata:
            raise ValueError("Unsupported movie note format")
        title = lines[0].strip()
        movie = {"title": title, "director": metadata["director"]}
        if title in movies and movies[title] != movie:
            raise ValueError("Conflicting movie entries require clarification")
        movies[title] = movie
    if not movies:
        raise ValueError("Missing movie titles")
    return list(movies.values())


def grounded_reply(state):
    if len(state.before) != 1 or state.before[0].get("kind") != "phone_notes_snapshot":
        raise ValueError("Missing Phone pre-state")
    snapshot = state.before[0]
    candidates = [c for c in state.candidates if c.source == "application"]
    if len(candidates) != 1:
        raise ValueError("Requires one evidenced recipient")
    phone = candidates[0].object_id
    contacts = [c for c in snapshot["contacts"] if c["phone_number"] == phone]
    if len(contacts) != 1:
        raise ValueError("Recipient is not uniquely grounded in contacts")
    request = next(
        (m for m in snapshot["messages"] if m["text_message_id"] == state.request_id), None
    )
    if (
        request is None
        or request["sender"]["phone_number"] != phone
        or request["receiver"]["phone_number"] != snapshot["self_phone"]
    ):
        raise ValueError("Movie request does not belong to the intended incoming conversation")
    if not any(word in request["message"].casefold() for word in ("movie", "film")):
        raise ValueError("Source message is not a movie request")
    if not state.director or state.director.casefold() not in request["message"].casefold():
        raise ValueError("Director lacks an exact source in the incoming request")
    titles = [
        m["title"]
        for m in movie_catalog(snapshot)
        if m["director"].casefold() == state.director.casefold()
    ]
    if not titles:
        raise ValueError("No evidenced movies satisfy the requested director")
    return Action(
        tool="phone.send_text_message",
        arguments={"phone_number": phone, "message": ", ".join(titles)},
    ), titles


def check_phone_preconditions(state, environment):
    expected, titles = grounded_reply(state)
    if state.action != expected or state.selected_titles != titles:
        raise ValueError("Message differs from the grounded movie selection")
    if environment.observe_state() != state.before:
        raise ValueError("Phone or note evidence changed before execution")


def verify_phone(state, after):
    before = state.before[0]
    observed = after[0] if len(after) == 1 else {}
    old = {m["text_message_id"]: m for m in before["messages"]}
    messages = observed.get("messages", [])
    current = {m["text_message_id"]: m for m in messages}
    new = [m for key, m in current.items() if key not in old]
    checks = {
        "unique_message_ids": len(messages) == len(current),
        "existing_messages_unchanged": all(current.get(key) == value for key, value in old.items()),
        "notes_unchanged": observed.get("notes") == before["notes"],
        "contacts_unchanged": observed.get("contacts") == before["contacts"],
        "account_unchanged": observed.get("self_phone") == before["self_phone"],
        "exactly_one_correct_outgoing_message": len(new) == 1
        and new[0]["sender"]["phone_number"] == before["self_phone"]
        and new[0]["receiver"]["phone_number"] == state.action.arguments["phone_number"]
        and new[0]["message"] == state.action.arguments["message"],
    }
    return VerificationResult(passed=all(checks.values()), checks=checks)
