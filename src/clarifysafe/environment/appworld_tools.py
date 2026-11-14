"""Expose AppWorld APIs as capabilities independent of the task or dataset."""

import json
import re
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from clarifysafe.agent.policy import PolicyConfig, PolicyEngine
from clarifysafe.environment.appworld_batch import AppWorldBatchEnvironment
from clarifysafe.shared.schemas import Action, ToolRisk, ToolSpec, VerificationResult

# Only fields whose patches have independently verifiable state effects are registered.
UPDATE_FIELDS = {
    "phone.update_alarm": {
        "time",
        "enabled",
        "label",
        "repeat_days",
        "snooze_minutes",
        "vibration",
    },
    "simple_note.update_note": {"title", "content", "pinned", "tags"},
    "spotify.update_playlist": {"title", "is_public"},
    "todoist.update_task": {"title", "description", "priority", "is_completed"},
    "todoist.update_project": {"name", "description", "is_archived"},
}

DELETE_TOOLS = {"phone.delete_text_message", "phone.delete_voice_message"}
REMINDER_TOOL = "venmo.remind_payment_request"
SEND_TOOLS = {"phone.send_text_message", "venmo.create_transaction", REMINDER_TOOL}


class AppWorldAPIError(ValueError):
    def __init__(self, tool, status):
        self.status = status
        super().__init__(f"{tool} returned HTTP {status}")


class AppWorldTools:
    """Register API schemas and verified patch contracts for one local AppWorld account."""

    supports_batch_operations = True

    def __init__(self, world, api_docs: Path):
        self.connection = AppWorldBatchEnvironment(world, "")
        self.specs = {}
        self.authenticated = set()
        self.updates = {}
        self.deletions = set()
        self.reads = {}
        self.apps = {}
        apps = set(world.task.allowed_apps) - {"admin", "supervisor", "api_docs"}
        for app in sorted(apps):
            openapi = api_docs / "openapi" / f"{app}.json"
            functions = api_docs / "function_calling" / f"{app}.json"
            if openapi.is_file() and functions.is_file():
                self.load_app(app, openapi, functions)

    def load_app(self, app, openapi, functions):
        functions = {
            item["function"]["name"]: item["function"] for item in json.loads(functions.read_text())
        }
        paths = json.loads(openapi.read_text())["paths"]
        self.apps[app] = self.connection.world.task.app_descriptions.get(app, app)

        def register(operation, fields=None, identity=(), *, irreversible=False):
            entry = functions[operation["operationId"]]
            name = entry["name"].replace("__", ".", 1)
            schema = deepcopy(entry["parameters"])
            properties = schema["properties"]
            if properties.pop("access_token", None) is not None:
                self.authenticated.add(name)
            if fields is not None:
                schema["properties"] = {
                    key: properties[key] for key in (*identity, *sorted(fields))
                }
                for key in identity:
                    if schema["properties"][key].get("type") == "integer":
                        schema["properties"][key]["minimum"] = 1
                if name == "phone.update_alarm":
                    schema["properties"]["time"]["pattern"] = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
                schema["minProperties"] = len(identity) + bool(fields)
            schema["required"] = [
                key for key in schema.get("required", []) if key in schema["properties"]
            ]
            if name == "venmo.create_transaction":
                schema["required"] = sorted(schema["properties"])
            if name == REMINDER_TOOL:
                schema["properties"]["payment_request_id"]["minimum"] = 1
            schema["additionalProperties"] = False
            spec = ToolSpec(
                name=name,
                description=entry["description"],
                parameters=schema,
                effect="write" if fields is not None else "read",
                risk=ToolRisk(
                    externally_visible=name != "simple_note.update_note",
                    reversible=not irreversible,
                    affected_objects=1,
                    required_permissions=[name],
                )
                if fields is not None
                else None,
            )
            Draft202012Validator.check_schema(schema)
            self.specs[name] = spec
            return name

        for path, operations in paths.items():
            post = operations.get("post")
            if post and post["operationId"] in functions:
                write = post["operationId"].replace("__", ".", 1)
                if write in SEND_TOOLS:
                    props = functions[post["operationId"]]["parameters"]["properties"]
                    register(post, props.keys() - {"access_token"}, irreversible=True)
            get = operations.get("get")
            if not get or get["operationId"] not in functions:
                continue
            name = get["operationId"].replace("__", ".", 1)
            api = name.split(".", 1)[1]
            restricted = api in {
                "show_account",
                "show_profile",
                "show_payment_card",
                "show_payment_cards",
            }
            if api.startswith("download_") or (restricted and name != "venmo.show_payment_cards"):
                continue
            register(get)
            if name == "venmo.show_payment_cards":
                self.specs[name].description = (
                    "List your payment card IDs, last four digits, expiration, and expired status. "
                    "Use the card ID for a payment. Reject expired cards."
                )
            self.reads[name] = path
            identity = tuple(re.findall(r"{(\w+)}", path))
            delete = operations.get("delete")
            if delete and identity and delete["operationId"] in functions:
                write = delete["operationId"].replace("__", ".", 1)
                if write in DELETE_TOOLS:
                    register(delete, set(), identity, irreversible=True)
                    self.updates[write] = (name, identity)
                    self.deletions.add(write)
            patch = operations.get("patch")
            if not patch or not identity or patch["operationId"] not in functions:
                continue
            write = patch["operationId"].replace("__", ".", 1)
            properties = functions[patch["operationId"]]["parameters"]["properties"]
            fields = UPDATE_FIELDS.get(write, set()) & properties.keys()
            if fields:
                register(patch, fields, identity)
                self.updates[write] = (name, identity)

    def get_tool_schemas(self):
        return list(self.specs.values())

    def policy(self):
        writes = sorted(name for name, spec in self.specs.items() if spec.effect == "write")
        return PolicyEngine(
            PolicyConfig(
                allowed_write_tools=writes,
                granted_permissions=writes,
            )
        )

    def describe(self, names):
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in self.specs for name in names)
        ):
            raise ValueError("Choose documented tool names from the catalog")
        return [self.specs[name].model_dump() for name in names]

    def catalog(self, apps):
        if not apps or any(app not in self.apps for app in apps):
            raise ValueError("Choose application names from the catalog")
        return [
            {**spec.model_dump(include={"name", "description", "effect"})}
            for name, spec in self.specs.items()
            if name.split(".", 1)[0] in apps
        ]

    def request(self, action):
        app, api = action.tool.split(".", 1)
        arguments = dict(action.arguments)
        if action.tool in self.authenticated:
            arguments["access_token"] = self.connection.authenticate(app)
        requester = self.connection.world.requester
        requester.reset_request_count()
        # AppWorld's public request() flattens some HTTP errors to successful-looking
        # message dictionaries. Inspect status before decoding the local response.
        try:
            response = requester._request(app, api, raise_on_failure=False, **arguments)
        except Exception as error:
            raise ValueError(f"{action.tool} failed ({type(error).__name__})") from None
        if response.status_code != 200:
            raise AppWorldAPIError(action.tool, response.status_code)
        return requester.response_to_json(response)

    def validate(self, action, effect):
        spec = self.specs.get(action.tool)
        if spec is None or spec.effect != effect:
            raise ValueError(f"Unavailable {effect} tool: {action.tool}")
        try:
            Draft202012Validator(spec.parameters, format_checker=FormatChecker()).validate(
                action.arguments
            )
        except ValidationError as error:
            path = ".".join(str(part) for part in error.path) or "arguments"
            raise ValueError(f"Invalid {path} for {action.tool}") from None

    def call_read_tool(self, action):
        self.validate(action, "read")
        result = self.request(action)
        if action.tool == "venmo.show_payment_cards":
            today = self.connection.now()
            return [
                {
                    "payment_card_id": card["payment_card_id"],
                    "last_four_digits": str(card["card_number"]).split(".", 1)[0][-4:],
                    "expiry_year": card["expiry_year"],
                    "expiry_month": card["expiry_month"],
                    "expired": (card["expiry_year"], card["expiry_month"])
                    < (today.year, today.month),
                }
                for card in result
            ]
        return result

    def grounding_keys(self, action):
        self.validate(action, "write")
        if action.tool == "phone.send_text_message":
            return [("phone", "phone_number", action.arguments["phone_number"])]
        if action.tool == "venmo.create_transaction":
            return [
                ("venmo", "payment_card_id", action.arguments["payment_card_id"]),
                ("", "email", action.arguments["receiver_email"]),
            ]
        if action.tool == REMINDER_TOOL:
            return [("venmo", "payment_request_id", action.arguments["payment_request_id"])]
        app = action.tool.split(".", 1)[0]
        return [(app, key, action.arguments[key]) for key in self.updates[action.tool][1]]

    def target(self, action):
        self.validate(action, "write")
        if action.tool in SEND_TOOLS:
            return action
        reader, identity = self.updates[action.tool]
        return Action(tool=reader, arguments={key: action.arguments[key] for key in identity})

    def outbound_snapshot(self, action):
        if action.tool == REMINDER_TOOL:
            request_id = action.arguments["payment_request_id"]
            for page in range(100):
                rows = self.call_read_tool(
                    Action(
                        tool="venmo.show_sent_payment_requests",
                        arguments={"status": "pending", "page_limit": 20, "page_index": page},
                    )
                )
                for row in rows:
                    if row["payment_request_id"] == request_id:
                        return row
                if len(rows) < 20:
                    break
            raise ValueError("The payment request is not in your pending sent requests")
        if action.tool == "phone.send_text_message":
            number = action.arguments["phone_number"]
            contacts = self.call_read_tool(
                Action(tool="phone.search_contacts", arguments={"query": number})
            )
            matches = [row for row in contacts if row["phone_number"] == number]
            return {"phone_number": number, "contacts": matches}
        cards = self.call_read_tool(Action(tool="venmo.show_payment_cards", arguments={}))
        card_id = action.arguments["payment_card_id"]
        card = next(
            (candidate for candidate in cards if candidate["payment_card_id"] == card_id), None
        )
        if card is None or card["expired"]:
            raise ValueError("The selected payment card is missing or expired")
        return card

    def observe_plan(self, actions):
        snapshots = []
        for action in actions:
            reader = self.target(action)
            target = reader.model_dump()
            if any(row["target"] == target for row in snapshots):
                raise ValueError("Combine changes to the same record into one operation")
            if action.tool in SEND_TOOLS:
                snapshots.append({"target": target, "resource": self.outbound_snapshot(action)})
                continue
            try:
                resource = self.call_read_tool(reader)
            except AppWorldAPIError as error:
                if action.tool not in self.deletions or error.status != 409:
                    raise
                resource = None
            fields = action.arguments.keys() - reader.arguments.keys()
            if resource is not None and (
                not isinstance(resource, dict) or not fields <= resource.keys()
            ):
                raise ValueError(f"Cannot independently verify all fields of {action.tool}")
            snapshots.append(
                {
                    "target": target,
                    "resource": {
                        key: value for key, value in resource.items() if key != "updated_at"
                    }
                    if resource is not None
                    else None,
                }
            )
        return snapshots

    def operation_snapshot(self, action, rows):
        target = self.target(action).model_dump()
        return [row for row in rows if row["target"] == target]

    def observe_operation(self, action):
        return self.observe_plan([action])

    def advance_plan(self, rows, action):
        if action.tool in SEND_TOOLS:
            return rows
        expected = deepcopy(rows)
        (record,) = self.operation_snapshot(action, expected)
        if action.tool in self.deletions:
            record["resource"] = None
            return expected
        resource = record["resource"]
        for key in action.arguments.keys() - record["target"]["arguments"].keys():
            resource[key] = action.arguments[key]
        if action.tool == "spotify.update_playlist" and "is_public" in action.arguments:
            resource["shareable_link"] = (
                f"https://spotify.com/playlists/{action.arguments['playlist_id']}"
                if action.arguments["is_public"]
                else None
            )
        return expected

    def call_write_tool(self, action, operation_id):
        self.validate(action, "write")
        result = self.request(action)
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected acknowledgement from {action.tool}")
        return result

    def verify_plan_state(self, actions, expected):
        return self.observe_plan(actions) == expected

    def verify_operation(self, state, result):
        action = state.action
        args = action.arguments
        if action.tool == REMINDER_TOOL:
            # Sender-side API does not expose the recipient's notification.
            passed = result.get("message") == "Payment request reminder sent."
            return VerificationResult(passed=passed, checks={"receipt_only": passed})
        if action.tool == "phone.send_text_message":
            message_id = result.get("text_message_id")
            if type(message_id) is not int:
                return VerificationResult(passed=False, checks={"outbound_readback_matches": False})
            row = self.call_read_tool(
                Action(
                    tool="phone.show_text_message",
                    arguments={"text_message_id": message_id},
                )
            )
            passed = (
                row["sender"]["phone_number"] == self.connection.phone_number
                and row["receiver"]["phone_number"] == args["phone_number"]
                and row["message"] == args["message"]
            )
            return VerificationResult(passed=passed, checks={"outbound_readback_matches": passed})
        if action.tool == "venmo.create_transaction":
            tx_id = result.get("transaction_id")
            if type(tx_id) is not int:
                return VerificationResult(
                    passed=False, checks={"transaction_readback_matches": False}
                )
            row = self.call_read_tool(
                Action(
                    tool="venmo.show_transaction",
                    arguments={"transaction_id": tx_id},
                )
            )
            passed = (
                row["receiver"]["email"] == args["receiver_email"]
                and row["amount"] == args["amount"]
                and row["private"] is args["private"]
                and row["description"] == args["description"]
                and row["payment_card_digits"] == state.before[0]["resource"]["last_four_digits"]
            )
            return VerificationResult(
                passed=passed, checks={"transaction_readback_matches": passed}
            )
        expected = self.advance_plan(state.before, state.action)
        passed = self.verify_plan_state([state.action], expected)
        return VerificationResult(passed=passed, checks={"scoped_readback_matches": passed})
