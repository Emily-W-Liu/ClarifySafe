"""Agent-visible AppWorld tools, scoped observations and verification."""

from copy import deepcopy
from datetime import datetime

from clarifysafe.agent.phone_contract import verify_phone
from clarifysafe.agent.policy import PolicyConfig, PolicyEngine
from clarifysafe.shared.schemas import ToolRisk, ToolSpec, VerificationResult

READ_TOOLS = {
    "phone": {
        "show_alarm",
        "show_alarms",
        "search_contacts",
        "search_text_messages",
        "search_voice_messages",
        "show_text_message",
    },
    "simple_note": {"search_notes", "show_note"},
    "venmo": {
        "show_payment_cards",
        "show_sent_payment_requests",
        "show_transaction",
        "show_transactions",
    },
}


class AppWorldBatchEnvironment:
    supports_batch_operations = True

    def __init__(self, world, family):
        self.world, self.family = world, family
        self.tokens = {}
        self.calls = []
        self.phone_number = None
        self.refresher = None

    def request(self, app, api, **arguments):
        self.world.requester.reset_request_count()
        result = self.world.requester.request(app, api, raise_on_failure=False, **arguments)
        if isinstance(result, dict) and ("detail" in result or "error" in result):
            raise ValueError(f"{app}.{api} failed")
        return result

    def authenticate(self, app):
        if app not in self.tokens:
            profile = self.request("supervisor", "show_profile")
            passwords = self.request("supervisor", "show_account_passwords")
            (password,) = [entry["password"] for entry in passwords if entry["account_name"] == app]
            username = profile["phone_number" if app == "phone" else "email"]
            if app == "phone":
                self.phone_number = username
            login = self.request(app, "login", username=username, password=password)
            token = login.get("access_token") if isinstance(login, dict) else None
            if not isinstance(token, str) or not token:
                raise ValueError(f"Could not authenticate to {app}")
            self.tokens[app] = token
        return self.tokens[app]

    def read(self, app, api, **arguments):
        if api not in READ_TOOLS.get(app, ()):
            raise ValueError(f"{app}.{api} is outside the read-only allowlist")
        token = self.authenticate(app)
        self.calls.append(f"{app}.{api}")
        return self.request(app, api, access_token=token, **arguments)

    def pages(self, app, api, **arguments):
        rows = []
        for index in range(100):
            batch = self.read(app, api, page_index=index, page_limit=20, **arguments)
            if not isinstance(batch, list) or not all(isinstance(row, dict) for row in batch):
                raise ValueError(f"Unexpected response from {app}.{api}")
            rows.extend(batch)
            if len(batch) < 20:
                return rows
        raise ValueError(f"Could not read every page of {app}.{api}")

    def now(self):
        shown = self.request("phone", "get_current_date_and_time")
        return datetime.strptime(shown["date"] + " " + shown["time"], "%A, %B %d, %Y %I:%M %p")

    def alarms(self):
        return sorted(self.pages("phone", "show_alarms"), key=lambda row: row["alarm_id"])

    def requests(self):
        return sorted(
            self.pages("venmo", "show_sent_payment_requests", status="pending"),
            key=lambda row: row["payment_request_id"],
        )

    def policy(self):
        tools = self.get_tool_schemas()
        return PolicyEngine(
            PolicyConfig(
                allowed_write_tools=[tool.name for tool in tools if tool.effect == "write"],
                granted_permissions=sorted(
                    {
                        permission
                        for tool in tools
                        if tool.risk
                        for permission in tool.risk.required_permissions
                    }
                ),
            )
        )

    def get_tool_schemas(self):
        if self.family == "message_deletion":
            return [
                ToolSpec(
                    name=f"phone.delete_{kind}_message",
                    description=f"Delete one {kind} message",
                    effect="write",
                    parameters={
                        "type": "object",
                        "properties": {f"{kind}_message_id": {"type": "integer"}},
                        "required": [f"{kind}_message_id"],
                        "additionalProperties": False,
                    },
                    risk=ToolRisk(
                        externally_visible=False,
                        reversible=False,
                        affected_objects=1,
                        required_permissions=["phone.messages.delete"],
                    ),
                )
                for kind in ("text", "voice")
            ]
        if self.family in {"checklist", "movie_reply", "payment"}:
            from clarifysafe.experiments.official_interactive import NoteEnvironment
            from clarifysafe.experiments.official_phone import PhoneEnvironment
            from clarifysafe.experiments.official_venmo import PaymentEnvironment

            template = {
                "checklist": NoteEnvironment,
                "movie_reply": PhoneEnvironment,
                "payment": PaymentEnvironment,
            }[self.family]
            tools = template(None, None).get_tool_schemas()
            return tools
        if self.family == "alarms":
            name = "phone.update_alarm"
            parameters = {
                "type": "object",
                "properties": {
                    "alarm_id": {"type": "integer"},
                    "time": {"type": "string", "pattern": r"^(?:[01]\d|2[0-3]):[0-5]\d$"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["alarm_id"],
                "additionalProperties": False,
                "minProperties": 2,
            }
            risk = ToolRisk(
                externally_visible=False,
                reversible=True,
                affected_objects=1,
                required_permissions=["alarms.update"],
            )
        elif self.family == "payment_reminders":
            name = "venmo.remind_payment_request"
            parameters = {
                "type": "object",
                "properties": {"payment_request_id": {"type": "integer"}},
                "required": ["payment_request_id"],
                "additionalProperties": False,
            }
            risk = ToolRisk(
                externally_visible=True,
                reversible=False,
                affected_objects=1,
                required_permissions=["payments.remind"],
            )
        else:
            return []
        return [
            ToolSpec(name=name, description=name, parameters=parameters, effect="write", risk=risk)
        ]

    def observe_plan(self, actions):
        if self.refresher is not None:
            return self.refresher()
        if self.family == "alarms":
            return self.alarms()
        return self.requests()

    def operation_snapshot(self, action, rows):
        if self.family in {"checklist", "movie_reply", "payment"}:
            return rows
        if self.family == "message_deletion":
            (key,) = action.arguments
        else:
            key = "alarm_id" if self.family == "alarms" else "payment_request_id"
        if self.family == "payment_reminders":
            rows = rows[0]["requests"]
        selected = [row for row in rows if row.get(key) == action.arguments[key]]
        if len(selected) != 1:
            raise ValueError("Operation target is not uniquely present")
        return selected

    def observe_operation(self, action):
        if self.family == "alarms":
            row = self.read("phone", "show_alarm", alarm_id=action.arguments["alarm_id"])
            return self.operation_snapshot(action, [row])
        return self.operation_snapshot(action, self.observe_plan([action]))

    def advance_plan(self, rows, action):
        if self.family == "message_deletion":
            (victim,) = self.operation_snapshot(action, rows)
            return [row for row in rows if row != victim]
        if self.family != "alarms":
            return rows
        expected = deepcopy(rows)
        self.operation_snapshot(action, expected)[0].update(action.arguments)
        return expected

    def call_write_tool(self, action, operation_id):
        app, api = action.tool.split(".")
        self.calls.append(action.tool)
        result = self.request(app, api, access_token=self.tokens[app], **action.arguments)
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected response from {action.tool}")
        return result

    def verify_plan_state(self, actions, expected):
        if self.family in {"alarms", "message_deletion", "payment_reminders"}:
            return self.observe_plan(actions) == expected
        return True

    def verify_operation(self, state, result):
        if self.family == "checklist":
            from clarifysafe.agent.note_contract import verify_note

            return verify_note(state, self.observe_plan([state.action]), target_only=True)
        if self.family == "movie_reply":
            return verify_phone(state, self.observe_plan([state.action]))
        if self.family == "payment":
            args = state.action.arguments
            transaction_id = result.get("transaction_id")
            row = (
                self.read("venmo", "show_transaction", transaction_id=transaction_id)
                if type(transaction_id) is int
                else {}
            )
            card = state.before[0]["card"]
            expected = (
                row.get("transaction_id") == transaction_id
                and type(transaction_id) is int
                and row.get("receiver", {}).get("email") == args["receiver_email"]
                and row.get("amount") == args["amount"]
                and row.get("private") is args["private"]
                and row.get("description") == args["description"]
                and row.get("payment_card_digits") == str(card["card_number"]).split(".", 1)[0][-4:]
            )
            return VerificationResult(
                passed=expected, checks={"payment_readback_matches": expected}
            )
        if self.family == "message_deletion":
            (key,) = state.action.arguments
            passed = not any(
                row.get(key) == state.action.arguments[key]
                for row in self.observe_plan([state.action])
            )
            return VerificationResult(passed=passed, checks={"deleted_message_absent": passed})
        if self.family == "alarms":
            expected = {**state.before[0], **state.action.arguments}
            passed = self.observe_operation(state.action) == [expected]
            return VerificationResult(passed=passed, checks={"alarm_readback_matches": passed})
        # Venmo exposes the notification only to the recipient. The sender can check the acknowledgement.
        passed = result.get("message") == "Payment request reminder sent."
        return VerificationResult(passed=passed, checks={"receipt_only": passed})
