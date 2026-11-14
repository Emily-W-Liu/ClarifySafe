import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from filelock import FileLock, Timeout
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from clarifysafe.agent.executor import GuardedExecutor, issue_approval
from clarifysafe.agent.grounder import Grounder
from clarifysafe.agent.models.client import ModelConfig, ModelError, OpenAICompatibleClient
from clarifysafe.agent.models.structured import StructuredModel
from clarifysafe.agent.policy import PolicyConfig, PolicyEngine
from clarifysafe.agent.safety import verify_message
from clarifysafe.environment.mock import EventStore, MockEnvironment
from clarifysafe.shared.schemas import (
    Action,
    AgentInput,
    AgentResult,
    Candidate,
    TaskState,
    UserResponse,
)


class GraphState(TypedDict):
    task: dict


def build_graph(saver, environment_factory, events, get_grounder, policy, executor):
    def emit(state, key, data):
        events.emit(state.run_id, key, key, data)

    def ground(raw):
        state = TaskState.model_validate(raw["task"])
        if state.mode == "llm_mock":
            try:
                state = get_grounder().run(state, environment_factory(state.run_id))
            except (ModelError, ValueError, TimeoutError, OSError) as error:
                state.status, state.reason = "stopped", str(error)
            if state.status in {"stopped", "declined", "completed"}:
                events.emit(
                    state.run_id,
                    f"result:{state.clarification_round}",
                    "result",
                    {
                        "status": state.status,
                        "reason": state.reason,
                        "answer": state.answer,
                        "verification": state.verification.model_dump()
                        if state.verification
                        else None,
                    },
                )
            return {"task": state.model_dump()}
        # Keep the P1 scripted fixture for deterministic regression and offline demos.
        supported = state.instruction in {"把会议纪要发给 Alex", "Send the meeting minutes to Alex"}
        if not supported:
            state.status, state.decision = "declined", "decline"
            state.reason = "P1 scripted demo supports only: 把会议纪要发给 Alex"
            emit(state, "declined", {"reason": state.reason})
        else:
            env = environment_factory(state.run_id)
            contacts = env.call_read_tool("search_contacts", {"query": "Alex"})
            state.candidates = [
                Candidate(object_id=c["id"], label=c["name"], source="application")
                for c in contacts
            ]
            state.before = env.observe_state()
            state.status, state.decision = "waiting_for_user", "ask"
            emit(state, "evidence", {"candidates": [c.model_dump() for c in state.candidates]})
        return {"task": state.model_dump()}

    def ask(raw):
        state = TaskState.model_validate(raw["task"])
        suffix = f":{state.clarification_round}" if state.mode == "llm_mock" else ""
        payload = {
            "intervention_id": f"{state.run_id}:ask{suffix}",
            "type": "ask",
            "question": state.question or "你指的是哪位 Alex？",
            "choices": [c.model_dump() for c in state.candidates],
            "accepts_text": state.mode == "llm_mock",
        }
        events.emit(state.run_id, "ask" + suffix, "ask", payload)
        response = UserResponse.model_validate(interrupt(payload))
        if state.mode == "llm_mock" and response.text is not None:
            state.user_answers.append(response.text)
            state.clarification_round += 1
            state.status, state.decision = "running", None
            events.emit(
                state.run_id, "user_text" + suffix, "user_response", {"text": response.text}
            )
            return {"task": state.model_dump()}
        env = environment_factory(state.run_id)
        state.action = Action(
            tool="send_message",
            arguments={
                "recipient_id": response.choice,
                "content": state.minutes_content
                if state.mode == "llm_mock"
                else env.call_read_tool("read_minutes", {}),
            },
        )
        state.decision = "confirm"
        emit(state, "user_choice", {"choice": response.choice})
        return {"task": state.model_dump()}

    def assess(raw):
        state = TaskState.model_validate(raw["task"])
        tools = environment_factory(state.run_id).get_tool_schemas()
        if state.policy_digest != policy.digest(tools):
            state.status, state.reason = "stopped", "Policy or tool registration changed"
        else:
            state.risk_assessment = policy.assess(state.action, tools)
            state.decision = state.risk_assessment.decision
            state.status = {
                "act": "running",
                "ask": "waiting_for_user",
                "confirm": "waiting_for_user",
                "decline": "declined",
            }[state.decision]
            state.reason = "; ".join(state.risk_assessment.reasons)
            emit(state, "risk_assessed", state.risk_assessment.model_dump())
        return {"task": state.model_dump()}

    def confirm(raw):
        state = TaskState.model_validate(raw["task"])
        if state.risk_assessment is None or state.policy_digest is None:
            state.status, state.reason = (
                "stopped",
                "Legacy task requires a new run for scoped approval",
            )
            emit(state, "stopped", {"reason": state.reason})
            return {"task": state.model_dump()}
        payload = {
            "intervention_id": f"{state.run_id}:confirm",
            "type": "confirm",
            "question": "是否授权发送以下消息？",
            "action": state.action.model_dump(),
            "risk": state.risk_assessment.model_dump(),
            "approval_ttl_seconds": policy.config.approval_ttl_seconds,
        }
        emit(state, "confirm", payload)
        response = UserResponse.model_validate(interrupt(payload))
        state.approval = issue_approval(state, policy, response.approved)
        if not response.approved:
            state.status, state.decision = "declined", "decline"
            state.reason = "User denied this action"
        else:
            state.status, state.decision = "running", "act"
        emit(state, "approval", state.approval.model_dump())
        return {"task": state.model_dump()}

    def execute(raw):
        state = TaskState.model_validate(raw["task"])
        try:
            result = executor.execute(state, environment_factory(state.run_id))
            emit(state, "tool_result", result)
        except (ValueError, TimeoutError, OSError, sqlite3.Error) as error:
            state.status = "stopped"
            state.reason = f"Execution not verified; no automatic write retry: {error}"
            emit(state, "stopped", {"reason": state.reason})
        return {"task": state.model_dump()}

    def verify(raw):
        state = TaskState.model_validate(raw["task"])
        try:
            state.verification = verify_message(
                state, environment_factory(state.run_id).observe_state()
            )
        except (ValueError, TimeoutError, OSError) as error:
            state.status, state.reason = "stopped", f"Post-state observation failed: {error}"
            emit(state, "result", {"status": state.status, "reason": state.reason})
            return {"task": state.model_dump()}
        state.status = "completed" if state.verification.passed else "stopped"
        state.reason = (
            None if state.verification.passed else "Observed state failed the execution contract"
        )
        emit(state, "verification", state.verification.model_dump())
        emit(state, "result", {"status": state.status, "reason": state.reason})
        return {"task": state.model_dump()}

    graph = StateGraph(GraphState)
    for name, node in [
        ("ground", ground),
        ("ask", ask),
        ("assess", assess),
        ("confirm", confirm),
        ("execute", execute),
        ("verify", verify),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "ground")
    graph.add_conditional_edges(
        "ground",
        lambda s: (
            END
            if s["task"]["status"] in {"declined", "stopped", "completed"}
            else "assess"
            if s["task"]["decision"] == "confirm"
            else "ask"
        ),
    )
    graph.add_conditional_edges(
        "ask", lambda s: "ground" if s["task"]["status"] == "running" else "assess"
    )
    graph.add_conditional_edges(
        "assess",
        lambda s: (
            END
            if s["task"]["status"] in {"stopped", "declined"}
            else {"act": "execute", "confirm": "confirm", "ask": "ask"}[s["task"]["decision"]]
        ),
    )
    graph.add_conditional_edges(
        "confirm", lambda s: END if s["task"]["status"] in {"declined", "stopped"} else "execute"
    )
    graph.add_conditional_edges(
        "execute", lambda s: END if s["task"]["status"] == "stopped" else "verify"
    )
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=saver)


class Runtime:
    """Single-writer local runtime. Do not serve concurrent resumes until P5 locking exists."""

    def __init__(self, saver, db, model_client=None, max_model_calls=8, policy_config=None):
        self.events = EventStore(db)
        self.environment = lambda run_id: MockEnvironment(db, run_id)
        self.model_client = model_client
        self.max_model_calls = max_model_calls
        self.owned_client = None
        self.policy = PolicyEngine(policy_config)
        self.executor = GuardedExecutor(db, self.policy, self.events)
        self.graph = build_graph(
            saver, self.environment, self.events, self.get_grounder, self.policy, self.executor
        )

    def get_grounder(self):
        if self.model_client is None:
            config = ModelConfig.from_env()
            self.owned_client = OpenAICompatibleClient(config)
            self.model_client = self.owned_client
            self.max_model_calls = config.max_calls_per_run
        return Grounder(
            StructuredModel(self.model_client, self.events, self.max_model_calls), self.events
        )

    def config(self, run_id):
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 20}

    def start(self, request: AgentInput, mode="scripted_mock") -> AgentResult:
        if mode not in {"scripted_mock", "llm_mock"}:
            raise ValueError("Use the dedicated AppWorld runner for AppWorld tasks")
        if mode == "llm_mock":
            self.get_grounder()  # Fail fast before creating a run if config is missing.
        run_id = uuid4().hex
        state = TaskState(
            run_id=run_id,
            instruction=request.instruction,
            mode=mode,
            policy_digest=self.policy.digest(self.environment(run_id).get_tool_schemas()),
        )
        self.events.emit(run_id, "started", "started", {"mode": mode, **request.model_dump()})
        self.graph.invoke({"task": state.model_dump()}, self.config(run_id))
        return self.result(run_id)

    def result(self, run_id: str) -> AgentResult:
        snapshot = self.graph.get_state(self.config(run_id))
        if not snapshot.values:
            raise ValueError("Unknown run ID")
        pending = next((item.value for task in snapshot.tasks for item in task.interrupts), None)
        state = TaskState.model_validate(snapshot.values["task"])
        return AgentResult(
            run_id=run_id, status=state.status, mode=state.mode, state=state, pending=pending
        )

    def resume(self, run_id: str, response: UserResponse) -> AgentResult:
        result = self.result(run_id)
        pending = result.pending
        if not pending or pending["intervention_id"] != response.intervention_id:
            raise ValueError("Stale response or no pending intervention")
        if pending["type"] == "ask":
            text_ok = (
                result.mode == "llm_mock"
                and response.text is not None
                and bool(response.text.strip())
                and response.choice is None
            )
            choice_ok = (
                response.choice in {c["object_id"] for c in pending["choices"]}
                and response.text is None
            )
            if not (text_ok or choice_ok) or response.approved is not None:
                raise ValueError("Ask requires one advertised choice or supported free-text answer")
        elif response.approved is None or response.choice is not None or response.text is not None:
            raise ValueError("Confirm requires an explicit boolean approval only")
        self.graph.invoke(Command(resume=response.model_dump()), self.config(run_id))
        return self.result(run_id)


class RuntimeBusy(ValueError):
    """Another process/thread owns this runtime directory."""


@contextmanager
def open_runtime(
    directory: str | Path,
    model_client=None,
    max_model_calls=8,
    policy_config: PolicyConfig | None = None,
):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root / "runtime.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout:
        raise RuntimeBusy("Runtime is busy; retry after the current request completes") from None
    try:
        with _open_locked_runtime(root, model_client, max_model_calls, policy_config) as runtime:
            yield runtime
    finally:
        lock.release()


@contextmanager
def _open_locked_runtime(root, model_client, max_model_calls, policy_config):
    with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as saver:
        db = sqlite3.connect(root / "application.sqlite")
        runtime = None
        try:
            runtime = Runtime(saver, db, model_client, max_model_calls, policy_config)
            yield runtime
        finally:
            if runtime is not None and runtime.owned_client is not None:
                runtime.owned_client.close()
            db.close()
