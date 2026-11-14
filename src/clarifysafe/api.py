"""Local demo HTTP service. Runtime and CLI share the same process-safe file lock."""

import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import Field

from clarifysafe.agent.models.client import ModelError
from clarifysafe.agent.runtime import RuntimeBusy, open_runtime
from clarifysafe.shared.schemas import AgentEvent, AgentInput, AgentResult, Contract, UserResponse


class EventPage(Contract):
    events: list[AgentEvent]
    next_after: int = Field(ge=0)


def create_app(
    directory=".clarifysafe",
    mode="scripted_mock",
    policy_config=None,
    allowed_origins=None,
    model_client=None,
):
    if mode not in {"scripted_mock", "llm_mock"}:
        raise ValueError("Unsupported server mode")
    root = Path(directory).resolve()
    app = FastAPI(title="ClarifySafe Agent API", version="0.4.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or [],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def runtime():
        return open_runtime(root, model_client=model_client, policy_config=policy_config)

    def require_run(agent, run_id):
        try:
            return agent.result(run_id)
        except ValueError as error:
            if str(error) == "Unknown run ID":
                raise HTTPException(
                    404, detail={"code": "run_not_found", "message": str(error)}
                ) from None
            raise

    @app.exception_handler(RuntimeBusy)
    async def busy_handler(request, error):
        return JSONResponse(
            status_code=409,
            headers={"Retry-After": "1"},
            content={"detail": {"code": "runtime_busy", "message": str(error)}},
        )

    @app.exception_handler(ModelError)
    async def model_handler(request, error):
        return JSONResponse(
            status_code=503,
            content={"detail": {"code": "model_unavailable", "message": str(error)}},
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_handler(request, error):
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "storage_unavailable",
                    "message": "Unable to access runtime storage",
                }
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": mode, "environment": "mock", "version": "0.4.0"}

    @app.post("/runs", response_model=AgentResult, status_code=201)
    def start(request: AgentInput):
        with runtime() as agent:
            return agent.start(request, mode=mode)

    @app.get("/runs/{run_id}", response_model=AgentResult)
    def status(run_id: str):
        with runtime() as agent:
            return require_run(agent, run_id)

    @app.post("/runs/{run_id}/responses", response_model=AgentResult)
    def respond(run_id: str, response: UserResponse):
        with runtime() as agent:
            current = require_run(agent, run_id)
            if (
                not current.pending
                or current.pending["intervention_id"] != response.intervention_id
            ):
                raise HTTPException(
                    409,
                    detail={
                        "code": "stale_intervention",
                        "message": "Refresh the task before submitting a response",
                    },
                )
            try:
                return agent.resume(run_id, response)
            except ValueError as error:
                raise HTTPException(
                    422, detail={"code": "invalid_response", "message": str(error)}
                ) from None

    @app.get("/runs/{run_id}/events", response_model=EventPage)
    def events(run_id: str, after: int = Query(default=0, ge=0)):
        with runtime() as agent:
            require_run(agent, run_id)
            rows = agent.events.read(run_id, after=after)
            return EventPage(
                events=[AgentEvent.model_validate(row) for row in rows],
                next_after=rows[-1]["sequence"] if rows else after,
            )

    return app
