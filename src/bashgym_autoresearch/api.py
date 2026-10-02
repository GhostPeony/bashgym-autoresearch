"""HTTP API. Every route authenticates a bearer token and delegates to the service.

Annotations are evaluated eagerly (no ``from __future__ import annotations``) so
FastAPI can resolve the dependency aliases defined inside ``create_app``.
"""

import secrets
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from bashgym_autoresearch.auth import AuthError, Forbidden, Principal, authenticate
from bashgym_autoresearch.contracts import (
    ApprovalKind,
    CampaignSpec,
    Change,
    ExperimentRole,
    StageProfile,
)
from bashgym_autoresearch.decision import EvaluationMismatch, ProposalError
from bashgym_autoresearch.seal import Sealer
from bashgym_autoresearch.service import NotFound, RuleError, Service
from bashgym_autoresearch.store import ConflictError, Store

SEAL_KEY_FILENAME = "seal.key"
STATE_FILENAME = "state.db"


def open_home(home: Path) -> Service:
    """Open (or initialize) the state directory and return its service."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    key_path = home / SEAL_KEY_FILENAME
    if not key_path.exists():
        key_path.write_bytes(secrets.token_bytes(32))
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
    return Service(Store(home / STATE_FILENAME), Sealer(key_path.read_bytes()), home)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposeBody(_Body):
    role: ExperimentRole
    change: Change | None = None
    recipe: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str
    estimated_cost: float = Field(ge=0, allow_inf_nan=False)


class ApprovalBody(_Body):
    kind: ApprovalKind
    payload: dict[str, Any] = Field(default_factory=dict)


class DecisionBody(_Body):
    grant: bool


class GuidanceBody(_Body):
    text: str


class TokenBody(_Body):
    role: Literal["agent", "human"]
    label: str


_STATUS = [
    (AuthError, 401),
    (Forbidden, 403),
    (NotFound, 404),
    (RuleError, 409),
    (ConflictError, 409),
    (ProposalError, 422),
    (EvaluationMismatch, 422),
]


def create_app(service: Service) -> FastAPI:
    app = FastAPI(title="bashgym-autoresearch", version="0.1.0")

    for error, status in _STATUS:

        def handler(request: Request, exc: Exception, status: int = status) -> JSONResponse:
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)}, status_code=status
            )

        app.add_exception_handler(error, handler)

    def principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        return authenticate(service.store, token)

    def idempotency(idempotency_key: Annotated[str | None, Header()] = None) -> str:
        if not idempotency_key or len(idempotency_key) > 200:
            raise HTTPException(
                400, "an Idempotency-Key header of at most 200 characters is required"
            )
        return idempotency_key

    Who = Annotated[Principal, Depends(principal)]
    Key = Annotated[str, Depends(idempotency)]

    @app.get("/v1/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/v1/tokens")
    def create_token(body: TokenBody, who: Who) -> dict:
        return service.create_token(who, body.role, body.label)

    @app.post("/v1/profiles")
    def register_profile(profile: StageProfile, who: Who) -> dict:
        return service.register_profile(who, profile)

    @app.post("/v1/campaigns")
    def create_campaign(spec: CampaignSpec, who: Who, key: Key) -> dict:
        return service.create_campaign(who, spec, key)

    @app.get("/v1/campaigns")
    def list_campaigns(who: Who) -> list[dict]:
        return service.list_campaigns(who)

    @app.get("/v1/campaigns/{campaign_id}/brief")
    def brief(campaign_id: str, who: Who) -> dict:
        return service.brief(who, campaign_id)

    @app.get("/v1/campaigns/{campaign_id}/wait")
    def wait(campaign_id: str, who: Who, after: int = 0, timeout: float = 30.0) -> dict:
        return service.wait(who, campaign_id, after, timeout)

    @app.post("/v1/campaigns/{campaign_id}/experiments")
    def propose(campaign_id: str, body: ProposeBody, who: Who, key: Key) -> dict:
        return service.propose(
            who,
            campaign_id,
            role=body.role,
            change=body.change,
            recipe=body.recipe,
            hypothesis=body.hypothesis,
            estimated_cost=body.estimated_cost,
            idempotency_key=key,
        )

    @app.get("/v1/campaigns/{campaign_id}/results")
    def results(campaign_id: str, who: Who) -> list[dict]:
        return service.results(who, campaign_id)

    @app.get("/v1/campaigns/{campaign_id}/failures")
    def failures(campaign_id: str, who: Who) -> dict:
        return service.failures(who, campaign_id)

    @app.post("/v1/campaigns/{campaign_id}/pause")
    def pause(campaign_id: str, who: Who) -> dict:
        return service.pause(who, campaign_id)

    @app.post("/v1/campaigns/{campaign_id}/resume")
    def resume(campaign_id: str, who: Who) -> dict:
        return service.resume(who, campaign_id)

    @app.post("/v1/campaigns/{campaign_id}/cancel")
    def cancel(campaign_id: str, who: Who) -> dict:
        return service.cancel(who, campaign_id)

    @app.post("/v1/campaigns/{campaign_id}/approvals")
    def request_approval(campaign_id: str, body: ApprovalBody, who: Who, key: Key) -> dict:
        return service.request_approval(who, campaign_id, body.kind, body.payload, key)

    @app.post("/v1/approvals/{approval_id}/decision")
    def decide_approval(approval_id: str, body: DecisionBody, who: Who) -> dict:
        return service.decide_approval(who, approval_id, body.grant)

    @app.put("/v1/campaigns/{campaign_id}/guidance")
    def set_guidance(campaign_id: str, body: GuidanceBody, who: Who) -> dict:
        return service.set_guidance(who, campaign_id, body.text)

    @app.get("/v1/campaigns/{campaign_id}/report")
    def report(campaign_id: str, who: Who) -> dict:
        return service.report(who, campaign_id)

    return app
