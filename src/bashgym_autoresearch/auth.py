"""Role-scoped bearer tokens. Agents can request gated actions; humans grant them."""

from __future__ import annotations

import hashlib
import secrets
from typing import Literal

from bashgym_autoresearch.contracts import FrozenModel
from bashgym_autoresearch.store import Store, utc_now

Role = Literal["agent", "human"]
TOKEN_PREFIX = "bgar_"


class AuthError(PermissionError):
    """The request carries no valid token."""


class Forbidden(PermissionError):
    """The authenticated principal may not perform this action."""


class Principal(FrozenModel):
    role: Role
    label: str


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_token(store: Store, role: Role, label: str) -> str:
    if role not in ("agent", "human"):
        raise ValueError("role must be 'agent' or 'human'")
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with store.transaction() as db:
        db.execute(
            "INSERT INTO tokens(hash, role, label, created_at) VALUES (?, ?, ?, ?)",
            (_hash(token), role, label, utc_now()),
        )
    return token


def authenticate(store: Store, token: str | None) -> Principal:
    if not token or not token.startswith(TOKEN_PREFIX):
        raise AuthError("a valid bearer token is required")
    row = store.read_one("SELECT role, label FROM tokens WHERE hash = ?", (_hash(token),))
    if row is None:
        raise AuthError("a valid bearer token is required")
    return Principal(role=row["role"], label=row["label"])


def require_human(principal: Principal) -> None:
    if principal.role != "human":
        raise Forbidden("this action requires a human token; agents may request approval")
