"""HTTP client used by the CLI and the MCP server. It holds no rules of its own."""

from __future__ import annotations

import uuid
from typing import Any

import httpx


class ClientError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class Client:
    def __init__(self, base_url: str, token: str, *, http: httpx.Client | None = None):
        """``http`` lets callers supply a preconfigured client, such as a test client."""
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(30.0, read=330.0)
        )
        self._http.headers["Authorization"] = f"Bearer {token}"

    def _call(self, method: str, path: str, *, key: str | None = None, **kwargs: Any) -> Any:
        headers = {"Idempotency-Key": key} if key else None
        response = self._http.request(method, f"/v1{path}", headers=headers, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise ClientError(response.status_code, detail)
        return response.json()

    @staticmethod
    def _key(key: str | None) -> str:
        return key or uuid.uuid4().hex

    # agent operations
    def brief(self, campaign_id: str) -> dict:
        return self._call("GET", f"/campaigns/{campaign_id}/brief")

    def wait(self, campaign_id: str, after: int = 0, timeout: float = 30.0) -> dict:
        return self._call(
            "GET", f"/campaigns/{campaign_id}/wait", params={"after": after, "timeout": timeout}
        )

    def propose(self, campaign_id: str, body: dict, key: str | None = None) -> dict:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/experiments", json=body, key=self._key(key)
        )

    def results(self, campaign_id: str) -> list[dict]:
        return self._call("GET", f"/campaigns/{campaign_id}/results")

    def failures(self, campaign_id: str) -> dict:
        return self._call("GET", f"/campaigns/{campaign_id}/failures")

    def pause(self, campaign_id: str) -> dict:
        return self._call("POST", f"/campaigns/{campaign_id}/pause")

    def resume(self, campaign_id: str) -> dict:
        return self._call("POST", f"/campaigns/{campaign_id}/resume")

    def cancel(self, campaign_id: str) -> dict:
        return self._call("POST", f"/campaigns/{campaign_id}/cancel")

    def request_approval(
        self, campaign_id: str, kind: str, payload: dict | None = None, key: str | None = None
    ) -> dict:
        return self._call(
            "POST",
            f"/campaigns/{campaign_id}/approvals",
            json={"kind": kind, "payload": payload or {}},
            key=self._key(key),
        )

    def report(self, campaign_id: str) -> dict:
        return self._call("GET", f"/campaigns/{campaign_id}/report")

    # human operations
    def register_profile(self, profile: dict) -> dict:
        return self._call("POST", "/profiles", json=profile)

    def create_campaign(self, spec: dict, key: str | None = None) -> dict:
        return self._call("POST", "/campaigns", json=spec, key=self._key(key))

    def list_campaigns(self) -> list[dict]:
        return self._call("GET", "/campaigns")

    def decide_approval(self, approval_id: str, grant: bool) -> dict:
        return self._call("POST", f"/approvals/{approval_id}/decision", json={"grant": grant})

    def set_guidance(self, campaign_id: str, text: str) -> dict:
        return self._call("PUT", f"/campaigns/{campaign_id}/guidance", json={"text": text})
