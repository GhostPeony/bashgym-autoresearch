"""MCP server exposing the ten agent operations over the HTTP API.

The server authenticates with the token in ``BGAR_TOKEN`` (normally an agent
token) against ``BGAR_URL``. It cannot do anything the token cannot.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer

from bashgym_autoresearch.client import Client

DEFAULT_URL = "http://127.0.0.1:8765"
AGENT_TOOLS = (
    "brief",
    "wait",
    "propose",
    "results",
    "failures",
    "pause",
    "resume",
    "cancel",
    "request_approval",
    "report",
)

INSTRUCTIONS = """Run model-improvement experiments in a campaign.
Loop: call brief; if next_action is propose_baseline or propose_candidate, propose one
experiment (a candidate must change exactly one variable); call wait until the experiment
is decided; read results and failures; repeat. Stop when next_action is stop. Read the
human guidance in every brief. Use request_approval for start, budget, promote, publish.
Decisions are made by the platform: keep only when the improvement interval clears the
minimum; inconclusive means add tasks or repeats rather than claiming a gain."""


def build_server(client: Client) -> MCPServer:
    server = MCPServer(name="bashgym-autoresearch", instructions=INSTRUCTIONS)

    @server.tool()
    def brief(campaign_id: str) -> dict[str, Any]:
        """Current state: next action, incumbent, recent results, budget, guidance, approvals."""
        return client.brief(campaign_id)

    @server.tool()
    def wait(campaign_id: str, after_seq: int = 0, timeout_seconds: float = 60) -> dict[str, Any]:
        """Block until new campaign events arrive after ``after_seq`` or the timeout passes."""
        return client.wait(campaign_id, after_seq, timeout_seconds)

    @server.tool()
    def propose(
        campaign_id: str,
        role: str,
        hypothesis: str,
        estimated_cost: float,
        recipe: dict[str, Any] | None = None,
        change_variable: str | None = None,
        change_before: Any = None,
        change_after: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Propose a baseline (no change) or a candidate that changes exactly one variable."""
        change = None
        if change_variable is not None:
            change = {"variable": change_variable, "before": change_before, "after": change_after}
        body = {
            "role": role,
            "change": change,
            "recipe": recipe or {},
            "hypothesis": hypothesis,
            "estimated_cost": estimated_cost,
        }
        return client.propose(campaign_id, body, key=idempotency_key)

    @server.tool()
    def results(campaign_id: str) -> list[dict[str, Any]]:
        """All decided experiments with improvement intervals and reasons."""
        return client.results(campaign_id)

    @server.tool()
    def failures(campaign_id: str) -> dict[str, Any]:
        """The latest crashed or incomplete experiment with stage exit codes and stderr tails."""
        return client.failures(campaign_id)

    @server.tool()
    def pause(campaign_id: str) -> dict[str, Any]:
        """Pause the campaign; running stages finish but nothing new starts."""
        return client.pause(campaign_id)

    @server.tool()
    def resume(campaign_id: str) -> dict[str, Any]:
        """Resume a paused campaign (a human pause can only be resumed by a human)."""
        return client.resume(campaign_id)

    @server.tool()
    def cancel(campaign_id: str) -> dict[str, Any]:
        """Cancel the campaign permanently and stop any running stage."""
        return client.cancel(campaign_id)

    @server.tool()
    def request_approval(
        campaign_id: str, kind: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Ask a human to approve start, budget (payload.amount), promote or publish."""
        return client.request_approval(campaign_id, kind, payload)

    @server.tool()
    def report(campaign_id: str) -> dict[str, Any]:
        """The full experiment record for the human to review."""
        return client.report(campaign_id)

    return server


def main() -> None:
    token = os.environ.get("BGAR_TOKEN")
    if not token:
        raise SystemExit("set BGAR_TOKEN to an agent token")
    client = Client(os.environ.get("BGAR_URL", DEFAULT_URL), token)
    build_server(client).run("stdio")
