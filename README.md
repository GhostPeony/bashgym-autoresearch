# bashgym-autoresearch

**Status: early, built in public.** Milestone 1, the GPU-free core loop, works
end to end; see [What works today](#what-works-today). Training, the real
evaluation suites, trace capture, and the dashboards are still being built.

bashgym-autoresearch lets an AI agent improve a model through repeated,
controlled experiments while a human checks in and approves the results:

1. evaluate the starting model on a fixed, sealed evaluation suite;
2. inspect task-level failures;
3. propose one controlled change to data, training, or reward;
4. train a candidate;
5. evaluate it on the identical suite;
6. keep it only if the improvement is statistically real, otherwise discard it;
7. repeat until a stop rule is met, then produce a report for human approval.

The agent supplies the scientific judgment. The platform supplies execution,
the frozen evaluation, the experiment record, and the keep/discard decision,
and it enforces those rules itself rather than relying on the agent's
instructions. It is the streamlined successor to the experiment loop in
[BashGym](https://github.com/GhostPeony/bashgym).

## Design

- **Any agent harness.** The same small set of operations is available through
  an MCP server, a CLI, and an HTTP API, so Claude Code, Codex, OpenCode, Hermes
  Agent, or another compatible agent drive the loop the same way.
- **Server-side enforcement.** Evaluation assets are pinned by SHA-256 and kept
  outside the agent's control. Agent credentials can request a campaign start,
  budget increase, promotion, or publication; only human credentials can grant
  them.
- **Sealed evaluation.** Gold answers never enter the sandbox. An infrastructure
  failure makes an evaluation incomplete instead of shrinking its denominator.
- **Statistically gated decisions.** A candidate is kept only when the lower
  bound of a paired bootstrap confidence interval clears the declared minimum
  improvement and protected metrics hold. Otherwise the result is inconclusive.
- **Maintained training tools.** Training runs through existing open-source
  tooling such as Hugging Face TRL and NVIDIA NeMo AutoModel rather than a
  platform-specific trainer.
- **Trace capture.** Coding-agent sessions can be imported as training data,
  with secrets and personal paths redacted at import time.
- **Web dashboards.** Campaign status, live training curves, the experiment
  timeline, agent activity, and pending approvals are viewable in a browser.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned components and
milestones.

## What works today

Milestone 1 provides the loop itself, proven with smoke stages that need no GPU
([examples/smoke](examples/smoke)):

- `bashgym-ar serve` runs the HTTP API and a worker that launches registered
  stage programs as local processes, survives restarts by adopting runs that
  are still going, enforces per-stage time limits, and seals each stage's
  outputs with an HMAC.
- The ten agent operations are available through MCP (`bashgym-ar mcp`), the
  CLI, and HTTP. [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) describes the loop
  for any harness.
- Agent tokens can request but not grant start, budget, promotion, and
  publication approvals; they cannot edit guidance or register stage programs.
- Evaluation stages receive a platform-issued context and must echo its digest;
  a registered script whose SHA-256 changes is refused at launch.
- Candidates are kept only when the cluster-bootstrap interval's lower bound
  clears the minimum improvement and protected metrics hold; otherwise the
  result is discarded or inconclusive.
- Smoke-scope results are labelled throughout and cannot be promoted.

Not yet available: real training and evaluation runners, remote (SSH)
execution, trace capture, notifications, and the web dashboards.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run black --check .
```

## License

[MIT](LICENSE)
