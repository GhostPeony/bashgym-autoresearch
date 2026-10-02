# bashgym-autoresearch

**Status: early, built in public.** This repository currently contains the
project skeleton and the design. Nothing below the "Design" heading is
implemented yet; each milestone lands with tests and is described here only
once it works.

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

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run black --check .
```

## License

[MIT](LICENSE)
