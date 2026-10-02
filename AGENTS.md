# Agent instructions for developing bashgym-autoresearch

These instructions apply to any coding agent (Claude Code, Codex, OpenCode,
and others) working on this repository. `CLAUDE.md` points here.

## Purpose

Build a small, robust platform that lets an agent run model-improvement
experiments for hours with minimal human intervention: sealed evaluation, one
controlled change per experiment, statistically gated keep/discard, human
approval at irreversible points. Read `docs/ARCHITECTURE.md` before changing
behavior.

## Rules

- Enforce rules in the backend. Never rely on agent instructions, prompts, or
  harness-specific hooks to protect evaluation integrity, budgets, or approvals.
  Hooks may add defense in depth only.
- Keep the agent surface small. Adding an agent operation requires a clear need
  that existing operations cannot meet.
- Unsupported paths fail clearly. Never simulate success; label smoke results
  as smoke.
- Keep model, dataset, evaluation, run, and artifact identities explicit and
  digest-pinned across boundaries.
- Prefer maintained open-source tools (TRL, NeMo AutoModel, NeMo Evaluator,
  OpenEnv) over platform-specific reimplementations.
- Do not add features, abstractions, or configuration beyond what the current
  milestone needs.
- Public files must not contain personal hostnames, hardware inventories,
  filesystem paths, or credentials. Describe only behavior proven by tests.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run black --check .
```

Write tests first for new behavior. Run ruff and black before committing.
Commit messages use conventional commits.
