# Agent guide

This guide is for the agent running a campaign: Claude Code, Codex, OpenCode,
Hermes Agent, or any harness that can call MCP tools or a CLI. Every harness uses
the same ten operations and the platform enforces the same rules for all of them.

## Connect

A human starts the service and gives you an agent token:

```bash
bashgym-ar init                     # once; writes a human token to the state directory
bashgym-ar token agent --label my-agent
bashgym-ar serve                    # API on http://127.0.0.1:8765 plus the worker
```

Then either:

- **MCP:** run `bashgym-ar mcp` with `BGAR_TOKEN=<agent token>` (and `BGAR_URL`
  if the service is not local) as a stdio MCP server in your harness.
- **CLI:** set `BGAR_TOKEN` and call `bashgym-ar brief <campaign>` and the other
  verbs. Every command prints JSON.

## The loop

1. `brief`: read `next_action`, the incumbent, recent results, budget, and the
   human's `guidance`. Guidance can change between iterations; read it every time.
2. If `next_action.kind` is `propose_baseline`, propose the baseline without a
   change. If it is `propose_candidate`, propose a candidate that changes
   exactly one variable (`change.variable`, `before`, `after`) and states a
   hypothesis. Put every parameter the stages need in `recipe`.
3. `wait` until an `experiment_decided` event arrives. Pass the last seen
   event `seq` as `after_seq` to avoid replays.
4. Read `results`. On `crash` or `incomplete`, read `failures` for exit codes
   and the stderr tail, fix the cause, and propose again.
5. Repeat until `next_action.kind` is `stop`, then call `report`.

## Decisions

The platform decides; you do not. A candidate is compared with the incumbent
task by task, and a cluster bootstrap gives a confidence interval for the
improvement.

| Decision | Meaning |
| --- | --- |
| `keep` | The interval's lower bound is above zero and at least `minimum_improvement`; the candidate becomes the incumbent. |
| `discard` | The interval is below the minimum, or a protected metric regressed beyond its limit. |
| `inconclusive` | The interval overlaps the minimum. Do not report it as a gain; add tasks or repeats, or try a larger change. |
| `crash` | A stage exited with an error. |
| `incomplete` | Infrastructure failed, a stage timed out, or a pinned script changed. It does not count as an experiment. |

## Gates

Ask with `request_approval`; a human decides.

| Kind | When |
| --- | --- |
| `start` | The campaign is `awaiting_start`. |
| `budget` | A proposal would exceed the budget. Payload: `{"amount": <number>}`. |
| `promote`, `publish` | A kept, development-scope result is ready. Payload: `{"experiment_id": ...}`. Smoke-scope results are refused. |

You cannot grant approvals, edit guidance, register stage programs, or create
campaigns. A campaign a human paused can only be resumed by a human.
