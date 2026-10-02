# Architecture (planned)

This document describes the design. Components marked **(implemented)** exist
and are tested; everything else is still a plan.

## The loop

```text
human: objective, fixed evaluation suite, budget, stop rule  ── approve start
                                │
agent (any harness) ── brief ──►│◄── guidance (human-edited, read every iteration)
        │                       │
        └── propose one change ─► platform validates proposal and budget
                                │
                       train candidate (TRL / NeMo AutoModel, local or SSH)
                                │
                       sealed evaluation on the identical suite
                                │
              keep / discard / inconclusive (paired bootstrap CI, protected gates)
                                │
                     repeat until stop rule ── report ── human approval
```

The agent never grades its own work. Proposals, results, and decisions are
recorded in one SQLite database with versioned rows and idempotent writes, so a
campaign survives restarts and can be resumed by any harness.

## Agent operations

The same operations are exposed through MCP, the CLI, and HTTP:

| Operation | Purpose |
| --- | --- |
| `brief` | Next action, incumbent, recent results with confidence intervals, budget, guidance, pending approvals |
| `wait` | Long-poll until state changes |
| `propose` | Submit a baseline or a candidate with exactly one declared change |
| `results` | Results with intervals and per-task outcomes |
| `failures` | A bounded failure packet for the latest attempt |
| `pause` / `resume` / `cancel` | Campaign control |
| `request_approval` | Ask a human to start, raise the budget, promote, or publish |
| `report` | Final report, including an independent audit of the run |

## Enforcement

- **Credentials:** agent credentials can request but not grant start, budget,
  promotion, and publication. Human credentials grant.
- **Evaluation pinning:** evaluator code, datasets, task data, and sandbox
  images are pinned by SHA-256. Runners verify every digest before generating.
- **Sandboxing:** programs run in an offline container with read-only data.
  Grading happens outside the container.
- **Sealing:** results are sealed with an HMAC and can be recomputed from saved
  per-task outputs.
- **Budgets:** a per-stage wall-clock limit, a per-campaign cost budget, and a
  stop rule are checked by the platform on every step.

## Components

| Component | Responsibility |
| --- | --- |
| `contracts` **(implemented)** | Frozen data models and canonical hashing |
| `store` **(implemented)** | SQLite persistence with optimistic versioning, idempotency, and an event log |
| `decision` **(implemented)** | Keep/discard/inconclusive rule, next-action selection, single-change check |
| `stats` **(implemented)** | Clustered paired bootstrap |
| `seal`, `evidence` **(implemented)** | Result sealing and evaluation verification |
| `worker` **(implemented)** | Mechanical progression: launch, observe, seal, decide, stop rules |
| `executors` | Local execution with restart-safe run identity and deadlines **(implemented)**; remote compute is handled by running the service on the GPU host ([DEPLOYMENT.md](DEPLOYMENT.md)); HF Jobs for multi-GPU runs is planned |
| `runners` | Generic evaluation harness with coding and data-analysis suites, and TRL LoRA SFT training **(implemented, GPU run pending)**; NeMo AutoModel training planned |
| `auth`, `service` **(implemented)**; `notify` | Human gates and guidance (implemented); check-in digests (planned) |
| `traces` | Agent-session import, redaction, quality classification, training export |
| `api`, `mcp_server`, `cli` **(implemented)** | The agent and human surfaces |
| `web/` | Browser dashboards generated from campaign state |

## Milestones

1. **Core loop (done):** contracts, store, decision, statistics, sealing, approvals,
   and the agent operations, proven end to end with a labeled smoke stage.
2. **Evaluation harness (done):** one task-suite interface, with a coding suite and a
   data-analysis suite (SmolDataEnvs).
3. **Training (in progress):** TRL LoRA SFT runner and GPU-host deployment;
   NeMo AutoModel and HF Jobs later.
4. **Traces:** import, redaction, classification, and training export.
5. **Dashboards:** a web interface for campaign, training, evaluation, agent,
   and approval state.
6. **Long runs:** harness setup guides, check-in notifications, and run audits.
7. **Calibration:** a full campaign checked against published reference results.
