# Smoke campaign

A GPU-free campaign that exercises the whole loop. `train.py` writes a "model"
whose only weight is the recipe's `boost`; `evaluate.py` passes a task from
`tasks.json` when its difficulty is below `0.5 + boost`. Results are labelled
`smoke`, so they prove the wiring, not model quality, and cannot be promoted.

```bash
bashgym-ar init
export BGAR_TOKEN=$(cat ~/.bashgym-autoresearch/human.token)
bashgym-ar serve &

bashgym-ar profile smoke-eval --kind evaluate --script examples/smoke/evaluate.py --timeout 120
bashgym-ar profile smoke-train --kind train --script examples/smoke/train.py --timeout 120
```

Create `spec.json`, replacing the digest with the output of
`sha256sum examples/smoke/tasks.json`:

```json
{
  "name": "smoke",
  "objective": "Raise the smoke pass rate with one controlled change at a time.",
  "primary": {"name": "pass_rate", "direction": "maximize"},
  "minimum_improvement": 0.05,
  "stop": {"max_experiments": 4, "max_cost": 4},
  "evaluation": {"suite_id": "smoke-tasks", "dataset_sha256": "<sha256 of tasks.json>", "profile": "smoke-eval"},
  "train_profile": "smoke-train"
}
```

```bash
bashgym-ar campaign create spec.json         # prints the campaign id
```

Give an agent a token (`bashgym-ar token agent`, run with your human token)
and point it at
[docs/AGENT_GUIDE.md](../../docs/AGENT_GUIDE.md). It will request `start`;
approve it with `bashgym-ar approve <approval_id>`. `tests/test_e2e_smoke.py`
runs this campaign automatically.
