# Evaluation harness

One harness runs every evaluation suite as an `evaluate` stage:

```text
{python} -m bashgym_autoresearch.runners.eval_harness {script} {run_dir} {inputs}
```

`{script}` is the suite config JSON. The stage profile pins it by SHA-256, so the
config, and through it the dataset and sandbox image, cannot change during a
campaign. Evidence also records a digest of the harness and suite source code;
results from a different evaluator version are not compared.

## Suites

| Suite | Task | Grading |
| --- | --- | --- |
| `coding` | HumanEval-style function completion (raw continuation prompt) | `prompt + completion + tests + check(entry_point)` runs in the sandbox; exit 0 passes, a timeout fails. |
| `data_analysis` | SmolDataEnvs questions about real data files (chat prompt) | The last fenced Python block runs in the sandbox with the task's data mounted read-only at `/home/user/input`; the last printed line is graded on the host with the vendored SmolDataEnvs grader. Shell-command-shaped answers fail. |

Clusters for the bootstrap are the task (`coding`) or the data folder
(`data_analysis`), because questions about the same data are correlated. Per-tier
pass rates are reported as `pass_rate.<tier>`.

## Config

```json
{
  "schema": "bashgym_autoresearch.eval_config.v1",
  "suite": "data_analysis",
  "scope": "development",
  "dataset": "/abs/path/tasks.jsonl",
  "dataset_sha256": "<sha256 of tasks.jsonl; must equal the campaign's dataset_sha256>",
  "expected_task_count": 250,
  "base_model": "/abs/path/base-checkpoint",
  "generation": {"dtype": "bfloat16", "device": "cuda", "max_new_tokens": 1024,
                 "chat_template_kwargs": {"enable_thinking": false}},
  "sandbox": {"image": "<image@sha256:...>", "memory": "8g", "cpus": 2,
              "program_timeout_seconds": 90},
  "data_analysis": {"data_root": "/abs/path/staged-data", "math_verify": true}
}
```

- The base model is evaluated for the baseline; candidates use the model their
  training stage produced. Model directories must be full checkpoints; adapter-only
  directories are refused.
- Generation is greedy and local (transformers, installed with the `eval` extra
  plus a PyTorch build for your platform).
- With `math_verify: true`, `math-verify` must be installed; the grader fails
  instead of silently skipping that tier.

## Datasets

- `bashgym_autoresearch.datasets.coding`: `prepare_mbpp_sanitized` and
  `prepare_humaneval_plus` turn pinned sources into task rows without reference
  solutions.
- `bashgym_autoresearch.datasets.smoldataenvs`: `stage_bucket_prefixes`
  downloads task data from the Hub bucket with a size and SHA-256 manifest, and
  `prepare_smoldataenvs` builds task rows with the upstream prompt. Before the
  first task, the harness checks that each data folder contains exactly the
  pinned files.

Serialize rows with `datasets.coding.encode_jsonl` and use the SHA-256 of those
bytes as the campaign's `dataset_sha256`.

## Sandbox

`sandbox-images/python-analysis` builds the offline image used by both suites.
Its analysis packages match the FineEnvs SmolDataEnvs reference sandbox, and its
base image is pinned by a multi-arch digest. Build it on the evaluation host and
reference it by digest. Programs run with no network, a read-only root, an
unprivileged user, no capabilities, and bounded memory, CPU, processes, time, and
output. Docker problems make the evaluation incomplete rather than failing tasks.

## Failure semantics

The first infrastructure error, generation timeout, or model error stops the
evaluation and writes evidence with `complete: false`, so the experiment is
`incomplete` and tasks are never dropped from the denominator. Configuration and
input problems (digest mismatch, task count, missing data) also produce
incomplete evidence before any generation.
