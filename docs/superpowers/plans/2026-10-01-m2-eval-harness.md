# M2 Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One generic evaluation harness that runs as an `evaluate` stage, plus two task suites ported from BashGym:
- **coding:** HumanEval-style function completion;
- **data_analysis:** SmolDataEnvs.

Both produce M1 evidence that the worker can seal and decide on.

**Architecture:** An evaluate profile runs `python -m bashgym_autoresearch.runners.eval_harness {script} {run_dir} {inputs}`. Here `{script}` is a suite config JSON, which is what the profile pins by sha256. The harness:
1. checks the context and dataset digests;
2. loads tasks through the suite;
3. generates greedily with a local model (transformers, imported lazily);
4. grades each completion through the suite, using a fail-closed Docker sandbox that never pulls images;
5. stops at the first infrastructure error (`complete=false`);
6. writes `outputs/evaluation.json` (EvalEvidence) and `outputs/task_results.json`.

Per-task outcomes use the suite's cluster key. For SmolDataEnvs that is the data folder, because tasks on the same data are correlated.

**Tech Stack:** M1 plus the `docker` SDK. `transformers`/`torch` are optional (`[eval]` extra) and imported only when generating. `math-verify` is part of the `[eval]` extra.

**Spec:** `docs/ARCHITECTURE.md`. Sources:
- BashGym `campaigns/first_party_coding_runner.py`, `campaigns/data_analysis_runner.py`
- `environments/docker_coding.py`, `environments/short_answer_grader.py`
- `datasets/coding_benchmarks.py`, `datasets/data_analysis_benchmarks.py`

## Global Constraints

- The harness never pulls images. The sandbox image must be referenced by digest (`name@sha256:...` or `sha256:...`).
- Gold answers never enter the sandbox; data-analysis grading happens on the host.
- An infrastructure error stops evaluation and marks the evidence `complete=false`. Tasks are never dropped from the denominator.
- Config `dataset_sha256` must equal both the campaign's `context.dataset_sha256` and the sha256 of the dataset file.
- Evidence adds `provenance` (a string map): suite name, harness/suite source digest, sandbox image.

## Review Focus

1. **The model prints the gold answer by reading a file.** The gold answer is never mounted, and the data mount is read-only. Covered by the data_analysis sandbox spec test.
2. **A dataset file swapped after the campaign started.** Must be rejected before generation. Covered by a harness test.
3. **Docker unavailable or image missing.** Must produce `infrastructure_error` and `complete=false`, not a crash or zero scores. Covered by a sandbox test.
4. **A program that floods stdout or never exits.** Output is bounded and the timeout is enforced (counts as failed for data_analysis, `test_timeout` for coding). Covered by a sandbox test.
5. **A model directory that is a LoRA adapter.** Clear error before loading. Covered by a harness test.

---

### Task 1: Evidence provenance and grading ports
- Modify `contracts.EvalEvidence`: add `provenance: dict[str, str] = {}`, at most 32 entries.
- Create `grading/short_answer.py`: verbatim port of the vendored SmolDataEnvs grader, with its 17 tests and a differential check.
- Create `datasets/coding.py`: port `coding_benchmarks.py` (`prepare_mbpp_sanitized`, `prepare_humaneval_plus`, `encode_jsonl`), with tests.
- Create `datasets/smoldataenvs.py`: port `data_analysis_benchmarks.py` (`smoldataenvs_messages`, `prepare_smoldataenvs`, `stage_bucket_prefixes`), with tests.
- Commit.

### Task 2: Docker sandbox
- Create `sandbox/docker.py`, implementing `run_program(image, files, argv, *, workdir, read_only_mounts, memory, cpus, timeout, max_output) -> SandboxResult(exit_code, stdout, stderr, timed_out, error)`.
- Container settings: network none, read-only root, uid 65534, `no-new-privileges`, `cap_drop ALL`, a pids limit, and a tmpfs `/tmp`. Files are written to a temp workspace mounted at `/workspace`.
- Unavailable Docker or a missing image → `error` set, never raised.
- Tests use a fake Docker client covering the container options, output truncation, timeout kill, missing-image error and mount validation. A real-Docker test is marked `docker` and skipped when Docker is unavailable.
- Commit.

### Task 3: Suites and harness
- Create `runners/suites/base.py`: the `TaskSuite` protocol (`load`, `prompt`, `grade`) and a `Graded(status, prediction, method, cluster)` result.
- Create `runners/suites/coding.py`: raw prompt; `humaneval_body_v1` or `raw` completion protocol; grades with the `human_eval` checker inside the sandbox; statuses `passed`/`failed`/`test_timeout`/`infrastructure_error`.
- Create `runners/suites/data_analysis.py`:
  - chat messages;
  - last fenced python block;
  - the program runs in the sandbox with the data folder mounted read-only at `/home/user/input`;
  - the last stdout line is graded on the host;
  - shell-command-shaped answers fail.
  - Data files are verified against the task manifest (sha256) before the first task.
- Create `runners/eval_harness.py`:
  - a `HarnessConfig` model;
  - `run(config_path, run_dir, inputs, *, complete=None, sandbox=None)`;
  - `main()`;
  - model loading through transformers (raw or chat template), lazily imported; adapter-only directories are refused.
- Tests:
  - the harness runs with fake completion and grading;
  - digest mismatches are refused;
  - an infrastructure error stops evaluation with `complete=false`;
  - per-tier metrics and clusters are correct;
  - the evidence context digest is echoed;
  - each suite's grading has its own unit tests.
- Commit.

### Task 4: Real sandbox validation and docs
- Port `sandbox-images/smoldataenvs` and add a `sandbox-images/coding` image (Python + `human-eval`).
- Real-Docker canary run (local, marked): gold programs pass, wrong programs fail, the mount is read-only.
- Docs: `docs/EVALUATION.md` (suite config reference, pinning, sandbox) and README status.
- Commit and push.
