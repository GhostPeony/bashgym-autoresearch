# Deployment

Run the service on the machine that has the GPU, as its own OS user, and let
agents and people connect to it over HTTP. Stages then run locally next to the
data and the accelerator, and the agent never shares a filesystem with the
service's state (see [SECURITY.md](SECURITY.md)).

```text
agent (any harness, any machine) ──HTTP──┐
browser (laptop, phone)          ──HTTP──┼──► bashgym-ar serve  (GPU host, dedicated user)
                                         │      ├─ worker: training + evaluation stages
                                         │      └─ state directory (0700)
```

## On the GPU host

1. Create a dedicated user and install as that user:

   ```bash
   sudo useradd --create-home --shell /bin/bash bgar
   sudo -iu bgar
   git clone https://github.com/GhostPeony/bashgym-autoresearch.git
   cd bashgym-autoresearch
   uv sync --extra train --extra eval
   uv pip install torch --index-url <the PyTorch index for your CUDA version and architecture>
   ```

2. Build the evaluation sandbox and note its digest. The service user needs
   access to Docker:

   ```bash
   docker build -t bgar/python-analysis sandbox-images/python-analysis
   docker image inspect bgar/python-analysis --format '{{.Id}}'
   ```

3. Initialize and serve. The first human token is written to the state
   directory; copy it somewhere only you can read:

   ```bash
   uv run bashgym-ar init
   uv run bashgym-ar serve --host 127.0.0.1 --port 8765
   ```

   To keep it running, use a systemd user service or `tmux`. Binding to
   `127.0.0.1` keeps the API off the network; reach it through an SSH tunnel.

## From your machine

```bash
ssh -N -L 8765:127.0.0.1:8765 <gpu-host>
export BGAR_URL=http://127.0.0.1:8765
export BGAR_TOKEN=<human token>
bashgym-ar token agent --label claude-code      # give this token to the agent
```

Register stage profiles with absolute paths on the GPU host (the profile's
`script` is hashed there), create the campaign, and approve the start when the
agent requests it. The agent's harness runs `bashgym-ar mcp` locally with
`BGAR_URL` and its agent token.

## Stage profiles

| Stage | argv | script (pinned) |
| --- | --- | --- |
| Training | `{python} -m bashgym_autoresearch.runners.train_trl {script} {run_dir} {inputs}` | train config JSON |
| Evaluation | `{python} -m bashgym_autoresearch.runners.eval_harness {script} {run_dir} {inputs}` | suite config JSON |

Set `cost_per_hour` on each profile to charge the budget for measured stage
time, and list any variables the stage needs (for example `HF_HOME`) in
`env_passthrough`.
