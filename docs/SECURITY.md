# Security model

bashgym-autoresearch assumes the agent driving a campaign may try, deliberately
or by accident, to improve its score by means other than a better model. Recent
studies of autonomous research agents found reward hacking in a large share of
open-ended tasks, most often through leaked test data or modified scorers. The
platform therefore enforces its rules in the service, not in the agent's
instructions.

## What the service enforces

| Rule | Mechanism |
| --- | --- |
| Humans grant, agents request | Agent tokens can request start, budget, promotion, and publication; only human tokens grant them, mint tokens, register stage programs, create campaigns, or edit guidance. A campaign paused by a human can only be resumed by a human. |
| One controlled change | A candidate's recipe may differ from the incumbent's only at the declared variable, with the declared before and after values. |
| Frozen evaluation | Stage programs are pinned by SHA-256 and snapshotted into each campaign at creation; a changed script is refused at launch. Evaluation stages never receive the agent's recipe. |
| Evidence binding | Each evaluation receives a context with a per-launch nonce; evidence must echo its digest. Evidence from a different evaluator version, task set, or cluster labelling than the incumbent's cannot be compared. |
| Sealed outputs | Stage outputs are hashed and signed with an HMAC key. The trained model is re-verified against its seal before evaluation launches and again before the decision. |
| Honest denominators | Infrastructure failures, timeouts, and verification failures make an experiment `incomplete`; too many incomplete experiments stop the campaign. |
| Statistical keep | A candidate is kept only when the bootstrap interval's lower bound clears the minimum improvement, enough independent clusters exist, and protected metrics hold against both the incumbent and the baseline. |
| Platform-measured cost | Each stage is charged its measured wall-clock time at the profile's human-set rate; the budget uses the larger of that and the agent's estimate. |
| Stage isolation | Stages run in their own process group (POSIX), receive only an allow-listed environment, and cannot write the files that record their exit status. Model programs in evaluation suites run in an offline, read-only Docker sandbox, and gold answers never enter it. |

## What deployment must provide

The service cannot protect its own state from a process that can read and write
its files. Run it so the agent cannot:

- **Run the service as a different OS user** than the agent, or on another
  machine, and give the agent only the API URL and an agent token. The state
  directory holds the database, the HMAC key, and the first human token; on
  POSIX `bashgym-ar init` creates it with mode 0700 and `serve` refuses to start
  if other users can read it.
- **Keep the human token away from the agent.** Mint agent tokens with
  `bashgym-ar token agent` and hand over only those.
- **Treat stage programs as trusted code.** Training and evaluation programs run
  on the host as the service user; only the model's generated programs are
  sandboxed. Register only programs you have reviewed.

On Windows, process-tree cleanup is best effort (no process groups), and
directory permissions are not checked.

## Reporting

Report vulnerabilities privately through GitHub's security advisory feature on
this repository rather than in a public issue.
