import { useEffect, useState } from "react";

import { decisionText, describeChange, elapsed, formatSigned, formatValue } from "../model.ts";
import { useStore } from "../store.ts";
import type { Dashboard, Experiment, InboxItem } from "../types.ts";
import { LossCurve } from "./LossCurve.tsx";
import { MarkGlyph } from "./Mark.tsx";

const approvalText: Record<InboxItem["kind"], string> = {
  start: "Start the campaign",
  budget: "Raise the budget",
  promote: "Promote a result",
  publish: "Publish a result",
};

export function Inbox({ items }: { items: InboxItem[] }) {
  const decide = useStore((s) => s.decide);
  const [busy, setBusy] = useState<string | null>(null);
  if (items.length === 0) return null;
  return (
    <section className="inbox" aria-labelledby="inbox-heading">
      <h2 id="inbox-heading">Needs your decision</h2>
      <ul>
        {items.map((item) => (
          <li key={item.approval_id}>
            <div>
              <strong>{approvalText[item.kind]}</strong>
              <span className="muted">
                {" "}
                for {item.campaign_name}
                {item.kind === "budget" && typeof item.payload.amount === "number"
                  ? `, by ${item.payload.amount}`
                  : ""}
                , requested by {item.requested_by}
              </span>
            </div>
            <div className="actions">
              <button
                type="button"
                className="primary"
                disabled={busy === item.approval_id}
                onClick={async () => {
                  setBusy(item.approval_id);
                  await decide(item.approval_id, true);
                  setBusy(null);
                }}
              >
                Approve
              </button>
              <button
                type="button"
                disabled={busy === item.approval_id}
                onClick={async () => {
                  setBusy(item.approval_id);
                  await decide(item.approval_id, false);
                  setBusy(null);
                }}
              >
                Deny
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function activeTraining(dashboard: Dashboard): Experiment | undefined {
  return dashboard.experiments.findLast((e) => e.stages.some((s) => s.kind === "train"));
}

export function Training({ dashboard }: { dashboard: Dashboard }) {
  const metrics = useStore((s) => s.metrics);
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 15_000);
    return () => clearInterval(timer);
  }, []);
  const experiment = activeTraining(dashboard);
  const running = dashboard.experiments.find((e) => e.experiment_id === dashboard.active_experiment?.experiment_id);
  return (
    <section aria-labelledby="training-heading">
      <h2 id="training-heading">Training</h2>
      {experiment ? (
        <>
          <p className="muted">
            #{experiment.seq} {describeChange(experiment)}
          </p>
          <LossCurve points={metrics[experiment.experiment_id]?.points ?? []} />
        </>
      ) : (
        <p className="muted">No candidate has been trained yet.</p>
      )}
      {running && (
        <ol className="stages">
          {running.stages.map((stage) => (
            <li key={stage.kind} className={`stage stage-${stage.status}`}>
              <span>{stage.kind === "train" ? "Training" : "Evaluation"}</span>
              <span className="muted">
                {stage.status === "running" ? `running for ${elapsed(stage.created_at, now)}` : stage.status}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export function Guidance({ dashboard }: { dashboard: Dashboard }) {
  const save = useStore((s) => s.saveGuidance);
  const [text, setText] = useState(dashboard.guidance.text);
  const [saving, setSaving] = useState(false);
  useEffect(() => setText(dashboard.guidance.text), [dashboard.guidance.version, dashboard.guidance.text]);
  const changed = text !== dashboard.guidance.text;
  return (
    <section aria-labelledby="guidance-heading">
      <h2 id="guidance-heading">Guidance for the agent</h2>
      <p className="muted">The agent reads this before every experiment.</p>
      <textarea
        aria-labelledby="guidance-heading"
        value={text}
        rows={5}
        onChange={(event) => setText(event.target.value)}
        placeholder="For example: focus on the hard tier; keep learning rates below 3e-4."
      />
      <div className="actions">
        <button
          type="button"
          className="primary"
          disabled={!changed || saving}
          onClick={async () => {
            setSaving(true);
            await save(text);
            setSaving(false);
          }}
        >
          {saving ? "Saving" : "Save guidance"}
        </button>
        <span className="muted">Version {dashboard.guidance.version}</span>
      </div>
    </section>
  );
}

const eventText: Record<string, (p: Record<string, unknown>) => string> = {
  campaign_created: () => "Campaign created",
  approval_requested: (p) => `Agent asked: ${approvalText[p.kind as InboxItem["kind"]] ?? String(p.kind)}`,
  approval_decided: (p) =>
    `${p.granted ? "Approved" : "Denied"}: ${approvalText[p.kind as InboxItem["kind"]] ?? String(p.kind)}`,
  experiment_proposed: (p) => `Agent proposed a ${String(p.role)}`,
  stage_launched: (p) => `${p.kind === "train" ? "Training" : "Evaluation"} started`,
  experiment_decided: (p) => `Result: ${decisionText[p.decision as keyof typeof decisionText] ?? String(p.decision)}`,
  guidance_updated: (p) => `Guidance updated to version ${String(p.version)}`,
  paused: (p) => `Paused by ${String(p.by)}`,
  resumed: (p) => `Resumed by ${String(p.by)}`,
  cancelled: (p) => `Cancelled by ${String(p.by)}`,
  campaign_stopped: (p) => `Stopped: ${String(p.reason)}`,
};

export function Activity({ dashboard }: { dashboard: Dashboard }) {
  const recent = [...dashboard.events].reverse().slice(0, 12);
  return (
    <section aria-labelledby="activity-heading">
      <h2 id="activity-heading">Activity</h2>
      <ol className="activity">
        {recent.map((event) => (
          <li key={event.seq}>
            <time dateTime={event.at}>{new Date(event.at).toLocaleTimeString()}</time>
            <span>{eventText[event.type]?.(event.payload) ?? event.type}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function Ledger({ dashboard }: { dashboard: Dashboard }) {
  const metric = dashboard.primary_metric.name;
  return (
    <section aria-labelledby="ledger-heading">
      <h2 id="ledger-heading">Experiments</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Change and hypothesis</th>
              <th scope="col">Result</th>
              <th scope="col" className="right">
                {metric}
              </th>
              <th scope="col" className="right">
                Improvement (95% interval)
              </th>
            </tr>
          </thead>
          <tbody>
            {[...dashboard.experiments].reverse().map((e) => (
              <tr key={e.experiment_id}>
                <td className="num">{e.seq}</td>
                <td>
                  <div>{describeChange(e)}</div>
                  <div className="muted">{e.hypothesis}</div>
                  {e.reason && e.decision !== "keep" && e.decision !== "baseline" && (
                    <div className="muted">{e.reason}</div>
                  )}
                </td>
                <td>
                  {e.decision ? (
                    <span className={`decision decision-${e.decision}`}>
                      <MarkGlyph decision={e.decision} />
                      {decisionText[e.decision]}
                    </span>
                  ) : (
                    <span className="muted">{e.status === "queued" ? "Queued" : "Running"}</span>
                  )}
                </td>
                <td className="num right">{formatValue(e.metrics?.[metric])}</td>
                <td className="num right">
                  {e.ci_low === null
                    ? "—"
                    : `${formatSigned(e.improvement)} (${formatSigned(e.ci_low)}, ${formatSigned(e.ci_high)})`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
