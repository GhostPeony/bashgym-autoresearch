import { useEffect, useState } from "react";

import { ForestPlot } from "./components/ForestPlot.tsx";
import { Activity, Guidance, Inbox, Ledger, Training } from "./components/Panels.tsx";
import { describeNextAction, formatValue } from "./model.ts";
import { useStore } from "./store.ts";

function SignIn() {
  const signIn = useStore((s) => s.signIn);
  const error = useStore((s) => s.error);
  const [token, setToken] = useState("");
  const [remember, setRemember] = useState(false);
  return (
    <main className="signin">
      <h1>AutoResearch</h1>
      <p>Paste a token to watch your campaigns. A human token can also approve requests and edit guidance.</p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void signIn(token, remember);
        }}
      >
        <label htmlFor="token">Token</label>
        <input
          id="token"
          type="password"
          autoComplete="off"
          value={token}
          onChange={(event) => setToken(event.target.value)}
        />
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} />
          Remember on this device
        </label>
        {error && <p role="alert" className="error">{error}</p>}
        <button type="submit" className="primary" disabled={!token.trim()}>
          Open dashboard
        </button>
      </form>
    </main>
  );
}

const statusText: Record<string, string> = {
  awaiting_start: "Awaiting start",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  cancelled: "Cancelled",
  exhausted: "Stopped",
};

export function App() {
  const token = useStore((s) => s.token);
  const campaigns = useStore((s) => s.campaigns);
  const selected = useStore((s) => s.selected);
  const dashboard = useStore((s) => s.dashboard);
  const inbox = useStore((s) => s.inbox);
  const error = useStore((s) => s.error);
  const live = useStore((s) => s.live);
  const select = useStore((s) => s.select);
  const refresh = useStore((s) => s.refresh);
  const signOut = useStore((s) => s.signOut);
  const togglePause = useStore((s) => s.togglePause);

  useEffect(() => {
    if (!token || selected) return;
    void (async () => {
      await refresh();
      const latest = useStore.getState().campaigns.at(-1);
      if (latest) await select(latest.campaign_id);
    })();
  }, [token, selected, refresh, select]);

  if (!token) return <SignIn />;

  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">AutoResearch</span>
        <label className="switcher">
          <span className="visually-hidden">Campaign</span>
          <select value={selected ?? ""} onChange={(event) => void select(event.target.value)}>
            {campaigns.length === 0 && <option value="">No campaigns yet</option>}
            {campaigns.map((c) => (
              <option key={c.campaign_id} value={c.campaign_id}>
                {c.name} ({statusText[c.status] ?? c.status})
              </option>
            ))}
          </select>
        </label>
        <span className={`live ${live ? "on" : ""}`}>{live ? "Live" : "Reconnecting"}</span>
        <button type="button" className="quiet" onClick={signOut}>
          Sign out
        </button>
      </header>

      {error && (
        <p role="alert" className="error banner">
          {error}
        </p>
      )}

      <main className="page">
        <Inbox items={inbox} />
        {dashboard ? (
          <>
            <section className="heading" aria-labelledby="objective">
              <h1 id="objective">{dashboard.objective}</h1>
              <p className="status-line">
                <strong>{statusText[dashboard.status] ?? dashboard.status}.</strong>{" "}
                {describeNextAction(dashboard.next_action.kind)}
                {dashboard.next_action.kind === "stop" ? ` (${dashboard.next_action.reason}).` : "."}
              </p>
              <dl className="facts">
                <div>
                  <dt>Best {dashboard.primary_metric.name}</dt>
                  <dd className="num">{formatValue(dashboard.incumbent?.value)}</dd>
                </div>
                <div>
                  <dt>Experiments</dt>
                  <dd className="num">{dashboard.experiments.length}</dd>
                </div>
                <div>
                  <dt>Budget used</dt>
                  <dd className="num">
                    {formatValue(dashboard.budget.used, 2)} of {formatValue(dashboard.budget.max, 2)}
                  </dd>
                </div>
              </dl>
              {dashboard.smoke_only && (
                <p className="note">These are smoke results: they check the wiring, not model quality.</p>
              )}
              {(dashboard.status === "running" || dashboard.status === "paused") && (
                <button type="button" onClick={() => void togglePause()}>
                  {dashboard.status === "paused" ? "Resume campaign" : "Pause campaign"}
                </button>
              )}
            </section>

            <section className="evidence" aria-labelledby="evidence-heading">
              <h2 id="evidence-heading">Evidence</h2>
              <ForestPlot dashboard={dashboard} />
            </section>

            <div className="columns">
              <Training dashboard={dashboard} />
              <Guidance dashboard={dashboard} />
              <Activity dashboard={dashboard} />
            </div>

            <Ledger dashboard={dashboard} />
          </>
        ) : (
          campaigns.length === 0 && (
            <p className="empty">
              No campaigns yet. Create one with <code>bashgym-ar campaign create spec.json</code>.
            </p>
          )
        )}
      </main>
    </div>
  );
}
