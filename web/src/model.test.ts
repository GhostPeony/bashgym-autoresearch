import assert from "node:assert/strict";
import { test } from "node:test";

import {
  describeChange,
  elapsed,
  forestScale,
  formatSigned,
  lossPath,
} from "./model.ts";
import type { Experiment } from "./types.ts";

function experiment(overrides: Partial<Experiment>): Experiment {
  return {
    experiment_id: "e",
    seq: 1,
    role: "candidate",
    status: "evaluated",
    change: { variable: "lr", before: 0.0001, after: 0.0002 },
    hypothesis: "h",
    decision: "keep",
    improvement: 0.1,
    ci_low: 0.05,
    ci_high: 0.15,
    breached_gate: null,
    reason: null,
    metrics: null,
    scope: "smoke",
    created_at: "2026-10-01T00:00:00Z",
    estimated_cost: 1,
    stages: [],
    ...overrides,
  };
}

test("forest scale always includes zero and the minimum improvement", () => {
  const scale = forestScale([experiment({ ci_low: 0.2, ci_high: 0.4 })], 0.1, 400);
  assert.ok(scale.min < 0 && scale.max > 0.4);
  assert.ok(scale.x(0) > 0 && scale.x(0.4) < 400);
  assert.ok(scale.ticks.includes(0));
});

test("forest scale handles an empty or degenerate range", () => {
  const scale = forestScale([], 0, 100);
  assert.ok(scale.max > scale.min);
  assert.ok(Number.isFinite(scale.x(0)));
});

test("loss path needs two finite points and spans the box", () => {
  assert.equal(lossPath([{ step: 1, loss: 1 }], 100, 50), null);
  const path = lossPath(
    [
      { step: 1, loss: 2 },
      { step: 2, loss: Number.NaN },
      { step: 3, loss: 1 },
    ],
    100,
    50,
  );
  assert.equal(path, "M0.0,0.0 L100.0,50.0");
});

test("formatting and change descriptions", () => {
  assert.equal(formatSigned(0.0123), "+0.012");
  assert.equal(formatSigned(-0.5, 1), "−0.5");
  assert.equal(formatSigned(null), "—");
  assert.equal(describeChange(experiment({})), "lr: 0.0001 to 0.0002");
  assert.equal(describeChange(experiment({ role: "baseline", change: null })), "Starting point");
});

test("elapsed time reads in seconds, minutes, then hours", () => {
  const now = new Date("2026-10-01T02:05:00Z");
  assert.equal(elapsed("2026-10-01T02:04:30Z", now), "30s");
  assert.equal(elapsed("2026-10-01T01:55:00Z", now), "10m");
  assert.equal(elapsed("2026-10-01T00:00:00Z", now), "2h 5m");
});
