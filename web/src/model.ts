import type { Decision, Experiment, MetricPoint } from "./types.ts";

/** Linear scale for the forest plot: always shows zero and the minimum improvement. */
export interface Scale {
  min: number;
  max: number;
  x: (value: number) => number;
  ticks: number[];
}

export function forestScale(experiments: Experiment[], minimum: number, width: number): Scale {
  const values = [0, minimum];
  for (const e of experiments) {
    if (e.ci_low !== null) values.push(e.ci_low);
    if (e.ci_high !== null) values.push(e.ci_high);
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (max - min < 1e-9) {
    min -= 0.05;
    max += 0.05;
  }
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  const span = max - min;
  const step = niceStep(span / 5);
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-12; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return { min, max, ticks, x: (value: number) => ((value - min) / span) * width };
}

function niceStep(raw: number): number {
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * power;
}

/** SVG path for a loss curve in a width x height box; null when there is nothing to draw. */
export function lossPath(points: MetricPoint[], width: number, height: number): string | null {
  const usable = points.filter(
    (p): p is MetricPoint & { step: number; loss: number } =>
      typeof p.step === "number" && typeof p.loss === "number" && Number.isFinite(p.loss),
  );
  if (usable.length < 2) return null;
  const steps = usable.map((p) => p.step);
  const losses = usable.map((p) => p.loss);
  const [s0, s1] = [Math.min(...steps), Math.max(...steps)];
  const [l0, l1] = [Math.min(...losses), Math.max(...losses)];
  const sx = (s: number) => (s1 === s0 ? 0 : ((s - s0) / (s1 - s0)) * width);
  const sy = (l: number) => (l1 === l0 ? height / 2 : height - ((l - l0) / (l1 - l0)) * height);
  return usable
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`)
    .join(" ");
}

export const decisionText: Record<Decision, string> = {
  baseline: "Baseline",
  keep: "Kept",
  discard: "Discarded",
  inconclusive: "Inconclusive",
  crash: "Crashed",
  incomplete: "Incomplete",
};

/** Every decision has a shape as well as a color, so meaning never depends on color alone. */
export const decisionMark: Record<Decision, "square" | "circle" | "diamond" | "cross" | "ring"> = {
  baseline: "square",
  keep: "circle",
  discard: "cross",
  inconclusive: "diamond",
  crash: "cross",
  incomplete: "ring",
};

export function formatValue(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function formatSigned(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export function describeChange(experiment: Experiment): string {
  if (experiment.role === "baseline" || !experiment.change) return "Starting point";
  const { variable, before, after } = experiment.change;
  return `${variable}: ${JSON.stringify(before ?? null)} to ${JSON.stringify(after ?? null)}`;
}

const nextActionText = {
  await_start: "Waiting for you to approve the start",
  wait: "An experiment is running",
  propose_baseline: "Waiting for the agent to propose a baseline",
  propose_candidate: "Waiting for the agent to propose the next change",
  stop: "Finished",
} as const;

export function describeNextAction(kind: keyof typeof nextActionText): string {
  return nextActionText[kind];
}

export function elapsed(fromIso: string, now: Date): string {
  const seconds = Math.max(0, Math.round((now.getTime() - new Date(fromIso).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}
