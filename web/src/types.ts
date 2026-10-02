export type Decision = "baseline" | "keep" | "discard" | "inconclusive" | "crash" | "incomplete";

export type NextActionKind =
  | "await_start"
  | "wait"
  | "propose_baseline"
  | "propose_candidate"
  | "stop";

export interface Change {
  variable: string;
  before: unknown;
  after: unknown;
}

export interface Stage {
  kind: "train" | "evaluate";
  profile: string;
  status: string;
  exit_code: number | null;
  reason: string | null;
  cost: number;
  created_at: string;
  updated_at: string;
}

export interface Experiment {
  experiment_id: string;
  seq: number;
  role: "baseline" | "candidate";
  status: string;
  change: Change | null;
  hypothesis: string;
  decision: Decision | null;
  improvement: number | null;
  ci_low: number | null;
  ci_high: number | null;
  breached_gate: string | null;
  reason: string | null;
  metrics: Record<string, number> | null;
  scope: "smoke" | "development" | null;
  created_at: string;
  estimated_cost: number;
  stages: Stage[];
}

export interface Approval {
  id: string;
  kind: "start" | "budget" | "promote" | "publish";
  status: "pending" | "granted" | "denied";
  payload: Record<string, unknown>;
  requested_by: string;
  decided_by: string | null;
  created_at: string;
}

export interface CampaignEvent {
  seq: number;
  type: string;
  payload: Record<string, unknown>;
  at: string;
}

export interface Dashboard {
  campaign_id: string;
  name: string;
  objective: string;
  status: string;
  next_action: { kind: NextActionKind; reason: string };
  primary_metric: { name: string; direction: "maximize" | "minimize" };
  minimum_improvement: number;
  incumbent: { experiment_id: string; value: number | null; scope: string } | null;
  active_experiment: { experiment_id: string; status: string } | null;
  budget: { used: number; max: number };
  experiments: Experiment[];
  approvals: Approval[];
  guidance: { text: string; version: number };
  smoke_only: boolean;
  event_seq: number;
  events: CampaignEvent[];
}

export interface CampaignSummary {
  campaign_id: string;
  name: string;
  status: string;
  created_at: string;
}

export interface InboxItem {
  approval_id: string;
  campaign_id: string;
  campaign_name: string;
  kind: Approval["kind"];
  payload: Record<string, unknown>;
  requested_by: string;
  created_at: string;
}

export interface MetricPoint {
  step?: number;
  loss?: number;
  [key: string]: number | undefined;
}
