import type {
  CampaignEvent,
  CampaignSummary,
  Dashboard,
  InboxItem,
  MetricPoint,
} from "./types.ts";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface Api {
  campaigns(): Promise<CampaignSummary[]>;
  dashboard(campaignId: string): Promise<Dashboard>;
  inbox(): Promise<InboxItem[]>;
  trainingMetrics(experimentId: string, after: number): Promise<{ points: MetricPoint[]; next: number }>;
  decide(approvalId: string, grant: boolean): Promise<unknown>;
  setGuidance(campaignId: string, text: string): Promise<unknown>;
  pause(campaignId: string): Promise<unknown>;
  resume(campaignId: string): Promise<unknown>;
  stream(campaignId: string, after: number, onEvent: (event: CampaignEvent) => void, signal: AbortSignal): Promise<number>;
}

export function createApi(token: string, base = ""): Api {
  async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await fetch(`${base}/v1${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const data = (await response.json()) as { detail?: unknown };
        if (typeof data.detail === "string") detail = data.detail;
      } catch {
        // keep the status text
      }
      throw new ApiError(response.status, detail);
    }
    return (await response.json()) as T;
  }

  return {
    campaigns: () => call("GET", "/campaigns"),
    dashboard: (id) => call("GET", `/campaigns/${encodeURIComponent(id)}/dashboard`),
    inbox: () => call("GET", "/approvals"),
    trainingMetrics: (id, after) =>
      call("GET", `/experiments/${encodeURIComponent(id)}/training-metrics?after=${after}`),
    decide: (id, grant) => call("POST", `/approvals/${encodeURIComponent(id)}/decision`, { grant }),
    setGuidance: (id, text) => call("PUT", `/campaigns/${encodeURIComponent(id)}/guidance`, { text }),
    pause: (id) => call("POST", `/campaigns/${encodeURIComponent(id)}/pause`),
    resume: (id) => call("POST", `/campaigns/${encodeURIComponent(id)}/resume`),
    async stream(id, after, onEvent, signal) {
      const response = await fetch(
        `${base}/v1/campaigns/${encodeURIComponent(id)}/stream?after=${after}`,
        { headers: { Authorization: `Bearer ${token}` }, signal },
      );
      if (!response.ok || !response.body) throw new ApiError(response.status, "stream unavailable");
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      let cursor = after;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) return cursor;
        buffer += value;
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = block
            .split("\n")
            .filter((line) => line.startsWith("data: "))
            .map((line) => line.slice(6))
            .join("\n");
          if (data) {
            const event = JSON.parse(data) as CampaignEvent;
            cursor = event.seq;
            onEvent(event);
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    },
  };
}
