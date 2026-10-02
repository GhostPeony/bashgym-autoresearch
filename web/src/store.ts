import { create } from "zustand";

import { ApiError, createApi, type Api } from "./api.ts";
import type { CampaignSummary, Dashboard, InboxItem, MetricPoint } from "./types.ts";

const TOKEN_KEY = "bgar.token";

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

interface State {
  token: string | null;
  api: Api | null;
  campaigns: CampaignSummary[];
  selected: string | null;
  dashboard: Dashboard | null;
  inbox: InboxItem[];
  metrics: Record<string, { points: MetricPoint[]; next: number }>;
  error: string | null;
  live: boolean;
  signIn(token: string, remember: boolean): Promise<void>;
  signOut(): void;
  select(campaignId: string): Promise<void>;
  refresh(): Promise<void>;
  loadMetrics(experimentId: string): Promise<void>;
  decide(approvalId: string, grant: boolean): Promise<void>;
  saveGuidance(text: string): Promise<void>;
  togglePause(): Promise<void>;
}

let streamAbort: AbortController | null = null;

export const useStore = create<State>((set, get) => {
  const fail = (error: unknown) => {
    if (error instanceof ApiError && error.status === 401) {
      get().signOut();
      set({ error: "That token was not accepted. Paste a valid token to continue." });
      return;
    }
    set({ error: error instanceof Error ? error.message : String(error) });
  };

  const follow = (campaignId: string, after: number) => {
    streamAbort?.abort();
    const controller = new AbortController();
    streamAbort = controller;
    const api = get().api;
    if (!api) return;
    let refreshing = false;
    const loop = async (cursor: number) => {
      while (!controller.signal.aborted) {
        try {
          set({ live: true });
          cursor = await api.stream(
            campaignId,
            cursor,
            () => {
              if (refreshing) return;
              refreshing = true;
              void get()
                .refresh()
                .finally(() => {
                  refreshing = false;
                });
            },
            controller.signal,
          );
        } catch {
          if (controller.signal.aborted) return;
          set({ live: false });
          await new Promise((resolve) => setTimeout(resolve, 3000));
        }
      }
    };
    void loop(after);
  };

  const initialToken = readToken();
  return {
    token: initialToken,
    api: initialToken ? createApi(initialToken) : null,
    campaigns: [],
    selected: null,
    dashboard: null,
    inbox: [],
    metrics: {},
    error: null,
    live: false,

    async signIn(token, remember) {
      const api = createApi(token.trim());
      try {
        const [campaigns, inbox] = await Promise.all([api.campaigns(), api.inbox()]);
        try {
          (remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token.trim());
        } catch {
          // storage unavailable; the session still works until reload
        }
        set({ token: token.trim(), api, campaigns, inbox, error: null });
        const first = campaigns.at(-1);
        if (first) await get().select(first.campaign_id);
      } catch (error) {
        set({ error: error instanceof ApiError && error.status === 401 ? "That token was not accepted." : String(error) });
      }
    },

    signOut() {
      streamAbort?.abort();
      try {
        localStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem(TOKEN_KEY);
      } catch {
        // nothing stored
      }
      set({ token: null, api: null, campaigns: [], dashboard: null, selected: null, inbox: [] });
    },

    async select(campaignId) {
      set({ selected: campaignId, dashboard: null, metrics: {} });
      await get().refresh();
      const dashboard = get().dashboard;
      if (dashboard) follow(campaignId, dashboard.event_seq);
    },

    async refresh() {
      const { api, selected } = get();
      if (!api) return;
      try {
        const [campaigns, inbox, dashboard] = await Promise.all([
          api.campaigns(),
          api.inbox(),
          selected ? api.dashboard(selected) : Promise.resolve(null),
        ]);
        set({ campaigns, inbox, dashboard, error: null });
        const active = dashboard?.experiments.findLast((e) => e.stages.some((s) => s.kind === "train"));
        if (active) await get().loadMetrics(active.experiment_id);
      } catch (error) {
        fail(error);
      }
    },

    async loadMetrics(experimentId) {
      const { api, metrics } = get();
      if (!api) return;
      const current = metrics[experimentId] ?? { points: [], next: 0 };
      try {
        const page = await api.trainingMetrics(experimentId, current.next);
        if (page.points.length === 0 && metrics[experimentId]) return;
        set({
          metrics: {
            ...get().metrics,
            [experimentId]: { points: [...current.points, ...page.points], next: page.next },
          },
        });
      } catch (error) {
        fail(error);
      }
    },

    async decide(approvalId, grant) {
      const { api } = get();
      if (!api) return;
      try {
        await api.decide(approvalId, grant);
        await get().refresh();
      } catch (error) {
        fail(error);
      }
    },

    async saveGuidance(text) {
      const { api, selected } = get();
      if (!api || !selected) return;
      try {
        await api.setGuidance(selected, text);
        await get().refresh();
      } catch (error) {
        fail(error);
      }
    },

    async togglePause() {
      const { api, selected, dashboard } = get();
      if (!api || !selected || !dashboard) return;
      try {
        await (dashboard.status === "paused" ? api.resume(selected) : api.pause(selected));
        await get().refresh();
      } catch (error) {
        fail(error);
      }
    },
  };
});
