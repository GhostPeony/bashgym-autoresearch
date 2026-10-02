import { useLayoutEffect, useRef, useState } from "react";

import { decisionText, describeChange, forestScale, formatSigned, formatValue } from "../model.ts";
import type { Dashboard } from "../types.ts";
import { Mark } from "./Mark.tsx";

const ROW = 52;
const TOP = 34;
const LABEL = 232;

/**
 * Each experiment's improvement over its incumbent with its confidence interval,
 * against the campaign's minimum improvement. Kept results sit clearly right of
 * the dashed line; inconclusive ones straddle it.
 */
export function ForestPlot({ dashboard }: { dashboard: Dashboard }) {
  const holder = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = holder.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(320, entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const rows = dashboard.experiments.filter((e) => e.decision !== null);
  if (width === 0) return <div className="forest" ref={holder} />;
  const narrow = width < 560;
  const label = narrow ? 0 : LABEL;
  const plotWidth = width - label - 24;
  const scale = forestScale(rows, dashboard.minimum_improvement, plotWidth);
  const height = TOP + Math.max(rows.length, 1) * ROW + 28;
  const at = (v: number) => label + scale.x(v);
  const metric = dashboard.primary_metric.name;

  if (rows.length === 0) {
    return (
      <div className="forest-empty" ref={holder}>
        <p>No results yet. Each experiment appears here once it has been evaluated.</p>
      </div>
    );
  }

  return (
    <div className="forest" ref={holder}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-labelledby="forest-title forest-desc"
      >
        <title id="forest-title">Improvement over the incumbent, per experiment</title>
        <desc id="forest-desc">
          {`Bars show 95% confidence intervals for the change in ${metric}. The dashed line is the minimum improvement of ${formatValue(dashboard.minimum_improvement)}.`}
        </desc>
        {scale.ticks.map((tick) => (
          <g key={tick}>
            <line className="forest-grid" x1={at(tick)} x2={at(tick)} y1={TOP - 8} y2={height - 24} />
            <text className="forest-tick num" x={at(tick)} y={height - 6} textAnchor="middle">
              {formatSigned(tick, 2)}
            </text>
          </g>
        ))}
        <line className="forest-zero" x1={at(0)} x2={at(0)} y1={TOP - 12} y2={height - 24} />
        <line
          className="forest-threshold"
          x1={at(dashboard.minimum_improvement)}
          x2={at(dashboard.minimum_improvement)}
          y1={TOP - 20}
          y2={height - 24}
        />
        <text className="forest-threshold-label" x={at(dashboard.minimum_improvement) + 6} y={TOP - 14}>
          minimum improvement
        </text>
        {rows.map((e, index) => {
          const y = TOP + index * ROW + ROW / 2;
          const decision = e.decision!;
          const hasInterval = e.ci_low !== null && e.ci_high !== null;
          const point = e.role === "baseline" ? 0 : e.improvement;
          return (
            <g key={e.experiment_id} className={`forest-row row-${decision}`}>
              <title>
                {`#${e.seq} ${decisionText[decision]}: ${describeChange(e)}. ${e.reason ?? ""}`}
              </title>
              {!narrow && (
                <>
                  <text className="forest-label" x={0} y={y - 4}>
                    <tspan className="num">#{e.seq}</tspan> {describeChange(e)}
                  </text>
                  <text className="forest-sub" x={0} y={y + 13}>
                    {decisionText[decision]}
                    {hasInterval ? `, ${formatSigned(e.ci_low)} to ${formatSigned(e.ci_high)}` : ""}
                  </text>
                </>
              )}
              {hasInterval && (
                <>
                  <line className="forest-bar" x1={at(e.ci_low!)} x2={at(e.ci_high!)} y1={y} y2={y} />
                  <line className="forest-cap" x1={at(e.ci_low!)} x2={at(e.ci_low!)} y1={y - 6} y2={y + 6} />
                  <line className="forest-cap" x1={at(e.ci_high!)} x2={at(e.ci_high!)} y1={y - 6} y2={y + 6} />
                </>
              )}
              {point !== null && point !== undefined ? (
                <Mark decision={decision} x={at(point)} y={y} />
              ) : (
                <Mark decision={decision} x={at(0)} y={y} size={5} />
              )}
              {narrow && (
                <text className="forest-sub" x={4} y={y - 12}>
                  #{e.seq} {decisionText[decision]}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
