import { lossPath } from "../model.ts";
import type { MetricPoint } from "../types.ts";

export function LossCurve({ points }: { points: MetricPoint[] }) {
  const width = 360;
  const height = 120;
  const path = lossPath(points, width, height);
  const last = points.findLast((p) => typeof p.loss === "number");
  if (!path) {
    return <p className="muted">Loss appears here as soon as the training stage logs its first steps.</p>;
  }
  return (
    <figure className="loss">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training loss by step">
        <path className="loss-line" d={path} />
      </svg>
      <figcaption>
        Loss <span className="num">{last?.loss?.toFixed(4)}</span> at step{" "}
        <span className="num">{last?.step}</span>
      </figcaption>
    </figure>
  );
}
