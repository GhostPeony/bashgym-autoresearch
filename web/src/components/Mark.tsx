import { decisionMark } from "../model.ts";
import type { Decision } from "../types.ts";

/** A decision symbol drawn at (x, y) inside an SVG. Shape and color both encode the decision. */
export function Mark({ decision, x, y, size = 6 }: { decision: Decision; x: number; y: number; size?: number }) {
  const className = `mark mark-${decision}`;
  switch (decisionMark[decision]) {
    case "square":
      return <rect className={className} x={x - size} y={y - size} width={size * 2} height={size * 2} />;
    case "circle":
      return <circle className={className} cx={x} cy={y} r={size} />;
    case "diamond":
      return (
        <polygon
          className={className}
          points={`${x},${y - size * 1.25} ${x + size * 1.25},${y} ${x},${y + size * 1.25} ${x - size * 1.25},${y}`}
        />
      );
    case "cross":
      return (
        <path
          className={`${className} mark-stroke`}
          d={`M${x - size},${y - size} L${x + size},${y + size} M${x + size},${y - size} L${x - size},${y + size}`}
        />
      );
    case "ring":
      return <circle className={`${className} mark-ring`} cx={x} cy={y} r={size} />;
  }
}

/** The same symbol as an inline legend glyph. */
export function MarkGlyph({ decision }: { decision: Decision }) {
  return (
    <svg className="glyph" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <Mark decision={decision} x={8} y={8} size={5} />
    </svg>
  );
}
