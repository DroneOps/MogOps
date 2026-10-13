// Display scale shared by every engine. Each engine's raw score is mapped by
// its rank among reference faces (SCUT-FBP5500 + MEBeauty) onto the target
// curve below, via knots fitted in training (common.display_knots). Twin of
// training/common.py TARGET_* and display_score().

/** Share of faces at or below each score (owner's pick, 2026-10-13). */
const TARGET_PERCENTILES = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.75, 0.85, 0.92, 0.97, 1];
const TARGET_SCORES = [0, 1, 2, 3, 6, 7, 8, 8.5, 9, 9.5, 10];

export type Knots = { raw: number[]; score: number[] };

/** Same as numpy.interp: linear, clamped to the end values. */
export function interp(v: number, xs: number[], ys: number[]): number {
  if (v <= xs[0]) return ys[0];
  if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (xs[i] < v) i++;
  const t = (v - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

export const displayScore = (raw: number, knots: Knots) => interp(raw, knots.raw, knots.score);

// ---------------------------------------------------------------------------
// PSL: the looksmaxxing forums' 1-8 scale, where 4 is average and each point
// is roughly one standard deviation. We read it off the 0-10 score's rank.

/** Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9). */
function probit(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = (q: number) =>
    (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
  if (p > 1 - 0.02425) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Share of reference faces scoring at or below `score`. */
export const percentileOf = (score: number) => interp(score, TARGET_SCORES, TARGET_PERCENTILES);

export function pslFromScore(score: number): number {
  const p = percentileOf(score);
  if (p <= 0) return 1;
  if (p >= 1) return 8;
  return Math.min(8, Math.max(1, 4 + probit(p)));
}

const TIERS: [number, string][] = [
  [7, "Gigachad"],
  [6, "Chad"],
  [5.25, "Chadlite"],
  [4.5, "High-tier normie"],
  [3.5, "Mid-tier normie"],
  [-Infinity, "Low-tier normie"],
];

export const pslTier = (psl: number) => TIERS.find(([min]) => psl >= min)![1];
