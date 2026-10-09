// Geometry scorer: facial ratios from MediaPipe landmarks, scored with
// per-trait curves learned from SCUT-FBP5500 ratings.
//
// computeMetrics() is a line-by-line twin of training/common.py
// geometry_metrics(); geometry.test.ts checks both give the same numbers.

import params from "./model-params.json";

export type Landmark = { x: number; y: number };
type Vec = [number, number];

export const METRIC_KEYS = [
  "symmetry",
  "canthalTilt",
  "eyeAspect",
  "eyeSpacing",
  "midfaceRatio",
  "jawRatio",
  "fwhr",
  "lowerThird",
] as const;
export type MetricKey = (typeof METRIC_KEYS)[number];
export type Metrics = Record<MetricKey, number>;

// MediaPipe Face Landmarker indices ("right" = subject's right).
export const L = {
  FOREHEAD: 10,
  GLABELLA: 9,
  NASION: 168,
  NOSE_TIP: 1,
  SUBNASALE: 2,
  UPPER_LIP: 0,
  LOWER_LIP: 17,
  CHIN: 152,
  R_EYE_OUTER: 33,
  R_EYE_INNER: 133,
  R_EYE_TOP: 159,
  R_EYE_BOTTOM: 145,
  L_EYE_OUTER: 263,
  L_EYE_INNER: 362,
  L_EYE_TOP: 386,
  L_EYE_BOTTOM: 374,
  R_BROW: 105,
  L_BROW: 334,
  R_ZYGION: 234,
  L_ZYGION: 454,
  R_GONION: 172,
  L_GONION: 397,
} as const;

const MIDLINE = [L.FOREHEAD, L.GLABELLA, L.NASION, L.NOSE_TIP, L.SUBNASALE, L.UPPER_LIP, L.LOWER_LIP, L.CHIN];

const SYMMETRY_PAIRS: ReadonlyArray<readonly [number, number]> = [
  [L.R_EYE_OUTER, L.L_EYE_OUTER],
  [L.R_EYE_INNER, L.L_EYE_INNER],
  [L.R_EYE_TOP, L.L_EYE_TOP],
  [L.R_EYE_BOTTOM, L.L_EYE_BOTTOM],
  [L.R_BROW, L.L_BROW],
  [70, 300], // outer brow
  [L.R_ZYGION, L.L_ZYGION],
  [116, 345], // cheekbone
  [L.R_GONION, L.L_GONION],
  [136, 365], // lower jaw
  [148, 377], // chin side
  [61, 291], // mouth corners
  [98, 327], // nostril wings
];

const DEG = 180 / Math.PI;

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Pixel-space points rotated so the eye line is horizontal (origin: right eye center). */
export function alignedPoints(landmarks: Landmark[], width: number, height: number): Vec[] {
  const px = landmarks.map((p): Vec => [p.x * width, p.y * height]);
  const center = (a: number, b: number): Vec => [(px[a][0] + px[b][0]) / 2, (px[a][1] + px[b][1]) / 2];
  const eyesR = center(L.R_EYE_OUTER, L.R_EYE_INNER);
  const eyesL = center(L.L_EYE_OUTER, L.L_EYE_INNER);
  const angle = Math.atan2(eyesL[1] - eyesR[1], eyesL[0] - eyesR[0]);
  const c = Math.cos(-angle);
  const s = Math.sin(-angle);
  return px.map(([x, y]): Vec => {
    const dx = x - eyesR[0];
    const dy = y - eyesR[1];
    return [c * dx - s * dy, s * dx + c * dy];
  });
}

export function computeMetrics(landmarks: Landmark[], width: number, height: number): Metrics {
  const p = alignedPoints(landmarks, width, height);
  const dist = (a: number, b: number) => Math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]);

  const faceWidth = dist(L.R_ZYGION, L.L_ZYGION);
  const faceHeight = dist(L.FOREHEAD, L.CHIN);

  // Mirror each right-side point across the midline; distance to its partner.
  const midX = median(MIDLINE.map((i) => p[i][0]));
  let residual = 0;
  for (const [r, l] of SYMMETRY_PAIRS) {
    residual += Math.abs(p[r][0] - midX + (p[l][0] - midX)) / faceWidth;
    residual += Math.abs(p[r][1] - p[l][1]) / faceHeight;
  }
  const asymmetry = residual / (2 * SYMMETRY_PAIRS.length);

  // Degrees the outer eye corner sits above the inner one.
  const tiltR = Math.atan2(p[L.R_EYE_INNER][1] - p[L.R_EYE_OUTER][1], p[L.R_EYE_INNER][0] - p[L.R_EYE_OUTER][0]) * DEG;
  const tiltL = Math.atan2(p[L.L_EYE_INNER][1] - p[L.L_EYE_OUTER][1], p[L.L_EYE_OUTER][0] - p[L.L_EYE_INNER][0]) * DEG;

  const eyeWR = dist(L.R_EYE_OUTER, L.R_EYE_INNER);
  const eyeWL = dist(L.L_EYE_OUTER, L.L_EYE_INNER);
  const eyeAspect = (dist(L.R_EYE_TOP, L.R_EYE_BOTTOM) / eyeWR + dist(L.L_EYE_TOP, L.L_EYE_BOTTOM) / eyeWL) / 2;
  const eyeSpacing = dist(L.R_EYE_INNER, L.L_EYE_INNER) / ((eyeWR + eyeWL) / 2);

  const eyeCenterR: Vec = [(p[L.R_EYE_OUTER][0] + p[L.R_EYE_INNER][0]) / 2, (p[L.R_EYE_OUTER][1] + p[L.R_EYE_INNER][1]) / 2];
  const eyeCenterL: Vec = [(p[L.L_EYE_OUTER][0] + p[L.L_EYE_INNER][0]) / 2, (p[L.L_EYE_OUTER][1] + p[L.L_EYE_INNER][1]) / 2];
  const eyeLineY = (eyeCenterR[1] + eyeCenterL[1]) / 2;
  const interpupil = Math.hypot(eyeCenterL[0] - eyeCenterR[0], eyeCenterL[1] - eyeCenterR[1]);
  const midfaceRatio = interpupil / Math.abs(p[L.UPPER_LIP][1] - eyeLineY);

  const browY = (p[L.R_BROW][1] + p[L.L_BROW][1]) / 2;

  return {
    symmetry: asymmetry,
    canthalTilt: (tiltR + tiltL) / 2,
    eyeAspect,
    eyeSpacing,
    midfaceRatio,
    jawRatio: dist(L.R_GONION, L.L_GONION) / faceWidth,
    fwhr: faceWidth / Math.abs(p[L.UPPER_LIP][1] - browY),
    lowerThird: Math.abs(p[L.CHIN][1] - p[L.SUBNASALE][1]) / Math.abs(p[L.SUBNASALE][1] - p[L.GLABELLA][1]),
  };
}

// ---------------------------------------------------------------------------
// Scoring with the learned curves (see training/02_geometry.py)

type Curve = { key: string; x: number[]; y: number[] };
const geometry = params.geometry as {
  curves: Curve[];
  weights: number[];
  display: { rawMean: number; rawStd: number; mean: number; std: number };
  accuracy: { pc5fold: number; pc6040: number };
};
export const GEOMETRY_ACCURACY = geometry.accuracy;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Same as numpy.interp: linear, clamped to the end values. */
function interp(v: number, xs: number[], ys: number[]): number {
  if (v <= xs[0]) return ys[0];
  if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (xs[i] < v) i++;
  const t = (v - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

function curveScore(v: number, curve: Curve): number {
  const lo = Math.min(...curve.y);
  const hi = Math.max(...curve.y);
  return ((interp(v, curve.x, curve.y) - lo) / Math.max(hi - lo, 1e-9)) * 10;
}

export function subScores(metrics: Metrics): Metrics {
  return Object.fromEntries(geometry.curves.map((c) => [c.key, curveScore(metrics[c.key as MetricKey], c)])) as Metrics;
}

/** Overall geometry score on the 0-10 display scale. */
export function geometryScore(subs: Metrics): number {
  const raw = geometry.curves.reduce((sum, c, i) => sum + subs[c.key as MetricKey] * geometry.weights[i], 0);
  const d = geometry.display;
  return clamp(d.mean + (d.std * (raw - d.rawMean)) / d.rawStd, 0, 10);
}

export function traitWeight(key: MetricKey): number {
  return geometry.weights[geometry.curves.findIndex((c) => c.key === key)];
}

// ---------------------------------------------------------------------------
// Labels for the breakdown, plus strongest / weakest trait

export const TRAITS: Record<MetricKey, { label: string; strength: string; low: string; high: string }> = {
  symmetry: { label: "Symmetry", strength: "Symmetric features", low: "Very symmetric", high: "Asymmetry" },
  canthalTilt: { label: "Canthal tilt", strength: "Eye tilt", low: "Negative canthal tilt", high: "Steep canthal tilt" },
  eyeAspect: { label: "Eye shape", strength: "Eye shape", low: "Narrow eye opening", high: "Round eyes" },
  eyeSpacing: { label: "Eye spacing", strength: "Eye spacing", low: "Close-set eyes", high: "Wide-set eyes" },
  midfaceRatio: { label: "Midface", strength: "Midface proportions", low: "Long midface", high: "Short midface" },
  jawRatio: { label: "Jaw width", strength: "Jaw proportions", low: "Narrow jaw", high: "Wide jaw" },
  fwhr: { label: "Width / height", strength: "Face width-to-height", low: "Long, narrow face", high: "Short, wide face" },
  lowerThird: { label: "Lower third", strength: "Facial thirds", low: "Short lower third", high: "Long lower third" },
};

export function strongestAndWeakest(metrics: Metrics, subs: Metrics): { strength: string; flaw: string } {
  const keys = METRIC_KEYS.filter((k) => traitWeight(k) > 0 || k === "symmetry");
  const best = keys.reduce((a, b) => (subs[b] > subs[a] ? b : a));
  const worst = keys.reduce((a, b) => (subs[b] < subs[a] ? b : a));
  // A low score means the value sits on one side of the curve's peak.
  const curve = geometry.curves.find((c) => c.key === worst)!;
  const peakX = curve.x[curve.y.indexOf(Math.max(...curve.y))];
  return {
    strength: TRAITS[best].strength,
    flaw: metrics[worst] < peakX ? TRAITS[worst].low : TRAITS[worst].high,
  };
}
