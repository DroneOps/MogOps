// "PSL ratios": the proportions looksmaxxing forums score faces by, each with
// the community's claimed ideal range (forum/clinic lore, not science).
// Scored the way those tools do: 10 inside the ideal range, falling linearly
// to 0 a couple of standard deviations outside it, then averaged.
// On SCUT-FBP5500 this has no real link to how people rate faces (see
// training/07_psl.py), so the app presents it as the meme it is.
//
// pslMetrics() is a line-by-line twin of training/common.py psl_metrics();
// psl.test.ts checks both give the same numbers.

import { displayScore, type Knots } from "./display";
import { alignedPoints, L, type Landmark } from "./geometry";
import params from "./model-params.json";

export const PSL_KEYS = [
  "canthalTilt",
  "fwhr",
  "midfaceRatio",
  "esr",
  "chinPhiltrum",
  "jawCheek",
  "noseEye",
  "mouthNose",
  "thirds",
  "lipRatio",
] as const;
export type PslKey = (typeof PSL_KEYS)[number];
export type PslMetrics = Record<PslKey, number>;

const R_IRIS = 468;
const L_IRIS = 473;
const R_ALA = 64;
const L_ALA = 294;
const R_MOUTH = 61;
const L_MOUTH = 291;
const UPPER_LIP_INNER = 13;
const LOWER_LIP_INNER = 14;
const DEG = 180 / Math.PI;

export function pslMetrics(landmarks: Landmark[], width: number, height: number): PslMetrics {
  const p = alignedPoints(landmarks, width, height);
  const dist = (a: number, b: number) => Math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]);

  const faceWidth = dist(L.R_ZYGION, L.L_ZYGION);
  const pupils = dist(R_IRIS, L_IRIS);
  const pupilY = (p[R_IRIS][1] + p[L_IRIS][1]) / 2;
  const stomionY = (p[UPPER_LIP_INNER][1] + p[LOWER_LIP_INNER][1]) / 2;
  const noseWidth = dist(R_ALA, L_ALA);
  const browY = (p[L.R_BROW][1] + p[L.L_BROW][1]) / 2;
  const tiltR = Math.atan2(p[L.R_EYE_INNER][1] - p[L.R_EYE_OUTER][1], p[L.R_EYE_INNER][0] - p[L.R_EYE_OUTER][0]) * DEG;
  const tiltL = Math.atan2(p[L.L_EYE_INNER][1] - p[L.L_EYE_OUTER][1], p[L.L_EYE_OUTER][0] - p[L.L_EYE_INNER][0]) * DEG;

  return {
    canthalTilt: (tiltR + tiltL) / 2,
    fwhr: faceWidth / Math.abs(p[L.UPPER_LIP][1] - browY),
    midfaceRatio: pupils / Math.abs(p[L.UPPER_LIP][1] - pupilY),
    esr: pupils / faceWidth,
    chinPhiltrum: Math.abs(p[L.CHIN][1] - stomionY) / Math.abs(stomionY - p[L.SUBNASALE][1]),
    jawCheek: dist(L.R_GONION, L.L_GONION) / faceWidth,
    noseEye: noseWidth / dist(L.R_EYE_INNER, L.L_EYE_INNER),
    mouthNose: dist(R_MOUTH, L_MOUTH) / noseWidth,
    thirds: Math.abs(p[L.CHIN][1] - p[L.SUBNASALE][1]) / Math.abs(p[L.SUBNASALE][1] - p[L.GLABELLA][1]),
    lipRatio: Math.abs(p[L.LOWER_LIP][1] - p[LOWER_LIP_INNER][1]) / Math.abs(p[UPPER_LIP_INNER][1] - p[L.UPPER_LIP][1]),
  };
}

type Ideal = { key: PslKey; lo: number; hi: number; tol: number };
const PSL = params.psl as { ideals: Ideal[]; display: Knots; accuracy: { pc5fold: number; pc6040: number } };
export const PSL_ACCURACY = PSL.accuracy;
export const PSL_IDEALS = PSL.ideals;

export function pslSubScores(m: PslMetrics): PslMetrics {
  return Object.fromEntries(
    PSL.ideals.map(({ key, lo, hi, tol }) => {
      const outside = Math.max(lo - m[key], 0) + Math.max(m[key] - hi, 0);
      return [key, Math.min(1, Math.max(0, 1 - outside / tol)) * 10];
    }),
  ) as PslMetrics;
}

/** Mean sub-score, on the shared 0-10 display scale. */
export function pslScore(subs: PslMetrics): number {
  const harmony = PSL_KEYS.reduce((s, k) => s + subs[k], 0) / PSL_KEYS.length;
  return displayScore(harmony, PSL.display);
}

export const PSL_LABELS: Record<PslKey, { label: string; unit?: string }> = {
  canthalTilt: { label: "Canthal tilt", unit: "°" },
  fwhr: { label: "FWHR" },
  midfaceRatio: { label: "Midface ratio" },
  esr: { label: "Eye separation (ESR)" },
  chinPhiltrum: { label: "Chin / philtrum" },
  jawCheek: { label: "Jaw / cheekbones" },
  noseEye: { label: "Nose / eye gap" },
  mouthNose: { label: "Mouth / nose" },
  thirds: { label: "Lower / middle third" },
  lipRatio: { label: "Lower / upper lip" },
};
