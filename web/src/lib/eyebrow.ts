// Raised-eyebrow detector ("The Rock" check), a simple math model.
//
// For each side: how far the brow sits above the eye corners, divided by the
// distance between the eyes (so it does not depend on camera distance).
// Eye corners barely move when you raise a brow, unlike the eyelids.
// Each person's resting brow height becomes a slowly adapting baseline; a
// brow held >= RAISE_RATIO above it for a few frames fires one event. It
// re-arms only once the brow comes back down.

import { alignedPoints, L, type Landmark } from "./geometry";

const R_BROW = [70, 63, 105, 66, 107];
const L_BROW = [300, 293, 334, 296, 336];

export const RAISE_RATIO = 1.15; // brow 15% higher than its resting height
const REARM_RATIO = 1.06; // must drop back to within 6% before firing again
const HOLD_FRAMES = 3; // raised for this many frames in a row
const WARMUP_FRAMES = 15; // frames averaged into the first baseline
const BASELINE_RATE = 0.03; // how fast the baseline follows a resting face
const COOLDOWN_MS = 1200;

/** Brow height above the eye corners, in inter-eye distances: [right, left]. */
export function browHeights(landmarks: Landmark[], width: number, height: number): [number, number] {
  const p = alignedPoints(landmarks, width, height);
  const mean = (ids: number[], axis: 0 | 1) => ids.reduce((s, i) => s + p[i][axis], 0) / ids.length;
  const eyeR = [L.R_EYE_OUTER, L.R_EYE_INNER];
  const eyeL = [L.L_EYE_OUTER, L.L_EYE_INNER];
  const interEye = Math.hypot(mean(eyeL, 0) - mean(eyeR, 0), mean(eyeL, 1) - mean(eyeR, 1));
  // Image y grows downward, so "eye y - brow y" is positive.
  return [(mean(eyeR, 1) - mean(R_BROW, 1)) / interEye, (mean(eyeL, 1) - mean(L_BROW, 1)) / interEye];
}

export class EyebrowDetector {
  private baseline: [number, number] | null = null;
  private warmup: [number, number][] = [];
  private held = 0;
  private armed = true;
  private lastFire = -Infinity;

  /** Feed one frame; returns true on the frame a raise is detected. */
  update(landmarks: Landmark[], width: number, height: number, now: number): boolean {
    const h = browHeights(landmarks, width, height);
    if (!this.baseline) {
      this.warmup.push(h);
      if (this.warmup.length >= WARMUP_FRAMES) {
        const n = this.warmup.length;
        this.baseline = [this.warmup.reduce((s, v) => s + v[0], 0) / n, this.warmup.reduce((s, v) => s + v[1], 0) / n];
      }
      return false;
    }
    const ratio = Math.max(h[0] / this.baseline[0], h[1] / this.baseline[1]);
    if (ratio < RAISE_RATIO) {
      this.held = 0;
      if (ratio < REARM_RATIO) this.armed = true;
      // Only learn from resting faces, so a held raise never becomes "normal".
      this.baseline = [
        this.baseline[0] + BASELINE_RATE * (h[0] - this.baseline[0]),
        this.baseline[1] + BASELINE_RATE * (h[1] - this.baseline[1]),
      ];
      return false;
    }
    this.held++;
    if (this.armed && this.held >= HOLD_FRAMES && now - this.lastFire > COOLDOWN_MS) {
      this.armed = false;
      this.lastFire = now;
      return true;
    }
    return false;
  }

  reset(): void {
    this.baseline = null;
    this.warmup = [];
    this.held = 0;
    this.armed = true;
  }
}
