// Raised-eyebrow detector ("The Rock" check), a simple math model.
//
// For each side: how far the brow sits above the eye corners, divided by the
// distance between the eyes (so it does not depend on camera distance).
// Eye corners barely move when you raise a brow, unlike the eyelids.
// The baseline is a low percentile of the last few seconds of brow heights,
// recalculated every frame, so it follows whoever is in front of the camera
// (people can swap without restarting) while a short raise barely moves it.
// A brow held >= RAISE_RATIO above it for a few frames fires one event. It
// re-arms once the brow comes back down (or the baseline catches up).

import { alignedPoints, L, type Landmark } from "./geometry";

const R_BROW = [70, 63, 105, 66, 107];
const L_BROW = [300, 293, 334, 296, 336];

export const RAISE_RATIO = 1.15; // brow 15% higher than its resting height
const REARM_RATIO = 1.06; // must drop back to within 6% before firing again
const HOLD_FRAMES = 3; // raised for this many frames in a row
const WARMUP_FRAMES = 15; // frames needed before the first baseline
const BASELINE_FRAMES = 90; // ~3 s of history at 30 fps
const BASELINE_PERCENTILE = 0.25; // a raise must fill 75% of the window to become "normal"
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

function percentile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(q * (sorted.length - 1))];
}

export class EyebrowDetector {
  private right: number[] = [];
  private left: number[] = [];
  private held = 0;
  private armed = true;
  private lastFire = -Infinity;

  /** Feed one frame; returns true on the frame a raise is detected. */
  update(landmarks: Landmark[], width: number, height: number, now: number): boolean {
    const [r, l] = browHeights(landmarks, width, height);
    this.right.push(r);
    this.left.push(l);
    if (this.right.length > BASELINE_FRAMES) {
      this.right.shift();
      this.left.shift();
    }
    if (this.right.length < WARMUP_FRAMES) return false;
    const ratio = Math.max(r / percentile(this.right, BASELINE_PERCENTILE), l / percentile(this.left, BASELINE_PERCENTILE));
    if (ratio < RAISE_RATIO) {
      this.held = 0;
      if (ratio < REARM_RATIO) this.armed = true;
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
    this.right = [];
    this.left = [];
    this.held = 0;
    this.armed = true;
  }
}
