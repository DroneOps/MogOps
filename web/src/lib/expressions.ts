// Meme-mode bonuses: deliberately gameable points for pulling looksmaxxing
// faces at the camera. They read MediaPipe's face blendshapes (52 expression
// strengths in 0-1, e.g. "browOuterUpLeft", "mouthPucker"), which the model
// estimates per face, so they need no per-person baseline.
//
// Each bonus ramps linearly from 0 at `from` to its full `points` at `to`,
// so you can feel yourself getting closer. Thresholds were set so ordinary
// photos earn almost nothing: on 600 SCUT + 600 MEBeauty faces the positive
// bonuses average +0.03 and ~5% of faces get more than +0.25 (MediaPipe reads
// a resting face as ~0.4 "squint" and has a long tail of "pucker"). They are
// not yet checked against deliberate faces on a webcam; the UI shows each
// meter live for tuning.

export type Blendshapes = Record<string, number>;

type Bonus = {
  id: string;
  label: string;
  hint: string;
  points: number; // negative = penalty
  from: number;
  to: number;
  /** Expression strength 0-1, or 0 when the face doesn't qualify. */
  strength: (b: Blendshapes) => number;
};

const avg = (b: Blendshapes, a: string, c: string) => ((b[a] ?? 0) + (b[c] ?? 0)) / 2;
const get = (b: Blendshapes, k: string) => b[k] ?? 0;

export const BONUSES: Bonus[] = [
  {
    id: "rock",
    label: "The Rock",
    hint: "Raise one eyebrow",
    points: 1.0,
    from: 0.3,
    to: 0.65,
    strength: (b) => Math.abs(get(b, "browOuterUpLeft") - get(b, "browOuterUpRight")),
  },
  {
    id: "hunter",
    label: "Hunter eyes",
    hint: "Squint a little, like you're sizing someone up",
    points: 0.75,
    from: 0.7,
    to: 0.9,
    // Squinting, not closing your eyes.
    strength: (b) => (avg(b, "eyeBlinkLeft", "eyeBlinkRight") < 0.75 ? avg(b, "eyeSquintLeft", "eyeSquintRight") : 0),
  },
  {
    id: "cheeks",
    label: "Hollow cheeks",
    hint: "Suck your cheeks in",
    points: 0.75,
    from: 0.7,
    to: 0.95,
    // MediaPipe has no "cheeks sucked in"; the fish-face pucker that comes with it is the tell.
    strength: (b) =>
      get(b, "jawOpen") < 0.3 && avg(b, "mouthSmileLeft", "mouthSmileRight") < 0.2 ? get(b, "mouthPucker") : 0,
  },
  {
    id: "mewing",
    label: "Mewing",
    hint: "Jaw forward, lips pressed",
    points: 0.5,
    from: 0.4,
    to: 0.7,
    // MediaPipe's jawForward is weak (under 0.01 on almost every photo), so it counts double.
    strength: (b) => Math.max(2 * get(b, "jawForward"), avg(b, "mouthPressLeft", "mouthPressRight")),
  },
  {
    id: "smile",
    label: "Smiling",
    hint: "Chads don't smile",
    points: -0.75,
    from: 0.35,
    to: 0.8,
    strength: (b) => avg(b, "mouthSmileLeft", "mouthSmileRight"),
  },
  {
    id: "puff",
    label: "Puffed cheeks",
    hint: "Not that kind of cheeks",
    points: -0.5,
    from: 0.3,
    to: 0.7,
    strength: (b) => get(b, "cheekPuff"),
  },
];

export type BonusState = { id: string; label: string; hint: string; points: number; max: number; level: number };

const ramp = (v: number, from: number, to: number) => Math.min(1, Math.max(0, (v - from) / (to - from)));

export function scoreBonuses(b: Blendshapes): { total: number; items: BonusState[] } {
  const items = BONUSES.map((bonus) => {
    const level = ramp(bonus.strength(b), bonus.from, bonus.to);
    return { id: bonus.id, label: bonus.label, hint: bonus.hint, points: level * bonus.points, max: bonus.points, level };
  });
  return { total: items.reduce((s, i) => s + i.points, 0), items };
}

/** Per-blendshape exponential smoothing across camera frames. */
export class BlendshapeSmoother {
  private state: Blendshapes = {};
  constructor(private readonly alpha = 0.35) {}

  push(b: Blendshapes): Blendshapes {
    for (const [k, v] of Object.entries(b)) this.state[k] = k in this.state ? this.state[k] + this.alpha * (v - this.state[k]) : v;
    return { ...this.state };
  }

  reset(): void {
    this.state = {};
  }
}

/** Headline score with bonuses, kept on the 0-10 scale. */
export const withBonus = (score: number, bonus: number) => Math.min(10, Math.max(0, score + bonus));
