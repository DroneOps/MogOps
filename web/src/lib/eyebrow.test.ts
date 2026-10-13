// Uses the synthetic mean face from the geometry parity fixture.
import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/geometry.json";
import { browHeights, EyebrowDetector } from "./eyebrow";
import type { Landmark } from "./geometry";

const { width, height, landmarks } = (fixture as { width: number; height: number; landmarks: number[][] }[])[0];
const face: Landmark[] = landmarks.map(([x, y]) => ({ x, y }));
const RIGHT_BROW = [70, 63, 105, 66, 107];

/** Move the right brow up by `amount` (normalized image units). */
const raised = (amount: number): Landmark[] =>
  face.map((p, i) => (RIGHT_BROW.includes(i) ? { x: p.x, y: p.y - amount } : p));

describe("EyebrowDetector", () => {
  const [restRight] = browHeights(face, width, height);
  // Enough lift for a ~30% higher right brow, well past the 15% threshold.
  // (The fixture face is level, so eye-center distance is purely horizontal.)
  const eyeCenterGap = Math.abs((face[263].x + face[362].x) / 2 - (face[33].x + face[133].x) / 2) * width;
  const lift = (0.3 * restRight * eyeCenterGap) / height;

  it("fires once per raise, ignores a resting face, re-arms after lowering", () => {
    const d = new EyebrowDetector();
    let t = 0;
    const step = (lm: Landmark[]) => d.update(lm, width, height, (t += 33));

    const rest = Array.from({ length: 40 }, () => step(face));
    expect(rest.some(Boolean)).toBe(false);

    const up = Array.from({ length: 30 }, () => step(raised(lift)));
    expect(up.filter(Boolean)).toHaveLength(1);

    Array.from({ length: 10 }, () => step(face));
    const again = Array.from({ length: 30 }, () => step(raised(lift)));
    expect(again.filter(Boolean)).toHaveLength(1);
  });

  it("does not fire on a small twitch", () => {
    const d = new EyebrowDetector();
    let t = 0;
    for (let i = 0; i < 40; i++) d.update(face, width, height, (t += 33));
    const twitch = Array.from({ length: 30 }, () => d.update(raised(lift / 4), width, height, (t += 33)));
    expect(twitch.some(Boolean)).toBe(false);
  });
});
