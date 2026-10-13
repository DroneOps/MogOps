import { describe, expect, it } from "vitest";
import { scoreBonuses, withBonus } from "./expressions";

describe("meme-mode bonuses", () => {
  it("gives a resting face nothing", () => {
    // Typical resting values from SCUT photos (median squint is ~0.4).
    const rest = { eyeSquintLeft: 0.4, eyeSquintRight: 0.42, browOuterUpLeft: 0.06, browOuterUpRight: 0.08, mouthPucker: 0.05 };
    expect(scoreBonuses(rest).total).toBe(0);
  });

  it("rewards one raised eyebrow and penalizes a big smile", () => {
    const rock = scoreBonuses({ browOuterUpLeft: 0.85, browOuterUpRight: 0.1 });
    expect(rock.items.find((b) => b.id === "rock")!.points).toBeCloseTo(1, 6);
    expect(scoreBonuses({ mouthSmileLeft: 0.9, mouthSmileRight: 0.9 }).total).toBeCloseTo(-0.75, 6);
  });

  it("does not count squinting with closed eyes", () => {
    const closed = scoreBonuses({ eyeSquintLeft: 0.9, eyeSquintRight: 0.9, eyeBlinkLeft: 0.9, eyeBlinkRight: 0.9 });
    expect(closed.total).toBe(0);
  });

  it("keeps the headline on 0-10", () => {
    expect(withBonus(9.6, 2)).toBe(10);
    expect(withBonus(0.2, -0.75)).toBe(0);
  });
});
