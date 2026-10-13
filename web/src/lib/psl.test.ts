// Parity test for the PSL-ratio scorer against training/07_psl.py, on the
// same synthetic faces as the geometry fixture.
import { describe, expect, it } from "vitest";
import faces from "./__fixtures__/geometry.json";
import fixture from "./__fixtures__/psl.json";
import { pslFromScore, pslTier } from "./display";
import { PSL_KEYS, pslMetrics, pslScore, pslSubScores } from "./psl";

type Case = { metrics: Record<string, number>; subScores: Record<string, number>; overall: number };

describe("PSL ratios parity with training/common.py", () => {
  (fixture as Case[]).forEach((c, i) => {
    it(`case ${i}`, () => {
      const { width, height, landmarks } = (faces as { width: number; height: number; landmarks: number[][] }[])[i];
      const metrics = pslMetrics(
        landmarks.map(([x, y]) => ({ x, y })),
        width,
        height,
      );
      const subs = pslSubScores(metrics);
      for (const key of PSL_KEYS) {
        expect(metrics[key]).toBeCloseTo(c.metrics[key], 6);
        expect(subs[key]).toBeCloseTo(c.subScores[key], 6);
      }
      expect(pslScore(subs)).toBeCloseTo(c.overall, 6);
    });
  });
});

describe("PSL scale", () => {
  it("puts the median face at PSL 4 and rises with the score", () => {
    expect(pslFromScore(6)).toBeCloseTo(4, 6);
    expect(pslFromScore(9.5)).toBeGreaterThan(pslFromScore(8));
    expect(pslTier(pslFromScore(6))).toBe("Mid-tier normie");
    expect(pslTier(pslFromScore(10))).toBe("Gigachad");
    expect(pslFromScore(0)).toBe(1);
  });
});
