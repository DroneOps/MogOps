// Parity test: the browser must measure and score faces exactly like the
// Python training code. The fixture is written by training/02_geometry.py
// from synthetic faces (mean SCUT face + noise), not real people.
import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/geometry.json";
import { computeMetrics, geometryScore, METRIC_KEYS, subScores } from "./geometry";
import { poseValues } from "./pose";

type Case = {
  width: number;
  height: number;
  landmarks: number[][];
  metrics: Record<string, number>;
  pose: { roll: number; yaw: number; pitch: number };
  subScores: Record<string, number>;
  overall: number;
};

describe("geometry and pose parity with training/common.py", () => {
  (fixture as Case[]).forEach((c, i) => {
    it(`case ${i} (${c.width}x${c.height})`, () => {
      const landmarks = c.landmarks.map(([x, y]) => ({ x, y }));
      const metrics = computeMetrics(landmarks, c.width, c.height);
      const subs = subScores(metrics);
      for (const key of METRIC_KEYS) {
        expect(metrics[key]).toBeCloseTo(c.metrics[key], 6);
        expect(subs[key]).toBeCloseTo(c.subScores[key], 6);
      }
      expect(geometryScore(subs)).toBeCloseTo(c.overall, 6);
      const pose = poseValues(landmarks, c.width, c.height);
      expect(pose.roll).toBeCloseTo(c.pose.roll, 6);
      expect(pose.yaw).toBeCloseTo(c.pose.yaw, 6);
      expect(pose.pitch).toBeCloseTo(c.pose.pitch, 6);
    });
  });
});
