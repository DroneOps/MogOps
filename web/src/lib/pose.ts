// "Face the camera" check. Twin of training/common.py pose_values(); the
// allowed ranges are SCUT's 1st-99th percentile (training/01_landmarks.py).
//
// Unlike Mogwars, a bad pose never scales the score down. We just pause
// scoring and tell the user what to fix, so the number only reflects the face.

import { alignedPoints, L, type Landmark } from "./geometry";
import params from "./model-params.json";

export type Pose = { roll: number; yaw: number; pitch: number };
export type PoseCheck = { ok: boolean; reasons: string[] };

const RANGES = params.pose as Record<keyof Pose, [number, number]>;

export function poseValues(landmarks: Landmark[], width: number, height: number): Pose {
  const px = landmarks.map((p) => [p.x * width, p.y * height]);
  const eyesR = [(px[L.R_EYE_OUTER][0] + px[L.R_EYE_INNER][0]) / 2, (px[L.R_EYE_OUTER][1] + px[L.R_EYE_INNER][1]) / 2];
  const eyesL = [(px[L.L_EYE_OUTER][0] + px[L.L_EYE_INNER][0]) / 2, (px[L.L_EYE_OUTER][1] + px[L.L_EYE_INNER][1]) / 2];
  const roll = (Math.atan2(eyesL[1] - eyesR[1], eyesL[0] - eyesR[0]) * 180) / Math.PI;

  const p = alignedPoints(landmarks, width, height);
  const faceWidth = Math.hypot(p[L.R_ZYGION][0] - p[L.L_ZYGION][0], p[L.R_ZYGION][1] - p[L.L_ZYGION][1]);
  const yaw = (p[L.NOSE_TIP][0] - (p[L.R_ZYGION][0] + p[L.L_ZYGION][0]) / 2) / faceWidth;
  const eyeY = (p[L.R_EYE_OUTER][1] + p[L.R_EYE_INNER][1] + p[L.L_EYE_OUTER][1] + p[L.L_EYE_INNER][1]) / 4;
  const pitch = (p[L.NOSE_TIP][1] - eyeY) / (p[L.CHIN][1] - eyeY);
  return { roll, yaw, pitch };
}

const inRange = (v: number, [lo, hi]: [number, number]) => v >= lo && v <= hi;

export function checkPose(landmarks: Landmark[], width: number, height: number): PoseCheck {
  const pose = poseValues(landmarks, width, height);
  const reasons: string[] = [];
  if (!inRange(pose.yaw, RANGES.yaw) || !inRange(pose.pitch, RANGES.pitch)) reasons.push("Face the camera straight on");
  if (!inRange(pose.roll, RANGES.roll)) reasons.push("Keep your head level");

  // The whole face must be visible and big enough to measure.
  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  if (x0 < 0.01 || y0 < 0.01 || x1 > 0.99 || y1 > 0.99) reasons.push("Keep your whole face in frame");
  else if (Math.max((x1 - x0) * width, (y1 - y0) * height) < 0.25 * Math.min(width, height)) reasons.push("Move closer");

  return { ok: reasons.length === 0, reasons };
}
