// SCUT-style face crop. Twin of training/common.py face_crop_box() and
// training/mognet.py crop_face(): a square around the landmarks, gray where it
// runs past the frame edge, resized to the model's input size.

import type { Landmark } from "./geometry";
import params from "./model-params.json";

export type CropBox = { cx: number; cy: number; side: number };

export function faceCropBox(landmarks: Landmark[], width: number, height: number): CropBox {
  const { scale, offsetX, offsetY } = params.crop;
  const xs = landmarks.map((p) => p.x * width);
  const ys = landmarks.map((p) => p.y * height);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const size = Math.max(x1 - x0, y1 - y0);
  return { cx: (x0 + x1) / 2 + offsetX * size, cy: (y0 + y1) / 2 + offsetY * size, side: size * scale };
}

/** Draw the crop onto a size x size canvas and return NCHW float pixels in [0, 1]. */
export function cropToTensor(source: CanvasImageSource, box: CropBox, size: number): Float32Array {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "rgb(128,128,128)";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingQuality = "high";
  const side = Math.round(box.side);
  const left = Math.round(box.cx - box.side / 2);
  const top = Math.round(box.cy - box.side / 2);
  // Map the source square onto the canvas; parts outside the frame stay gray.
  const k = size / side;
  ctx.drawImage(source, -left * k, -top * k, sourceWidth(source) * k, sourceHeight(source) * k);

  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = data[i * 4] / 255;
    out[plane + i] = data[i * 4 + 1] / 255;
    out[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return out;
}

export function sourceWidth(s: CanvasImageSource): number {
  if (s instanceof HTMLVideoElement) return s.videoWidth;
  if (s instanceof HTMLImageElement) return s.naturalWidth;
  return (s as { width: number }).width;
}

export function sourceHeight(s: CanvasImageSource): number {
  if (s instanceof HTMLVideoElement) return s.videoHeight;
  if (s instanceof HTMLImageElement) return s.naturalHeight;
  return (s as { height: number }).height;
}
