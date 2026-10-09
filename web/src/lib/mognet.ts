// MogNet: the learned 0-10 score. CLIP ViT-B/16 + ridge head, exported by
// training/05_export_web.py as one int8 ONNX graph that takes a 224x224 face
// crop in [0, 1] and returns the predicted SCUT rating (1-5).
import * as ort from "onnxruntime-web";
import params from "./model-params.json";

const BASE = import.meta.env.BASE_URL;
const INFO = params.mognet as {
  model: string;
  inputSize: number;
  bytes?: number;
  accuracy: { pc5fold: number; pc6040: number; mae5fold: number; rmse5fold: number };
};
export const MOGNET_INPUT = INFO.inputSize;
export const MOGNET_ACCURACY = INFO.accuracy;
export const MOGNET_MB = INFO.bytes ? Math.round(INFO.bytes / 1e6) : null;

ort.env.wasm.wasmPaths = `${BASE}vendor/ort/`;
// Threads need cross-origin isolation (see vite.config.ts).
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session: Promise<ort.InferenceSession> | null = null;

export function loadMogNet(): Promise<ort.InferenceSession> {
  session ??= ort.InferenceSession.create(`${BASE}models/${INFO.model}`, { executionProviders: ["wasm"] });
  return session;
}

/** SCUT 1-5 rating mapped linearly to 0-10. */
export const toTen = (rating: number) => Math.min(10, Math.max(0, ((rating - 1) / 4) * 10));

export async function predictMogNet(pixels: Float32Array): Promise<number> {
  const s = await loadMogNet();
  const input = new ort.Tensor("float32", pixels, [1, 3, MOGNET_INPUT, MOGNET_INPUT]);
  const output = await s.run({ image: input });
  return toTen(Number((output.score.data as Float32Array)[0]));
}
