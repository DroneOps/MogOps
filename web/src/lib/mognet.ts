// MogNet: the learned 0-10 score. CLIP ViT-B/16 + ridge head, exported by
// training/05_export_web.py as one int8 ONNX graph that takes a 224x224 face
// crop in [0, 1] and returns the predicted SCUT rating (1-5).
import type { InferenceSession } from "onnxruntime-web/wasm";
import params from "./model-params.json";

const BASE = import.meta.env.BASE_URL;
const INFO = params.mognet as {
  model: string;
  inputSize: number;
  bytes?: number;
  // Linear fix for the int8 model's rounding drift (training/05_export_web.py).
  calibration?: { scale: number; offset: number };
  accuracy: { pc5fold: number; pc6040: number; mae5fold: number; rmse5fold: number };
};
export const MOGNET_INPUT = INFO.inputSize;
export const MOGNET_ACCURACY = INFO.accuracy;
export const MOGNET_MB = INFO.bytes ? Math.round(INFO.bytes / 1e6) : null;

// ONNX Runtime is loaded at runtime from our own copy in public/vendor/ort
// (scripts/copy-wasm.mjs) instead of being bundled. Its worker threads start
// by re-loading the file that created them; if Vite bundled it, they would
// load the app itself and crash on `document`. A full URL also keeps Vite's
// dev server from rewriting the import.
type Ort = typeof import("onnxruntime-web/wasm");
const ORT_DIR = new URL(`${BASE}vendor/ort/`, location.href).href;
const runtime: Promise<Ort> = import(/* @vite-ignore */ `${ORT_DIR}ort.wasm.bundle.min.mjs`).then((ort: Ort) => {
  ort.env.wasm.wasmPaths = { wasm: `${ORT_DIR}ort-wasm-simd-threaded.wasm` };
  // Threads need cross-origin isolation (see vite.config.ts).
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  return ort;
});

let session: Promise<InferenceSession> | null = null;

export function loadMogNet(): Promise<InferenceSession> {
  session ??= runtime.then((ort) =>
    ort.InferenceSession.create(`${BASE}models/${INFO.model}`, { executionProviders: ["wasm"] }),
  );
  return session;
}

/** SCUT 1-5 rating mapped linearly to 0-10. */
export const toTen = (rating: number) => Math.min(10, Math.max(0, ((rating - 1) / 4) * 10));

export async function predictMogNet(pixels: Float32Array): Promise<number> {
  const [ort, s] = await Promise.all([runtime, loadMogNet()]);
  const input = new ort.Tensor("float32", pixels, [1, 3, MOGNET_INPUT, MOGNET_INPUT]);
  const output = await s.run({ image: input });
  const raw = Number((output.score.data as Float32Array)[0]);
  const { scale, offset } = INFO.calibration ?? { scale: 1, offset: 0 };
  return toTen(scale * raw + offset);
}
