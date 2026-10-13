// MogNet: the learned 0-10 score. The CLIP ViT-B/16 encoder is an int8 ONNX
// graph (training/05_export_web.py) that takes a 224x224 face crop in [0, 1]
// and returns a normalized 512-d embedding; the linear head trained on
// SCUT-FBP5500 + MEBeauty (training/04_teacher.py) lives in model-params.json.
import type { InferenceSession } from "onnxruntime-web/wasm";
import { displayScore, type Knots } from "./display";
import params from "./model-params.json";

const BASE = import.meta.env.BASE_URL;
const INFO = params.mognet as {
  model: string;
  inputSize: number;
  bytes?: number;
  head: { w: number[]; b: number };
  // Linear fix for the int8 encoder's rounding drift (training/05_export_web.py).
  calibration?: { scale: number; offset: number };
  display: Knots;
  accuracy: { pc5fold: number; pcMEBeautyTest: number };
};
export const MOGNET_INPUT = INFO.inputSize;
export const MOGNET_ACCURACY = INFO.accuracy;

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

export async function predictMogNet(pixels: Float32Array): Promise<number> {
  const [ort, s] = await Promise.all([runtime, loadMogNet()]);
  const input = new ort.Tensor("float32", pixels, [1, 3, MOGNET_INPUT, MOGNET_INPUT]);
  const output = await s.run({ image: input });
  const embedding = output.embedding.data as Float32Array;
  const raw = INFO.head.w.reduce((sum, w, i) => sum + w * embedding[i], INFO.head.b);
  const { scale, offset } = INFO.calibration ?? { scale: 1, offset: 0 };
  return displayScore(scale * raw + offset, INFO.display);
}
