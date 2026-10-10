// Copies the MediaPipe and ONNX Runtime WebAssembly files into public/ so the
// app serves them itself (no CDN). Runs automatically after `npm install`.
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const copies = [
  ["node_modules/@mediapipe/tasks-vision/wasm", "public/vendor/mediapipe", () => true],
  ["node_modules/onnxruntime-web/dist", "public/vendor/ort", (f) => f === "ort-wasm-simd-threaded.wasm" || f === "ort.wasm.bundle.min.mjs"],
];
for (const [from, to, keep] of copies) {
  mkdirSync(join(web, to), { recursive: true });
  for (const file of readdirSync(join(web, from)).filter(keep)) {
    cpSync(join(web, from, file), join(web, to, file));
  }
  console.log(`copied ${from} -> ${to}`);
}
