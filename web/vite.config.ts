import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Cross-origin isolation lets ONNX Runtime use several CPU threads
// (SharedArrayBuffer). Production hosts must send the same two headers;
// without them the model still runs, just single-threaded.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  server: { headers: isolation },
  preview: { headers: isolation },
});
