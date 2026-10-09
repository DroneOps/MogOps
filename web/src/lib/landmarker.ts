// MediaPipe Face Landmarker, served from this app (see scripts/copy-wasm.mjs).
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { Landmark } from "./geometry";

const BASE = import.meta.env.BASE_URL;
let instance: Promise<FaceLandmarker> | null = null;
let mode: "IMAGE" | "VIDEO" = "IMAGE";

function load(): Promise<FaceLandmarker> {
  instance ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks(`${BASE}vendor/mediapipe`);
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
      runningMode: mode,
      numFaces: 1,
    });
    try {
      return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
    } catch {
      return FaceLandmarker.createFromOptions(fileset, options("CPU"));
    }
  })();
  return instance;
}

async function withMode(next: "IMAGE" | "VIDEO"): Promise<FaceLandmarker> {
  const landmarker = await load();
  if (mode !== next) {
    mode = next;
    await landmarker.setOptions({ runningMode: next });
  }
  return landmarker;
}

export async function detectImage(image: HTMLImageElement): Promise<Landmark[] | null> {
  const result = (await withMode("IMAGE")).detect(image);
  return result.faceLandmarks[0] ?? null;
}

export async function detectVideo(video: HTMLVideoElement, timestamp: number): Promise<Landmark[] | null> {
  const result = (await withMode("VIDEO")).detectForVideo(video, timestamp);
  return result.faceLandmarks[0] ?? null;
}

export const preloadLandmarker = () => load().then(() => undefined);
