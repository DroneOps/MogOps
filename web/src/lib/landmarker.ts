// MediaPipe Face Landmarker, served from this app (see scripts/copy-wasm.mjs).
import { FaceLandmarker, type FaceLandmarkerResult, FilesetResolver } from "@mediapipe/tasks-vision";
import type { Blendshapes } from "./expressions";
import type { Landmark } from "./geometry";

export type Face = { landmarks: Landmark[]; blendshapes: Blendshapes };

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
      outputFaceBlendshapes: true, // expression strengths for the meme-mode bonuses
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

function firstFace(result: FaceLandmarkerResult): Face | null {
  const landmarks = result.faceLandmarks[0];
  if (!landmarks) return null;
  const categories = result.faceBlendshapes[0]?.categories ?? [];
  return { landmarks, blendshapes: Object.fromEntries(categories.map((c) => [c.categoryName, c.score])) };
}

export async function detectImage(image: HTMLImageElement): Promise<Face | null> {
  return firstFace((await withMode("IMAGE")).detect(image));
}

export async function detectVideo(video: HTMLVideoElement, timestamp: number): Promise<Face | null> {
  return firstFace((await withMode("VIDEO")).detectForVideo(video, timestamp));
}

export const preloadLandmarker = () => load().then(() => undefined);
