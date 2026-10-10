// Runs the full pipeline for a photo or a live camera:
//   MediaPipe landmarks -> pose check -> geometry breakdown (+ 30-frame window
//   in camera mode) -> SCUT-style crop -> MogNet, throttled and smoothed.
import { useCallback, useEffect, useRef, useState } from "react";
import { cropToTensor, faceCropBox, sourceHeight, sourceWidth, type CropBox } from "./crop";
import { computeMetrics, geometryScore, type Landmark, type Metrics, strongestAndWeakest, subScores } from "./geometry";
import { detectImage, detectVideo, preloadLandmarker } from "./landmarker";
import { loadMogNet, MOGNET_INPUT, predictMogNet } from "./mognet";
import { checkPose } from "./pose";
import { MetricWindow } from "./rolling";

export type Status = "idle" | "loading" | "analyzing" | "ready" | "adjust" | "noface" | "error";

export type Analysis = {
  status: Status;
  message: string;
  reasons: string[];
  geometry: number | null;
  mognet: number | null;
  mognetPending: boolean;
  subs: Metrics | null;
  strength: string | null;
  flaw: string | null;
  frames: number;
};

export type Overlay = { landmarks: Landmark[] | null; box: CropBox | null; width: number; height: number };

const EMPTY: Analysis = {
  status: "idle",
  message: "Waiting for input",
  reasons: [],
  geometry: null,
  mognet: null,
  mognetPending: false,
  subs: null,
  strength: null,
  flaw: null,
  frames: 0,
};

const MOGNET_INTERVAL_MS = 1500;
const UI_INTERVAL_MS = 100;

export function useFaceAnalysis() {
  const [analysis, setAnalysis] = useState<Analysis>(EMPTY);
  const [modelsReady, setModelsReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const overlay = useRef<Overlay>({ landmarks: null, box: null, width: 0, height: 0 });
  const camera = useRef<{ stop: boolean; raf: number; stream: MediaStream | null }>({ stop: true, raf: 0, stream: null });

  useEffect(() => {
    Promise.all([preloadLandmarker(), loadMogNet()])
      .then(() => setModelsReady(true))
      .catch((e: unknown) => setModelError(e instanceof Error ? e.message : String(e)));
  }, []);

  const scoreGeometry = (metrics: Metrics) => {
    const subs = subScores(metrics);
    return { subs, geometry: geometryScore(subs), ...strongestAndWeakest(metrics, subs) };
  };

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    stopCamera();
    setAnalysis({ ...EMPTY, status: "analyzing", message: "Analyzing" });
    try {
      const w = image.naturalWidth;
      const h = image.naturalHeight;
      const landmarks = await detectImage(image);
      if (!landmarks) {
        overlay.current = { landmarks: null, box: null, width: w, height: h };
        setAnalysis({ ...EMPTY, status: "noface", message: "No face found" });
        return;
      }
      const box = faceCropBox(landmarks, w, h);
      overlay.current = { landmarks, box, width: w, height: h };
      const pose = checkPose(landmarks, w, h);
      const geo = scoreGeometry(computeMetrics(landmarks, w, h));
      const base = {
        ...EMPTY,
        ...geo,
        status: (pose.ok ? "ready" : "adjust") as Status,
        message: pose.ok ? "Done" : "Scored, but the pose is off",
        reasons: pose.reasons,
        frames: 1,
      };
      setAnalysis({ ...base, mognetPending: true });
      const mognet = await predictMogNet(cropToTensor(image, box, MOGNET_INPUT));
      setAnalysis({ ...base, mognet });
    } catch (e) {
      setAnalysis({ ...EMPTY, status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  const startCamera = useCallback(async (video: HTMLVideoElement, deviceId?: string) => {
    stopCamera();
    setAnalysis({ ...EMPTY, status: "loading", message: "Starting camera" });
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    } catch (e) {
      setAnalysis({ ...EMPTY, status: "error", message: "Camera unavailable" });
      throw e;
    }
    const state = { stop: false, raf: 0, stream };
    camera.current = state;

    const history = new MetricWindow(30);
    let mognet: number | null = null;
    let busy = false;
    let lastMogNet = 0;
    let lastUi = 0;
    let latest: Analysis = { ...EMPTY, status: "analyzing", message: "Looking for a face" };

    const tick = async () => {
      if (state.stop) return;
      if (video.readyState >= 2) {
        const now = performance.now();
        const w = sourceWidth(video);
        const h = sourceHeight(video);
        const landmarks = await detectVideo(video, now);
        if (state.stop) return;
        if (!landmarks) {
          history.reset();
          overlay.current = { landmarks: null, box: null, width: w, height: h };
          latest = { ...latest, status: "noface", message: "No face found", reasons: [] };
        } else {
          const box = faceCropBox(landmarks, w, h);
          overlay.current = { landmarks, box, width: w, height: h };
          const pose = checkPose(landmarks, w, h);
          if (!pose.ok) {
            // Pause instead of penalizing: keep the last scores, ask for a better pose.
            latest = { ...latest, status: "adjust", message: "Adjust your pose", reasons: pose.reasons };
          } else {
            const smoothed = history.push(computeMetrics(landmarks, w, h));
            latest = {
              ...latest,
              ...scoreGeometry(smoothed),
              status: "ready",
              message: "Live",
              reasons: [],
              frames: history.count,
              mognet,
              mognetPending: mognet === null,
            };
            if (!busy && now - lastMogNet > MOGNET_INTERVAL_MS) {
              busy = true;
              lastMogNet = now;
              const pixels = cropToTensor(video, box, MOGNET_INPUT);
              predictMogNet(pixels)
                .then((v) => {
                  // Exponential smoothing across runs keeps the number steady.
                  mognet = mognet === null ? v : 0.6 * mognet + 0.4 * v;
                  latest = { ...latest, mognet, mognetPending: false };
                })
                .finally(() => {
                  busy = false;
                });
            }
          }
        }
        if (now - lastUi > UI_INTERVAL_MS) {
          lastUi = now;
          setAnalysis(latest);
        }
      }
      state.raf = requestAnimationFrame(tick);
    };
    state.raf = requestAnimationFrame(tick);
  }, []);

  const stopCamera = useCallback(() => {
    const state = camera.current;
    state.stop = true;
    cancelAnimationFrame(state.raf);
    state.stream?.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }, []);

  const reset = useCallback(() => {
    stopCamera();
    overlay.current = { landmarks: null, box: null, width: 0, height: 0 };
    setAnalysis(EMPTY);
  }, [stopCamera]);

  useEffect(() => stopCamera, [stopCamera]);

  return { analysis, overlay, modelsReady, modelError, analyzeImage, startCamera, stopCamera, reset };
}
