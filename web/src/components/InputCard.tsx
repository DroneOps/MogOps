import { type RefObject, useEffect, useRef, useState } from "react";
import type { Analysis, Overlay } from "../lib/useFaceAnalysis";
import type { Engine } from "./ScoreCard";

export type InputMode = "image" | "camera";

type Props = {
  mode: InputMode;
  onModeChange: (mode: InputMode) => void;
  engine: Engine;
  onEngineChange: (engine: Engine) => void;
  analysis: Analysis;
  overlay: RefObject<Overlay>;
  modelsReady: boolean;
  modelError: string | null;
  onImage: (image: HTMLImageElement) => void;
  onCamera: (video: HTMLVideoElement, deviceId?: string) => Promise<void>;
  onStop: () => void;
  showPoints: boolean;
  onShowPointsChange: (show: boolean) => void;
  soundOn: boolean;
  onSoundChange: (on: boolean) => void;
  /** Briefly true after a raised eyebrow is detected. */
  eyebrowFlash: boolean;
};

export function InputCard(props: Props) {
  const { mode, engine, analysis, overlay, modelsReady, modelError, showPoints } = props;
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [cameraError, setCameraError] = useState<string | null>(null);
  // The draw loop reads this ref so toggling does not restart the loop.
  const showPointsRef = useRef(showPoints);
  showPointsRef.current = showPoints;

  // Start / stop the camera when the mode or device changes.
  useEffect(() => {
    if (mode !== "camera" || !modelsReady || !videoRef.current) return;
    setCameraError(null);
    props
      .onCamera(videoRef.current, deviceId || undefined)
      .then(() => navigator.mediaDevices.enumerateDevices())
      .then((all) => setDevices(all.filter((d) => d.kind === "videoinput")))
      .catch((e: unknown) => setCameraError(e instanceof Error ? e.message : String(e)));
    return () => props.onStop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, deviceId, modelsReady]);

  // Draw landmarks and the model's crop box on top of the preview.
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const canvas = canvasRef.current;
      const { landmarks, box, width, height } = overlay.current;
      if (canvas && width && height) {
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, width, height);
        // Tiny, faint dots so the face stays clearly visible underneath.
        if (showPointsRef.current && landmarks) {
          const r = Math.max(1, width / 900);
          ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
          for (const p of landmarks) ctx.fillRect(p.x * width - r / 2, p.y * height - r / 2, r, r);
        }
        if (showPointsRef.current && box) {
          ctx.strokeStyle = "rgba(52, 211, 153, 0.35)";
          ctx.lineWidth = Math.max(1, width / 900);
          ctx.strokeRect(box.cx - box.side / 2, box.cy - box.side / 2, box.side, box.side);
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [overlay]);

  const onFile = (file: File | undefined) => {
    if (!file) return;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(URL.createObjectURL(file));
  };

  const mirrored = mode === "camera" ? "-scale-x-100" : "";

  return (
    <section className="card flex flex-col gap-5">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="card-title">Face input</h2>
          <p className="card-subtitle">Use a live camera or select one photo to analyze.</p>
        </div>
        <span className="rounded-md border border-zinc-700 px-2 py-0.5 text-xs text-zinc-400">
          {engine === "mognet" ? "MogNet" : "Geometry"}
        </span>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <label className="field">
          <span>Input mode</span>
          <div className="flex rounded-lg bg-zinc-800 p-1">
            {(["image", "camera"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => props.onModeChange(m)}
                className={`flex-1 rounded-md px-3 py-1.5 text-sm capitalize transition ${
                  mode === m ? "bg-zinc-100 text-zinc-900" : "text-zinc-300 hover:text-zinc-100"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </label>
        <label className="field">
          <span>Headline score</span>
          <select value={engine} onChange={(e) => props.onEngineChange(e.target.value as Engine)} className="select">
            <option value="mognet">MogNet (learned)</option>
            <option value="geometry">Geometry (ratios)</option>
          </select>
        </label>
      </div>

      <div className="flex flex-wrap gap-2">
        <Toggle on={showPoints} onChange={props.onShowPointsChange} label="Face points" />
        <Toggle
          on={props.soundOn}
          onChange={props.onSoundChange}
          label="Sound"
          hint="Giga Chad at 8.0+, boom on eyebrow raise"
        />
      </div>

      {mode === "camera" ? (
        <label className="field">
          <span>Camera</span>
          <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)} className="select">
            <option value="">Default camera</option>
            {devices.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </label>
      ) : (
        <label className="field">
          <span>Photo</span>
          <input
            type="file"
            accept="image/*"
            disabled={!modelsReady}
            onChange={(e) => onFile(e.target.files?.[0])}
            className="text-sm text-zinc-300 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-1.5 file:text-sm file:text-zinc-900 disabled:opacity-50"
          />
        </label>
      )}

      <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950">
        {mode === "camera" ? (
          <video ref={videoRef} playsInline muted className={`h-full w-full object-contain ${mirrored}`} />
        ) : (
          imageUrl && (
            <img
              ref={imageRef}
              src={imageUrl}
              alt="Selected face"
              onLoad={(e) => props.onImage(e.currentTarget)}
              className="h-full w-full object-contain"
            />
          )
        )}
        <canvas ref={canvasRef} className={`pointer-events-none absolute inset-0 h-full w-full object-contain ${mirrored}`} />
        {!modelsReady && !modelError && <Placeholder text="Loading models…" />}
        {modelError && <Placeholder text={`Could not load models: ${modelError}`} />}
        {modelsReady && mode === "image" && !imageUrl && <Placeholder text="Choose a photo to begin analysis." />}
        {cameraError && <Placeholder text={`Camera unavailable: ${cameraError}`} />}
        {props.eyebrowFlash && (
          <span className="absolute left-3 top-3 rounded-md bg-zinc-950/70 px-2.5 py-1 text-sm font-medium text-zinc-50">
            Eyebrow raise
          </span>
        )}
      </div>

      {analysis.reasons.length > 0 && (
        <ul className="flex flex-wrap gap-2">
          {analysis.reasons.map((r) => (
            <li key={r} className="rounded-md bg-amber-500/10 px-2.5 py-1 text-xs text-amber-200">
              {r}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Toggle({ on, onChange, label, hint }: { on: boolean; onChange: (on: boolean) => void; label: string; hint?: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      title={hint}
      onClick={() => onChange(!on)}
      className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition ${
        on ? "border-zinc-500 bg-zinc-800 text-zinc-100" : "border-zinc-800 text-zinc-400 hover:text-zinc-200"
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${on ? "bg-emerald-400" : "bg-zinc-600"}`} />
      {label}
      {hint && <span className="hidden text-xs text-zinc-500 sm:inline">{hint}</span>}
    </button>
  );
}

function Placeholder({ text }: { text: string }) {
  return <p className="absolute inset-x-6 text-center text-sm text-zinc-500">{text}</p>;
}
