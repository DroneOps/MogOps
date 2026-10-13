import { useEffect, useMemo, useRef, useState } from "react";
import { InputCard, type InputMode } from "./components/InputCard";
import { ScoreCard } from "./components/ScoreCard";
import { type Engine, headlineScore } from "./lib/headline";
import { createSounds, MOG_THRESHOLD } from "./lib/sounds";
import { useFaceAnalysis } from "./lib/useFaceAnalysis";

/** A boolean setting remembered in this browser. */
function useStoredFlag(key: string, initial: boolean) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored === null ? initial : stored === "1";
    } catch {
      return initial;
    }
  });
  const set = (next: boolean) => {
    setValue(next);
    try {
      localStorage.setItem(key, next ? "1" : "0");
    } catch {
      // Storage can be blocked (private mode); the setting just won't persist.
    }
  };
  return [value, set] as const;
}

export default function App() {
  const [mode, setMode] = useState<InputMode>("image");
  const [engine, setEngine] = useState<Engine>("mognet");
  const [showPoints, setShowPoints] = useStoredFlag("mogops.showPoints", false);
  const [soundOn, setSoundOn] = useStoredFlag("mogops.sound", true);
  const [memeMode, setMemeMode] = useStoredFlag("mogops.meme", true);
  const [eyebrowFlash, setEyebrowFlash] = useState(false);
  const sounds = useMemo(createSounds, []);
  const soundOnRef = useRef(soundOn);
  soundOnRef.current = soundOn;
  const flashTimer = useRef(0);

  const face = useFaceAnalysis({
    onEyebrowRaise: () => {
      if (soundOnRef.current) sounds.boom();
      setEyebrowFlash(true);
      clearTimeout(flashTimer.current);
      flashTimer.current = window.setTimeout(() => setEyebrowFlash(false), 1200);
    },
  });

  // Giga Chad theme while the headline score is at or above the threshold.
  // Pausing keeps its position, so it resumes instead of restarting.
  const headline = headlineScore(face.analysis, engine, memeMode);
  const faceScored = face.analysis.status === "ready" || face.analysis.status === "adjust";
  useEffect(() => {
    sounds.setMusic(soundOn && faceScored && headline !== null && headline >= MOG_THRESHOLD);
  }, [sounds, soundOn, faceScored, headline]);
  useEffect(() => () => sounds.stop(), [sounds]);

  const changeMode = (next: InputMode) => {
    face.reset();
    setMode(next);
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6">
      <header>
        <h1 className="text-xl font-semibold text-zinc-50">MogOps</h1>
        <p className="text-sm text-zinc-400">Face rating, fully on-device</p>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.4fr_1fr]">
        <InputCard
          mode={mode}
          onModeChange={changeMode}
          engine={engine}
          onEngineChange={setEngine}
          analysis={face.analysis}
          overlay={face.overlay}
          modelsReady={face.modelsReady}
          modelError={face.modelError}
          onImage={face.analyzeImage}
          onCamera={face.startCamera}
          onStop={face.stopCamera}
          showPoints={showPoints}
          onShowPointsChange={setShowPoints}
          soundOn={soundOn}
          onSoundChange={setSoundOn}
          memeMode={memeMode}
          onMemeModeChange={setMemeMode}
          eyebrowFlash={eyebrowFlash}
        />
        <ScoreCard analysis={face.analysis} engine={engine} memeMode={memeMode} />
      </div>

      <footer className="text-xs leading-relaxed text-zinc-500">
        For entertainment only, not an objective measure of anyone's worth. MogNet imitates the average opinion of the
        raters behind SCUT-FBP5500 and MEBeauty (adult faces from six ethnic groups), so it is still less reliable for
        faces unlike theirs. Meme mode adds points for pulling faces on purpose. Photos never leave your device.
      </footer>
    </main>
  );
}
