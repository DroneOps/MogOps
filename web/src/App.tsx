import { useEffect, useMemo, useRef, useState } from "react";
import { InputCard, type InputMode } from "./components/InputCard";
import { type Engine, ScoreCard } from "./components/ScoreCard";
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
  const { status, mognet, geometry } = face.analysis;
  const headline = engine === "mognet" ? mognet : geometry;
  const faceScored = status === "ready" || status === "adjust";
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
          eyebrowFlash={eyebrowFlash}
        />
        <ScoreCard analysis={face.analysis} engine={engine} />
      </div>

      <footer className="text-xs leading-relaxed text-zinc-500">
        For entertainment only, not an objective measure of anyone's worth. Scores imitate the average opinion of 60
        raters in the SCUT-FBP5500 dataset, which only contains Asian and Caucasian adults, so they are less reliable
        for everyone else. Photos never leave your device.
      </footer>
    </main>
  );
}
