import { useState } from "react";
import { InputCard, type InputMode } from "./components/InputCard";
import { type Engine, ScoreCard } from "./components/ScoreCard";
import { useFaceAnalysis } from "./lib/useFaceAnalysis";

export default function App() {
  const [mode, setMode] = useState<InputMode>("image");
  const [engine, setEngine] = useState<Engine>("mognet");
  const face = useFaceAnalysis();

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
