// Which number is the big one: the chosen engine's score, plus the meme-mode
// expression bonuses when they are on.
import { withBonus } from "./expressions";
import type { Analysis } from "./useFaceAnalysis";

export type Engine = "mognet" | "geometry" | "psl";

export const ENGINE_LABEL: Record<Engine, string> = { mognet: "MogNet", geometry: "Geometry", psl: "PSL ratios" };

export const baseScore = (a: Analysis, engine: Engine) => a[engine];

export function headlineScore(a: Analysis, engine: Engine, meme: boolean): number | null {
  const base = baseScore(a, engine);
  if (base === null) return null;
  return meme && a.bonus ? withBonus(base, a.bonus.total) : base;
}
