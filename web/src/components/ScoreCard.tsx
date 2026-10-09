import { GEOMETRY_ACCURACY, METRIC_KEYS, TRAITS, traitWeight } from "../lib/geometry";
import { MOGNET_ACCURACY } from "../lib/mognet";
import type { Analysis } from "../lib/useFaceAnalysis";

export type Engine = "mognet" | "geometry";

const STATUS_STYLE: Record<Analysis["status"], string> = {
  idle: "bg-zinc-800 text-zinc-300",
  loading: "bg-zinc-800 text-zinc-300",
  analyzing: "bg-sky-500/15 text-sky-300",
  ready: "bg-emerald-500/15 text-emerald-300",
  adjust: "bg-amber-500/15 text-amber-300",
  noface: "bg-amber-500/15 text-amber-300",
  error: "bg-red-500/15 text-red-300",
};

const fmt = (v: number | null) => (v === null ? "–" : v.toFixed(1));

export function ScoreCard({ analysis, engine }: { analysis: Analysis; engine: Engine }) {
  const headline = engine === "mognet" ? analysis.mognet : analysis.geometry;
  const headlinePending = engine === "mognet" && analysis.mognetPending && headline === null;

  return (
    <section className="card flex flex-col gap-5">
      <header>
        <h2 className="card-title">Score</h2>
        <p className="card-subtitle">Overall result from 0 to 10</p>
      </header>

      <div className="flex flex-col items-center gap-3 py-2">
        <div className="font-mono text-7xl font-semibold tabular-nums tracking-tight text-zinc-50">
          {headlinePending ? "…" : headline === null ? "0.0" : headline.toFixed(1)}
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLE[analysis.status]}`}>
          {analysis.message}
        </span>
        <div className="flex gap-4 text-sm text-zinc-400">
          <span className={engine === "mognet" ? "text-zinc-100" : ""}>
            MogNet <span className="font-mono tabular-nums">{analysis.mognetPending && analysis.mognet === null ? "…" : fmt(analysis.mognet)}</span>
          </span>
          <span className={engine === "geometry" ? "text-zinc-100" : ""}>
            Geometry <span className="font-mono tabular-nums">{fmt(analysis.geometry)}</span>
          </span>
        </div>
      </div>

      {analysis.strength && analysis.flaw && (
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2">
            <div className="text-xs text-emerald-300/80">Strongest</div>
            <div className="text-emerald-100">{analysis.strength}</div>
          </div>
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2">
            <div className="text-xs text-rose-300/80">Weakest</div>
            <div className="text-rose-100">{analysis.flaw}</div>
          </div>
        </div>
      )}

      <ul className="flex flex-col gap-3">
        {METRIC_KEYS.map((key) => {
          const value = analysis.subs?.[key] ?? 0;
          const weight = traitWeight(key);
          return (
            <li key={key} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between text-sm">
                <span className="text-zinc-200">
                  {TRAITS[key].label}
                  <span className="ml-2 text-xs text-zinc-500">
                    {weight > 0 ? `${Math.round(weight * 100)}% of geometry` : "not in overall"}
                  </span>
                </span>
                <span className="font-mono tabular-nums text-zinc-300">{analysis.subs ? value.toFixed(1) : "0.0"}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-zinc-200 transition-[width] duration-300"
                  style={{ width: `${analysis.subs ? value * 10 : 0}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      <footer className="border-t border-zinc-800 pt-4 text-xs leading-relaxed text-zinc-500">
        Agreement with human ratings on held-out SCUT-FBP5500 faces (Pearson r): MogNet{" "}
        {MOGNET_ACCURACY.pc5fold.toFixed(2)}, geometry {GEOMETRY_ACCURACY.pc5fold.toFixed(2)}.
        {analysis.frames > 1 && ` Averaging ${analysis.frames} frames.`}
      </footer>
    </section>
  );
}
