import { percentileOf, pslFromScore, pslTier } from "../lib/display";
import { countsInOverall, GEOMETRY_ACCURACY, METRIC_KEYS, TRAITS, traitWeight } from "../lib/geometry";
import { baseScore, type Engine, ENGINE_LABEL, headlineScore } from "../lib/headline";
import { MOGNET_ACCURACY } from "../lib/mognet";
import { PSL_ACCURACY, PSL_IDEALS, PSL_LABELS } from "../lib/psl";
import type { Analysis } from "../lib/useFaceAnalysis";

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
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;

type Props = { analysis: Analysis; engine: Engine; memeMode: boolean };

export function ScoreCard({ analysis, engine, memeMode }: Props) {
  const headline = headlineScore(analysis, engine, memeMode);
  const headlinePending = engine === "mognet" && analysis.mognetPending && analysis.mognet === null;
  const bonus = memeMode && baseScore(analysis, engine) !== null ? analysis.bonus : null;

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
        {headline !== null && !headlinePending && <PslLine score={headline} />}
        <span className={`rounded-full px-3 py-1 text-xs font-medium ${STATUS_STYLE[analysis.status]}`}>
          {analysis.message}
        </span>
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-sm text-zinc-400">
          {(["mognet", "geometry", "psl"] as const).map((e) => (
            <span key={e} className={engine === e ? "text-zinc-100" : ""}>
              {ENGINE_LABEL[e]}{" "}
              <span className="font-mono tabular-nums">
                {e === "mognet" && analysis.mognetPending && analysis.mognet === null ? "…" : fmt(analysis[e])}
              </span>
            </span>
          ))}
          {bonus && (
            <span className="text-violet-300">
              Bonus <span className="font-mono tabular-nums">{signed(bonus.total)}</span>
            </span>
          )}
        </div>
      </div>

      {bonus && <Bonuses items={bonus.items} />}

      {engine === "psl" ? (
        <PslBreakdown analysis={analysis} />
      ) : (
        <GeometryBreakdown analysis={analysis} />
      )}

      <footer className="border-t border-zinc-800 pt-4 text-xs leading-relaxed text-zinc-500">
        Agreement with human ratings on held-out faces (Pearson r): MogNet {MOGNET_ACCURACY.pc5fold.toFixed(2)} on
        SCUT-FBP5500 and {MOGNET_ACCURACY.pcMEBeautyTest.toFixed(2)} on MEBeauty, geometry{" "}
        {GEOMETRY_ACCURACY.pc5fold.toFixed(2)}, PSL ratios {PSL_ACCURACY.pc5fold.toFixed(2)} (forum lore: no real link).
        Scores are spread so that half of faces reach 6 and a quarter reach 8.
        {analysis.frames > 1 && ` Averaging ${analysis.frames} frames.`}
      </footer>
    </section>
  );
}

function PslLine({ score }: { score: number }) {
  const psl = pslFromScore(score);
  const top = Math.max(0.1, (1 - percentileOf(score)) * 100);
  return (
    <div className="flex flex-col items-center gap-0.5 text-sm">
      <span className="text-zinc-200">
        PSL <span className="font-mono tabular-nums">{psl.toFixed(1)}</span> · {pslTier(psl)}
      </span>
      <span className="text-xs text-zinc-500">Top {top < 1 ? top.toFixed(1) : Math.round(top)}% of faces</span>
    </div>
  );
}

function Bonuses({ items }: { items: NonNullable<Analysis["bonus"]>["items"] }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-violet-500/20 bg-violet-500/5 p-3">
      <div className="text-xs text-violet-300/80">Meme mode: pull these faces for bonus points</div>
      <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {items.map((b) => {
          const penalty = b.max < 0;
          const active = b.level > 0;
          return (
            <li key={b.id} className="flex flex-col gap-1" title={b.hint}>
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className={active ? (penalty ? "text-rose-200" : "text-violet-100") : "text-zinc-400"}>
                  {b.label}
                  <span className="ml-1.5 text-zinc-500">{b.hint}</span>
                </span>
                <span className={`font-mono tabular-nums ${active ? (penalty ? "text-rose-300" : "text-violet-200") : "text-zinc-600"}`}>
                  {signed(active ? b.points : b.max)}
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className={`h-full rounded-full transition-[width] duration-150 ${penalty ? "bg-rose-400" : "bg-violet-400"}`}
                  style={{ width: `${b.level * 100}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Bar({ value, label, note, right }: { value: number | null; label: string; note: string; right: string }) {
  return (
    <li className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-zinc-200">
          {label}
          <span className="ml-2 text-xs text-zinc-500">{note}</span>
        </span>
        <span className="font-mono tabular-nums text-zinc-300">{right}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-zinc-200 transition-[width] duration-300"
          style={{ width: `${value === null ? 0 : value * 10}%` }}
        />
      </div>
    </li>
  );
}

function GeometryBreakdown({ analysis }: { analysis: Analysis }) {
  return (
    <>
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
          const value = analysis.subs?.[key] ?? null;
          return (
            <Bar
              key={key}
              value={value}
              label={TRAITS[key].label}
              note={countsInOverall(key) ? `${Math.round(traitWeight(key) * 100)}% of geometry` : "not in overall"}
              right={value === null ? "0.0" : value.toFixed(1)}
            />
          );
        })}
      </ul>
    </>
  );
}

function PslBreakdown({ analysis }: { analysis: Analysis }) {
  return (
    <ul className="flex flex-col gap-3">
      {PSL_IDEALS.map(({ key, lo, hi }) => {
        const { label, unit = "" } = PSL_LABELS[key];
        const measured = analysis.pslRatios?.[key];
        const digits = unit ? 1 : 2;
        return (
          <Bar
            key={key}
            value={analysis.pslSubs?.[key] ?? null}
            label={label}
            note={`ideal ${lo.toFixed(digits)}–${hi.toFixed(digits)}${unit}`}
            right={measured === undefined ? "–" : `${measured.toFixed(digits)}${unit}`}
          />
        );
      })}
    </ul>
  );
}
