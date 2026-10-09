// Camera mode smooths the geometry ratios over the last N good frames with a
// trimmed mean (drop the highest and lowest 20%), so one bad frame from
// blinking or motion blur cannot swing the score.
import { METRIC_KEYS, type Metrics } from "./geometry";

export function trimmedMean(values: number[], trim = 0.2): number {
  const sorted = [...values].sort((a, b) => a - b);
  const cut = values.length >= 5 ? Math.floor(values.length * trim) : 0;
  const kept = sorted.slice(cut, sorted.length - cut);
  return kept.reduce((a, b) => a + b, 0) / kept.length;
}

export class MetricWindow {
  private frames: Metrics[] = [];
  constructor(private readonly size = 30) {}

  push(metrics: Metrics): Metrics {
    this.frames.push(metrics);
    if (this.frames.length > this.size) this.frames.shift();
    return Object.fromEntries(METRIC_KEYS.map((k) => [k, trimmedMean(this.frames.map((f) => f[k]))])) as Metrics;
  }

  get count(): number {
    return this.frames.length;
  }

  reset(): void {
    this.frames = [];
  }
}
