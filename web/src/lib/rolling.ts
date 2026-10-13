// Camera mode smooths the facial ratios over the last N good frames with a
// trimmed mean (drop the highest and lowest 20%), so one bad frame from
// blinking or motion blur cannot swing the score.

export function trimmedMean(values: number[], trim = 0.2): number {
  const sorted = [...values].sort((a, b) => a - b);
  const cut = values.length >= 5 ? Math.floor(values.length * trim) : 0;
  const kept = sorted.slice(cut, sorted.length - cut);
  return kept.reduce((a, b) => a + b, 0) / kept.length;
}

export class MetricWindow<K extends string> {
  private frames: Record<K, number>[] = [];
  constructor(
    private readonly keys: readonly K[],
    private readonly size = 30,
  ) {}

  push(metrics: Record<K, number>): Record<K, number> {
    this.frames.push(metrics);
    if (this.frames.length > this.size) this.frames.shift();
    return Object.fromEntries(this.keys.map((k) => [k, trimmedMean(this.frames.map((f) => f[k]))])) as Record<K, number>;
  }

  get count(): number {
    return this.frames.length;
  }

  reset(): void {
    this.frames = [];
  }
}
