"""Calibrate the geometry scorer (the Mogwars-style breakdown) from SCUT ratings.

Usage: uv run 02_geometry.py

Mogwars/Omoggle hand-pick an "ideal band" for each facial ratio (a flat 10
inside the band, falling to 0 outside) plus a hand-picked weight per trait.
We learn both from data instead:

  1. curve    For each ratio, split faces into 10 equal-size bins and record
              the median ratio and mean rating per bin (lightly smoothed).
              A face's sub-score is its rating on that curve, rescaled 0-10.
              Unlike a flat band this captures one-sided trends
              ("narrower jaw rated higher") as well as sweet spots.
  2. weights  Non-negative least squares of rating on the sub-scores.
  3. display  The weighted sum goes through the shared target display curve
              (common.display_knots), with SCUT + MEBeauty faces as reference.

On the official splits this reaches PC ~0.60, versus ~0.29 for the same
pipeline with Mogwars-style flat bands (printed below for comparison).

Writes the "geometry" section of web/src/lib/model-params.json and a parity
fixture used by web/src/lib/geometry.test.ts.
"""

import json

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr

from common import (
    ARTIFACTS,
    METRICS,
    REPO,
    all_labels,
    display_knots,
    display_score,
    geometry_metrics,
    official_splits,
    pose_values,
    save_params,
)

BINS = 10
# MediaPipe's symmetry measure correlates the "wrong" way with SCUT ratings
# (+0.23: more asymmetric faces were rated higher), almost certainly a dataset
# or mesh-smoothing confound. Keep it as a display-only sub-score where more
# symmetric always scores higher, and give it zero weight in the overall.
DISPLAY_ONLY = {"symmetry"}
FIXTURE = REPO / "web" / "src" / "lib" / "__fixtures__" / "geometry.json"


def fit_curve(v: np.ndarray, y: np.ndarray) -> dict:
    edges = np.quantile(v, np.linspace(0, 1, BINS + 1))
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, BINS - 1)
    xs = np.array([np.median(v[bins == k]) for k in range(BINS)])
    ys = np.array([y[bins == k].mean() for k in range(BINS)])
    ys = np.convolve(np.pad(ys, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    return {"x": xs.tolist(), "y": ys.tolist()}


def curve_score(v: np.ndarray, curve: dict) -> np.ndarray:
    ys = np.array(curve["y"])
    lo, hi = ys.min(), ys.max()
    return (np.interp(v, curve["x"], ys) - lo) / max(hi - lo, 1e-9) * 10


def sub_scores(M: np.ndarray, curves: list[dict]) -> np.ndarray:
    return np.stack([curve_score(M[:, j], c) for j, c in enumerate(curves)], axis=1)


def calibrate(M: np.ndarray, y: np.ndarray) -> dict:
    curves = []
    for j, key in enumerate(METRICS):
        if key in DISPLAY_ONLY:
            # Straight line: 10 at the 5th percentile of asymmetry, 0 at the 95th.
            lo, hi = np.quantile(M[:, j], [0.05, 0.95])
            curves.append({"key": key, "x": [float(lo), float(hi)], "y": [10.0, 0.0]})
        else:
            curves.append({"key": key, **fit_curve(M[:, j], y)})
    S = sub_scores(M, curves)
    # Non-negative weights + free intercept: a better trait never lowers the score.
    scored = [j for j, k in enumerate(METRICS) if k not in DISPLAY_ONLY]
    A = np.hstack([S[:, scored], np.ones((len(S), 1))])
    fit = lsq_linear(A, y, bounds=([0.0] * len(scored) + [-np.inf], np.inf))
    w = np.zeros(len(METRICS))
    w[scored] = fit.x[:-1]
    w = w / w.sum() if w.sum() > 0 else np.full(len(METRICS), 1 / len(METRICS))
    return {"curves": curves, "weights": w.tolist(), "display": display_knots(S @ w)}


def raw_score(M: np.ndarray, params: dict) -> np.ndarray:
    return sub_scores(M, params["curves"]) @ np.array(params["weights"])


def flat_band_baseline(Mtr, ytr, Mte):
    """Mogwars-style flat bands with data-picked edges, for comparison only."""
    top = ytr >= np.quantile(ytr, 0.8)
    S_tr, S_te = [], []
    for j in range(Mtr.shape[1]):
        good_lo, good_hi = np.quantile(Mtr[top, j], [0.35, 0.65])
        floor_lo, floor_hi = np.quantile(Mtr[:, j], [0.01, 0.99])

        def band(v):
            below = np.clip((v - floor_lo) / max(good_lo - floor_lo, 1e-9) * 10, 0, 10)
            above = np.clip((floor_hi - v) / max(floor_hi - good_hi, 1e-9) * 10, 0, 10)
            return np.where(v < good_lo, below, np.where(v > good_hi, above, 10.0))

        S_tr.append(band(Mtr[:, j]))
        S_te.append(band(Mte[:, j]))
    S_tr, S_te = np.stack(S_tr, 1), np.stack(S_te, 1)
    fit = lsq_linear(np.hstack([S_tr, np.ones((len(S_tr), 1))]), ytr, bounds=([0.0] * S_tr.shape[1] + [-np.inf], np.inf))
    return S_te @ fit.x[:-1]


def main() -> None:
    lm = np.load(ARTIFACTS / "landmarks.npz")
    index = {n: i for i, n in enumerate(lm["names"])}
    M = np.array(
        [list(geometry_metrics(p, w, h).values()) for p, (w, h) in zip(lm["points"], lm["sizes"])]
    )

    ours, bands = {"5fold": [], "60/40": []}, {"5fold": [], "60/40": []}
    for setting, (tr, y_tr), (te, y_te) in official_splits():
        Mtr, Mte = M[[index[n] for n in tr]], M[[index[n] for n in te]]
        ours[setting].append(pearsonr(raw_score(Mte, calibrate(Mtr, y_tr)), y_te)[0])
        bands[setting].append(pearsonr(flat_band_baseline(Mtr, y_tr, Mte), y_te)[0])

    pc5, pc64 = np.mean(ours["5fold"]), np.mean(ours["60/40"])
    print(f"learned curves (ours)    5fold PC={pc5:.4f}  60/40 PC={pc64:.4f}")
    print(f"flat bands (Mogwars-ish) 5fold PC={np.mean(bands['5fold']):.4f}  60/40 PC={np.mean(bands['60/40']):.4f}")

    names, y = all_labels()
    Mall = M[[index[n] for n in names]]
    for j, key in enumerate(METRICS):
        print(f"  {key:13s} correlation with rating {pearsonr(Mall[:, j], y)[0]:+.3f}")

    final = calibrate(Mall, y)
    me_path = ARTIFACTS / "mebeauty_landmarks.npz"
    if me_path.exists():
        # Display reference: SCUT + MEBeauty faces, closer to the variety a webcam sees.
        me = np.load(me_path)
        M_me = np.array([list(geometry_metrics(p, w, h).values()) for p, (w, h) in zip(me["points"], me["sizes"])])
        final["display"] = display_knots(np.concatenate([raw_score(Mall, final), raw_score(M_me, final)]))
    final["accuracy"] = {"pc5fold": round(float(pc5), 4), "pc6040": round(float(pc64), 4)}
    save_params("geometry", final)
    print("weights:", {k: round(w, 3) for k, w in zip(METRICS, final["weights"])})

    # Parity fixture: synthetic faces (mean SCUT face + noise, never a real
    # person) on non-square frames to exercise aspect-ratio handling.
    rng = np.random.default_rng(7)
    mean_face = lm["points"].mean(0)
    cases = []
    for k in range(4):
        pts = np.round(mean_face + rng.normal(0, 0.004, mean_face.shape) * (k > 0), 6)
        w, h = (640, 480) if k % 2 else (480, 640)
        m = geometry_metrics(pts, w, h)
        vec = np.array([list(m.values())])
        cases.append(
            {
                "width": w,
                "height": h,
                "landmarks": pts.tolist(),
                "metrics": m,
                "pose": pose_values(pts, w, h),
                "subScores": dict(zip(METRICS, sub_scores(vec, final["curves"])[0].tolist())),
                "overall": float(display_score(raw_score(vec, final), final["display"])[0]),
            }
        )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(cases))
    print(f"wrote {FIXTURE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
