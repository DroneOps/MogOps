"""The looksmaxxing "PSL" formula as a third, cheap geometric scorer, and how it holds up.

Usage: uv run 07_psl.py   (seconds; reads artifacts/landmarks.npz and, if
                           present, artifacts/mebeauty_landmarks.npz)

PSL (named after the PUAhate / Sluthate / Lookism forums) is a 1-8 rating
scale people assign by eye, not a formula. What forums and "face rating"
apps do compute are ratios with claimed ideal ranges (common.PSL_IDEALS).
This scores a face the way those tools do, with no learning from ratings:

  sub-score = 10 inside the ideal range, falling linearly to 0 at
              TOLERANCE_SD standard deviations (of SCUT faces) outside it
  harmony   = mean of the 10 sub-scores

It reports agreement with human ratings, how SCUT faces sit relative to each
ideal, and an upper bound where the same sub-scores get fitted weights.

Writes the "psl" section of web/src/lib/model-params.json (ideals, tolerances,
display curve) and web/src/lib/__fixtures__/psl.json (parity fixture).
Run after 02_geometry.py, whose fixture faces it reuses.
"""

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr

from common import (
    ARTIFACTS,
    PSL_IDEALS,
    PSL_METRICS,
    REPO,
    all_labels,
    display_knots,
    display_score,
    official_splits,
    psl_metrics,
    save_params,
)

TOLERANCE_SD = 2.0
FIXTURE = REPO / "web" / "src" / "lib" / "__fixtures__" / "psl.json"
REPORT = ARTIFACTS / "psl_report.json"


def band_scores(M: np.ndarray, tol: np.ndarray) -> np.ndarray:
    lo = np.array([PSL_IDEALS[k][0] for k in PSL_METRICS])
    hi = np.array([PSL_IDEALS[k][1] for k in PSL_METRICS])
    outside = np.maximum(lo - M, 0) + np.maximum(M - hi, 0)
    return np.clip(1 - outside / tol, 0, 1) * 10


def measure(lm_path) -> tuple[list[str], np.ndarray]:
    lm = np.load(lm_path)
    M = np.array([list(psl_metrics(p, w, h).values()) for p, (w, h) in zip(lm["points"], lm["sizes"])])
    return list(lm["names"]), M


def pc(a, b) -> float:
    return round(float(pearsonr(a, b)[0]), 3)


def main() -> None:
    names, M_all = measure(ARTIFACTS / "landmarks.npz")
    index = {n: i for i, n in enumerate(names)}
    label_names, y = all_labels()
    M = M_all[[index[n] for n in label_names]]
    tol = np.round(TOLERANCE_SD * M.std(0), 6)  # rounded as stored, so the parity fixture matches
    S = band_scores(M, tol)
    harmony = S.mean(1)
    y10 = (y - 1) / 4 * 10
    report: dict = {"scut": {}, "ratios": {}}

    print("ratio          ideal          SCUT median  in ideal  r(sub-score, rating)")
    for j, key in enumerate(PSL_METRICS):
        lo, hi, _ = PSL_IDEALS[key]
        inside = float(((M[:, j] >= lo) & (M[:, j] <= hi)).mean())
        r = pc(S[:, j], y) if S[:, j].std() > 0 else 0.0
        report["ratios"][key] = {"ideal": [lo, hi], "median": round(float(np.median(M[:, j])), 3), "inIdeal": round(inside, 3), "r": r}
        print(f"{key:13s} {lo:5.2f}-{hi:<5.2f}      {np.median(M[:, j]):7.3f}     {inside:6.1%}   {r:+.3f}")

    # Agreement on the official test folds (the formula itself has nothing to train).
    row = {n: i for i, n in enumerate(label_names)}
    folds = defaultdict(list)
    fitted = defaultdict(list)
    for setting, (tr, y_tr), (te, y_te) in official_splits():
        S_tr, S_te = S[[row[n] for n in tr]], S[[row[n] for n in te]]
        folds[setting].append(pearsonr(S_te.mean(1), y_te)[0])
        fit = lsq_linear(np.hstack([S_tr, np.ones((len(S_tr), 1))]), y_tr, bounds=([0.0] * S.shape[1] + [-np.inf], np.inf))
        fitted[setting].append(pearsonr(S_te @ fit.x[:-1], y_te)[0])
    groups = np.array([n[:2] for n in label_names])
    report["scut"] = {
        "pc5fold": round(float(np.mean(folds["5fold"])), 3),
        "pc6040": round(float(np.mean(folds["60/40"])), 3),
        "fittedWeightsPc5fold": round(float(np.mean(fitted["5fold"])), 3),
        "byGroup": {g: pc(harmony[groups == g], y[groups == g]) for g in ("AF", "AM", "CF", "CM")},
    }
    print(f"\nPSL harmony vs SCUT ratings: 5-fold r={report['scut']['pc5fold']}, 60/40 r={report['scut']['pc6040']}")
    print(f"  same sub-scores, NNLS-fitted weights: 5-fold r={report['scut']['fittedWeightsPc5fold']}")
    print(f"  by group: {report['scut']['byGroup']}")

    # Distribution of the literal formula vs. the human ratings (both 0-10).
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    report["distribution"] = {
        "humans": {"mean": round(float(y10.mean()), 2), "sd": round(float(y10.std()), 2), "quantiles": np.round(np.quantile(y10, qs), 2).tolist()},
        "pslHarmony": {"mean": round(float(harmony.mean()), 2), "sd": round(float(harmony.std()), 2), "quantiles": np.round(np.quantile(harmony, qs), 2).tolist()},
        "histogramEdges": list(range(11)),
        "humansHist": np.histogram(y10, bins=range(11))[0].tolist(),
        "pslHist": np.histogram(harmony, bins=range(11))[0].tolist(),
    }
    print(f"\ndistribution on 0-10 (5/25/50/75/95th pct):")
    print(f"  human ratings  mean {y10.mean():.2f} sd {y10.std():.2f}  {np.round(np.quantile(y10, qs), 2)}")
    print(f"  PSL harmony    mean {harmony.mean():.2f} sd {harmony.std():.2f}  {np.round(np.quantile(harmony, qs), 2)}")

    reference = [harmony]
    me_path = ARTIFACTS / "mebeauty_landmarks.npz"
    if me_path.exists():
        me = np.load(me_path)
        _, M_me = measure(me_path)
        h_me = band_scores(M_me, tol).mean(1)
        reference.append(h_me)
        by = {g: pc(h_me[me["group"] == g], me["y"][me["group"] == g]) for g in sorted(set(me["group"]))}
        report["mebeauty"] = {"pc": pc(h_me, me["y"]), "byGroup": by}
        print(f"\nMEBeauty (2.5k multi-ethnic faces, never seen): r={report['mebeauty']['pc']}")
        print(f"  by group: {by}")

    knots = display_knots(np.concatenate(reference))  # SCUT + MEBeauty faces
    save_params(
        "psl",
        {
            "ideals": [{"key": k, "lo": PSL_IDEALS[k][0], "hi": PSL_IDEALS[k][1], "tol": float(t)} for k, t in zip(PSL_METRICS, tol)],
            "display": knots,
            "accuracy": {"pc5fold": report["scut"]["pc5fold"], "pc6040": report["scut"]["pc6040"]},
        },
    )
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote psl params and {REPORT.relative_to(REPO)}")

    # Parity fixture: the same synthetic faces as the geometry fixture.
    geometry_cases = json.loads((FIXTURE.parent / "geometry.json").read_text())
    cases = []
    for case in geometry_cases:
        pts = np.array(case["landmarks"])
        m = psl_metrics(pts, case["width"], case["height"])
        subs = band_scores(np.array([list(m.values())]), tol)[0]
        cases.append({"metrics": m, "subScores": dict(zip(PSL_METRICS, subs.tolist())), "overall": float(display_score(subs.mean(), knots))})
    FIXTURE.write_text(json.dumps(cases))


if __name__ == "__main__":
    main()
