"""Does MEBeauty help? Cross-dataset tests of MogNet's head and the geometry scorer.

Usage: uv run 08_mebeauty_eval.py      (~3 min; needs 06_mebeauty_prep.py first)

Compares, on SCUT's official 5-fold test faces and on MEBeauty's official
test split:
  scut      the current recipe (CLIP + ridge trained on SCUT only)
  me        CLIP + ridge trained on MEBeauty train+val only
  both      SCUT train fold + MEBeauty train+val, each dataset's ratings
            z-scored first (they use different 1-5 / 1-10 scales and raters)

Writes artifacts/mebeauty_report.json.
"""

import importlib
import json
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from common import ARTIFACTS, FEATURES, geometry_metrics, load_params, official_splits

teacher = importlib.import_module("04_teacher")
geometry = importlib.import_module("02_geometry")


def z(y: np.ndarray) -> np.ndarray:
    return (y - y.mean()) / y.std()


def pc(a, b) -> float:
    return float(pearsonr(a, b)[0])


def main() -> None:
    scut = np.load(FEATURES / "clip_b16.npz")
    s_index = {n: i for i, n in enumerate(scut["names"])}
    me = np.load(ARTIFACTS / "mebeauty_landmarks.npz")
    me_feat = np.load(FEATURES / "mebeauty_clip_b16.npz")
    assert (me_feat["names"] == me["names"]).all()
    X_me, y_me, split, group = me_feat["features"], me["y"], me["split"], me["group"]
    tr_me, te_me = split != "test", split == "test"
    print(f"MEBeauty: {tr_me.sum()} train+val, {te_me.sum()} test faces; rating mean {y_me.mean():.2f} sd {y_me.std():.2f}")

    res = defaultdict(lambda: defaultdict(list))
    me_preds = defaultdict(list)
    for setting, (tr, y_tr), (te, y_te) in official_splits():
        if setting != "5fold":
            continue
        Xtr, Xte = scut["features"][[s_index[n] for n in tr]], scut["features"][[s_index[n] for n in te]]
        fits = {
            "scut": teacher.make_model("ridge").fit(Xtr, y_tr),
            "both": teacher.make_model("ridge").fit(np.vstack([Xtr, X_me[tr_me]]), np.concatenate([z(y_tr), z(y_me[tr_me])])),
        }
        if not me_preds["me"]:
            fits["me"] = teacher.make_model("ridge").fit(X_me[tr_me], y_me[tr_me])
        for name, model in fits.items():
            res[name]["scut"].append(pc(model.predict(Xte), y_te))
            p = model.predict(X_me[te_me])
            res[name]["me"].append(pc(p, y_me[te_me]))
            me_preds[name].append(p)
        print(f"fold done: " + ", ".join(f"{k} scut={res[k]['scut'][-1]:.3f} me={res[k]['me'][-1]:.3f}" for k in fits))

    report: dict = {"mogNet": {}, "byGroup": {}}
    groups = sorted(set(group[te_me]))
    for name in ("scut", "me", "both"):
        p = np.mean(me_preds[name], axis=0)
        report["mogNet"][name] = {"scutTest5fold": round(float(np.mean(res[name]["scut"])), 4), "meTest": round(float(np.mean(res[name]["me"])), 4)}
        report["byGroup"][name] = {g: round(pc(p[group[te_me] == g], y_me[te_me][group[te_me] == g]), 3) for g in groups}
    print("\nMogNet head (Pearson r)        SCUT test (5-fold)   MEBeauty test")
    for name, r in report["mogNet"].items():
        print(f"  trained on {name:5s}              {r['scutTest5fold']:.4f}             {r['meTest']:.4f}")
    print("\nMEBeauty test r by group:")
    for g in groups:
        print(f"  {g:20s} " + "  ".join(f"{n}={report['byGroup'][n][g]:+.3f}" for n in ("scut", "me", "both")) + f"   (n={int((group[te_me] == g).sum())})")

    # The learned geometry curves (fit on SCUT) on MEBeauty, and refit on both.
    M_me = np.array([list(geometry_metrics(p, w, h).values()) for p, (w, h) in zip(me["points"], me["sizes"])])
    geo = load_params()["geometry"]
    report["geometry"] = {"scutParamsOnMe": round(pc(geometry.raw_score(M_me[te_me], geo), y_me[te_me]), 4)}
    own = geometry.calibrate(M_me[tr_me], y_me[tr_me])
    report["geometry"]["meParamsOnMe"] = round(pc(geometry.raw_score(M_me[te_me], own), y_me[te_me]), 4)
    print(f"\ngeometry on MEBeauty test: SCUT curves r={report['geometry']['scutParamsOnMe']}, refit on MEBeauty r={report['geometry']['meParamsOnMe']}")

    (ARTIFACTS / "mebeauty_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
