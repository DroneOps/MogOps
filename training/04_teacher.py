"""Fit and evaluate the rating heads on the CLIP embeddings.

Usage: uv run 04_teacher.py [--skip-svr]   (~8 min, mostly the SVR grid search)

Two heads are evaluated on the official SCUT splits, with hyperparameters
tuned by an inner 3-fold search on each training split only:
  - ridge  linear, the browser head
  - svr    RBF kernel, slightly more accurate; the offline teacher used to
           label extra faces for distilling a small on-device student
           (--skip-svr keeps the saved one)

The browser head is trained on SCUT + MEBeauty (06_mebeauty_prep.py), with
each dataset's ratings z-scored, because a SCUT-only head drops from r 0.93
to 0.66 on MEBeauty's multi-ethnic faces (08_mebeauty_eval.py). Its output is
in z units; the app shows it through the target display curve, using
out-of-fold predictions on both datasets as the reference faces.

Writes:
  models/teacher_svr.joblib     offline teacher (SCUT only, sklearn pipeline on raw embeddings)
  models/ridge_head.npz         browser head (mean, scale, coef, intercept)
  web/src/lib/model-params.json "mognet": head (w, b on the L2-normalized
                                embedding), display curve, accuracy
"""

import sys

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from common import FEATURES, MODELS, all_labels, display_knots, official_splits, save_params


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


HEADS = {
    "ridge": (Ridge(), {"ridge__alpha": [30, 100, 300, 1000, 3000]}),
    "svr": (SVR(kernel="rbf", gamma="scale"), {"svr__C": [0.3, 1, 3], "svr__epsilon": [0.05, 0.1]}),
}


def make_model(head: str) -> GridSearchCV:
    estimator, grid = HEADS[head]
    pipe = make_pipeline(FunctionTransformer(l2_normalize), StandardScaler(), estimator)
    return GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)


def metrics(y, p):
    return pearsonr(y, p)[0], np.abs(y - p).mean(), np.sqrt(((y - p) ** 2).mean())


def z(y: np.ndarray) -> np.ndarray:
    return (y - y.mean()) / y.std()


def load_mebeauty():
    from common import ARTIFACTS

    feats = np.load(FEATURES / "mebeauty_clip_b16.npz")
    meta = np.load(ARTIFACTS / "mebeauty_landmarks.npz")
    return feats["features"], meta["y"], meta["split"]


def main() -> None:
    data = np.load(FEATURES / "clip_b16.npz")
    index = {n: i for i, n in enumerate(data["names"])}
    X = data["features"]
    names, y = all_labels()
    X_all = X[[index[n] for n in names]]
    MODELS.mkdir(exist_ok=True)

    if "--skip-svr" not in sys.argv:
        summary = {}
        for head in HEADS:
            rows = {"5fold": [], "60/40": []}
            for setting, (tr, y_tr), (te, y_te) in official_splits():
                model = make_model(head).fit(X[[index[n] for n in tr]], y_tr)
                rows[setting].append(metrics(y_te, model.predict(X[[index[n] for n in te]])))
            summary[head] = {s: np.mean(r, axis=0) for s, r in rows.items()}
            for s, (pc, mae, rmse) in summary[head].items():
                print(f"SCUT-only {head:6s} {s:6s} PC={pc:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")
        svr = make_model("svr").fit(X_all, y)
        joblib.dump(svr.best_estimator_, MODELS / "teacher_svr.joblib")
        print(f"saved offline teacher (svr {svr.best_params_}) -> models/teacher_svr.joblib")

    # Browser head: SCUT + MEBeauty. Evaluate on SCUT's official 5 folds and
    # MEBeauty's test split, and collect out-of-fold predictions for every
    # face of both datasets as the display reference.
    X_me, y_me, split_me = load_mebeauty()
    me_train, me_test = split_me != "test", split_me == "test"
    row = {n: i for i, n in enumerate(names)}
    oof_scut, pc_scut, pc_me = np.zeros(len(names)), [], []
    for setting, (tr, y_tr), (te, y_te) in official_splits():
        if setting != "5fold":
            continue
        tr_i, te_i = [row[n] for n in tr], [row[n] for n in te]
        model = make_model("ridge").fit(np.vstack([X_all[tr_i], X_me[me_train]]), np.concatenate([z(y_tr), z(y_me[me_train])]))
        oof_scut[te_i] = model.predict(X_all[te_i])
        pc_scut.append(pearsonr(oof_scut[te_i], y_te)[0])
        pc_me.append(pearsonr(model.predict(X_me[me_test]), y_me[me_test])[0])
    oof_me = np.zeros(len(y_me))
    for k, held in enumerate(np.array_split(np.random.default_rng(0).permutation(len(y_me)), 5)):
        keep = np.setdiff1d(np.arange(len(y_me)), held)
        model = make_model("ridge").fit(np.vstack([X_all, X_me[keep]]), np.concatenate([z(y), z(y_me[keep])]))
        oof_me[held] = model.predict(X_me[held])
    print(f"SCUT+MEBeauty ridge: SCUT 5-fold PC={np.mean(pc_scut):.4f}, MEBeauty test PC={np.mean(pc_me):.4f}")

    ridge = make_model("ridge").fit(np.vstack([X_all, X_me]), np.concatenate([z(y), z(y_me)])).best_estimator_
    scaler, reg = ridge.named_steps["standardscaler"], ridge.named_steps["ridge"]
    np.savez(MODELS / "ridge_head.npz", mean=scaler.mean_, scale=scaler.scale_, coef=reg.coef_, intercept=reg.intercept_)
    # The pipeline is linear after L2 normalization, so it folds into w.e + b.
    w = reg.coef_ / scaler.scale_
    b = float(reg.intercept_ - scaler.mean_ @ w)
    print(f"saved browser head (ridge alpha={reg.alpha}) -> models/ridge_head.npz")

    save_params(
        "mognet",
        {
            "model": "mognet-encoder-int8.onnx",
            "inputSize": 224,
            "head": {"w": [round(float(v), 7) for v in w], "b": round(b, 7)},
            "display": display_knots(np.concatenate([oof_scut, oof_me])),
            "accuracy": {
                "pc5fold": round(float(np.mean(pc_scut)), 4),
                "pcMEBeautyTest": round(float(np.mean(pc_me)), 4),
            },
        },
    )


if __name__ == "__main__":
    main()
