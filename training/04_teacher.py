"""Fit and evaluate the rating heads on the CLIP embeddings.

Usage: uv run 04_teacher.py      (~8 min, mostly the SVR grid search)

Two heads are evaluated on the official SCUT splits, with hyperparameters
tuned by an inner 3-fold search on each training split only:
  - ridge  linear, folds into the browser ONNX model (05_export_web.py)
  - svr    RBF kernel, slightly more accurate; the offline teacher used to
           label extra faces for distilling a small on-device student

Writes:
  models/teacher_svr.joblib     offline teacher (sklearn pipeline on raw embeddings)
  models/ridge_head.npz         browser head (mean, scale, coef, intercept)
  web/src/lib/model-params.json "mognet" accuracy numbers shown in the app
"""

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from common import FEATURES, MODELS, all_labels, official_splits, save_params


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


def main() -> None:
    data = np.load(FEATURES / "clip_b16.npz")
    index = {n: i for i, n in enumerate(data["names"])}
    X = data["features"]
    summary = {}
    for head in HEADS:
        rows = {"5fold": [], "60/40": []}
        for setting, (tr, y_tr), (te, y_te) in official_splits():
            model = make_model(head).fit(X[[index[n] for n in tr]], y_tr)
            rows[setting].append(metrics(y_te, model.predict(X[[index[n] for n in te]])))
        summary[head] = {s: np.mean(r, axis=0) for s, r in rows.items()}
        for s, (pc, mae, rmse) in summary[head].items():
            print(f"{head:6s} {s:6s} PC={pc:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")

    names, y = all_labels()
    X_all = X[[index[n] for n in names]]
    MODELS.mkdir(exist_ok=True)

    svr = make_model("svr").fit(X_all, y)
    joblib.dump(svr.best_estimator_, MODELS / "teacher_svr.joblib")
    print(f"saved offline teacher (svr {svr.best_params_}) -> models/teacher_svr.joblib")

    ridge = make_model("ridge").fit(X_all, y).best_estimator_
    scaler, reg = ridge.named_steps["standardscaler"], ridge.named_steps["ridge"]
    np.savez(MODELS / "ridge_head.npz", mean=scaler.mean_, scale=scaler.scale_, coef=reg.coef_, intercept=reg.intercept_)
    print(f"saved browser head (ridge alpha={reg.alpha}) -> models/ridge_head.npz")

    pc5, mae5, rmse5 = summary["ridge"]["5fold"]
    pc64 = summary["ridge"]["60/40"][0]
    save_params(
        "mognet",
        {
            "model": "mognet-int8.onnx",
            "inputSize": 224,
            "accuracy": {
                "pc5fold": round(float(pc5), 4),
                "pc6040": round(float(pc64), 4),
                "mae5fold": round(float(mae5), 4),
                "rmse5fold": round(float(rmse5), 4),
            },
        },
    )


if __name__ == "__main__":
    main()
