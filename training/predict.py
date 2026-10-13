"""Score any face photo the way the app does (minus the expression bonuses).

Usage: uv run predict.py photo1.jpg [photo2.jpg ...]

MediaPipe landmarks -> SCUT-style crop -> CLIP embedding -> browser head ->
display curve (fp32, so it can differ from the int8 app by ~0.1-0.2), plus
the geometry and PSL-ratio scores and the offline SVR teacher's raw SCUT
rating (1-5).
"""

import importlib
import sys

import joblib
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image, ImageOps

from common import LANDMARKER_TASK, METRICS, MODELS, display_score, geometry_metrics, load_params, psl_metrics
from mognet import ClipEmbedder, crop_face, to_tensor

geometry = importlib.import_module("02_geometry")
psl = importlib.import_module("07_psl")
# teacher_svr.joblib was pickled by 04_teacher.py running as __main__, so
# unpickling looks for its l2_normalize step on whatever __main__ is now.
l2_normalize = importlib.import_module("04_teacher").l2_normalize


def main(paths: list[str]) -> None:
    params = load_params()
    teacher = joblib.load(MODELS / "teacher_svr.joblib")
    embedder = ClipEmbedder()
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(LANDMARKER_TASK)), num_faces=1)
    with FaceLandmarker.create_from_options(options) as landmarker:
        for path in paths:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            pixels = np.ascontiguousarray(np.array(image))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels))
            if not result.face_landmarks:
                print(f"  no face found  {path}")
                continue
            points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
            with torch.inference_mode():
                emb = embedder(to_tensor([crop_face(image, points, params["crop"])])).numpy()
            teacher_rating = float(teacher.predict(emb)[0])
            head = params["mognet"]["head"]
            e = emb[0] / np.linalg.norm(emb[0])
            mognet = float(display_score(e @ np.array(head["w"]) + head["b"], params["mognet"]["display"]))
            vec = np.array([list(geometry_metrics(points, image.width, image.height).values())])
            geo = display_score(geometry.raw_score(vec, params["geometry"]), params["geometry"]["display"])[0]
            subs = geometry.sub_scores(vec, params["geometry"]["curves"])[0]
            tol = np.array([r["tol"] for r in params["psl"]["ideals"]])
            harmony = psl.band_scores(np.array([list(psl_metrics(points, image.width, image.height).values())]), tol).mean()
            ratios = display_score(harmony, params["psl"]["display"])
            breakdown = "  ".join(f"{k}={s:.1f}" for k, s in zip(METRICS, subs))
            print(
                f"MogNet {mognet:4.1f}/10  geometry {geo:4.1f}/10  PSL ratios {ratios:4.1f}/10  "
                f"(SVR teacher {teacher_rating:.2f}/5)  {path}\n    {breakdown}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
