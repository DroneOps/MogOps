"""Score any face photo with the offline teacher and the geometry scorer.

Usage: uv run predict.py photo1.jpg [photo2.jpg ...]

Runs the same steps as the browser app: MediaPipe landmarks -> SCUT-style
crop -> CLIP embedding -> teacher head; plus the learned geometry score.
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

from common import LANDMARKER_TASK, METRICS, MODELS, geometry_metrics, load_params
from mognet import ClipEmbedder, crop_face, to_tensor

geometry = importlib.import_module("02_geometry")


def to_ten(score_1_to_5: float) -> float:
    return float(np.clip((score_1_to_5 - 1) / 4 * 10, 0, 10))


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
            mognet = to_ten(float(teacher.predict(emb)[0]))
            vec = np.array([list(geometry_metrics(points, image.width, image.height).values())])
            geo = geometry.display_score(geometry.raw_score(vec, params["geometry"]), params["geometry"]["display"])[0]
            subs = geometry.sub_scores(vec, params["geometry"]["curves"])[0]
            breakdown = "  ".join(f"{k}={s:.1f}" for k, s in zip(METRICS, subs))
            print(f"MogNet {mognet:4.1f}/10  geometry {geo:4.1f}/10  {path}\n    {breakdown}")


if __name__ == "__main__":
    main(sys.argv[1:])
