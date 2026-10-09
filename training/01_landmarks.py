"""Run MediaPipe Face Landmarker on every SCUT face and measure its framing.

Usage: uv run 01_landmarks.py

Writes:
  artifacts/landmarks.npz         478 x 3 normalized points per image (+ sizes)
  web/src/lib/model-params.json   "crop": how SCUT photos frame the face, so
                                  the app and 03_embed.py crop the same way
                                  "pose": head-pose range of SCUT's frontal
                                  photos, used by the app's "face the camera" check

Re-derive just "crop"/"pose" from saved landmarks: uv run 01_landmarks.py --framing-only
"""

import sys


import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from common import ARTIFACTS, IMAGES, LANDMARKER_TASK, pose_values, save_params


def main() -> None:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_TASK)), num_faces=1
    )
    names, points, sizes, missing = [], [], [], []
    with FaceLandmarker.create_from_options(options) as landmarker:
        paths = sorted(IMAGES.glob("*.jpg"))
        for i, path in enumerate(paths):
            # mp.Image.create_from_file crashes in mediapipe 1.1.0, so decode with Pillow.
            pixels = np.ascontiguousarray(np.array(Image.open(path).convert("RGB")))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels)
            result = landmarker.detect(image)
            if not result.face_landmarks:
                missing.append(path.name)
                continue
            names.append(path.name)
            points.append([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
            sizes.append((image.width, image.height))
            if i % 1000 == 0:
                print(f"{i}/{len(paths)}", flush=True)

    points = np.array(points, dtype=np.float32)
    sizes = np.array(sizes)
    ARTIFACTS.mkdir(exist_ok=True)
    np.savez(ARTIFACTS / "landmarks.npz", names=np.array(names), points=points, sizes=sizes)
    print(f"landmarks for {len(names)} faces, no face found in {len(missing)}: {missing[:10]}")

    save_framing(points, sizes)


def save_framing(points: np.ndarray, sizes: np.ndarray) -> None:
    # Framing: SCUT photos are square crops around the face. Express the photo
    # as a square relative to the landmark bounding box: side = scale * box,
    # center = box center + offset * box.
    xy = points[:, :, :2] * sizes[:, None, :]
    lo, hi = xy.min(1), xy.max(1)
    box = (hi - lo).max(1)
    box_center = (lo + hi) / 2
    image_side = sizes.min(1)
    image_center = sizes / 2
    crop = {
        "scale": round(float(np.median(image_side / box)), 4),
        "offsetX": round(float(np.median((image_center[:, 0] - box_center[:, 0]) / box)), 4),
        "offsetY": round(float(np.median((image_center[:, 1] - box_center[:, 1]) / box)), 4),
    }
    save_params("crop", crop)
    print(f"image sizes seen: {sorted(set(map(tuple, sizes.tolist())))[:5]}")
    print(f"crop rule: {crop}")

    # Pose: allowed range = 1st-99th percentile of SCUT's frontal photos.
    values = [pose_values(p, w, h) for p, (w, h) in zip(points, sizes)]
    pose = {}
    for key in ("roll", "yaw", "pitch"):
        v = np.array([d[key] for d in values])
        lo, hi = np.quantile(v, [0.01, 0.99])
        pose[key] = [round(float(lo), 4), round(float(hi), 4)]
    save_params("pose", pose)
    print(f"pose range: {pose}")

if __name__ == "__main__":
    if "--framing-only" in sys.argv:
        saved = np.load(ARTIFACTS / "landmarks.npz")
        save_framing(saved["points"], saved["sizes"])
    else:
        main()
