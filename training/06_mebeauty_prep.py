"""Landmarks + CLIP embeddings for MEBeauty, a second, multi-ethnic dataset.

Usage:
  git clone --depth 1 --filter=blob:none --sparse https://github.com/fbplab/MEBeauty-database.git data/mebeauty/repo
  git -C data/mebeauty/repo sparse-checkout set scores original_images
  uv run 06_mebeauty_prep.py      (~5 min on an M1)

MEBeauty: 2,550 in-the-wild photos (Black, Asian, Caucasian, Hispanic,
Indian, Mideastern; female and male), rated 1-10 by ~300 raters. It ships
train/val/test lists of average scores. The repo has no license file, so
treat it like SCUT: research only, never committed.

Writes (git-ignored):
  artifacts/mebeauty_landmarks.npz   names, points, sizes, split, y, group
  features/mebeauty_clip_b16.npz     names, features (same crop + encoder as SCUT)
"""

import time

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image, ImageOps

from common import ARTIFACTS, FEATURES, LANDMARKER_TASK, ROOT, load_params
from mognet import ClipEmbedder, crop_face, to_tensor

ME = ROOT / "data" / "mebeauty" / "repo"
LISTED_PREFIX = "./cropped_images/images_crop_align_mtcnn/"
MAX_SIDE = 1280  # the originals are up to 1920 px; landmarks are resolution-free
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 32


def listed_faces():
    """(relative path, split, rating 1-10, "gender/ethnicity") for every listed photo that exists."""
    rows = []
    for split in ("train", "val", "test"):
        for line in (ME / "scores" / f"{split}_2022.txt").read_text().splitlines():
            if not line.strip():
                continue
            path, score = line.rsplit(" ", 1)
            rel = path.replace(LISTED_PREFIX, "")
            if (ME / "original_images" / rel).exists():
                rows.append((rel, split, float(score), "/".join(rel.split("/")[:2])))
    return rows


def load(rel: str) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(ME / "original_images" / rel)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def main() -> None:
    rows = listed_faces()
    print(f"{len(rows)} listed photos found")
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(LANDMARKER_TASK)), num_faces=1)
    kept, points, sizes, missing = [], [], [], 0
    with FaceLandmarker.create_from_options(options) as landmarker:
        for i, row in enumerate(rows):
            image = load(row[0])
            pixels = np.ascontiguousarray(np.array(image))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels))
            if not result.face_landmarks:
                missing += 1
                continue
            kept.append(row)
            points.append([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
            sizes.append(image.size)
            if i % 500 == 0:
                print(f"landmarks {i}/{len(rows)}", flush=True)
    names = np.array([r[0] for r in kept])
    points = np.array(points, dtype=np.float32)
    np.savez(
        ARTIFACTS / "mebeauty_landmarks.npz",
        names=names,
        points=points,
        sizes=np.array(sizes),
        split=np.array([r[1] for r in kept]),
        y=np.array([r[2] for r in kept]),
        group=np.array([r[3] for r in kept]),
    )
    print(f"landmarks for {len(kept)} faces, no face in {missing}")

    crop = load_params()["crop"]
    embedder = ClipEmbedder().to(DEVICE)
    out, start = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(names), BATCH):
            faces = [crop_face(load(n), p, crop) for n, p in zip(names[i : i + BATCH], points[i : i + BATCH])]
            out.append(embedder(to_tensor(faces).to(DEVICE)).float().cpu().numpy())
            if (i // BATCH) % 20 == 0:
                print(f"embedded {i + len(faces)}/{len(names)} ({time.time() - start:.0f}s)", flush=True)
    np.savez(FEATURES / "mebeauty_clip_b16.npz", names=names, features=np.concatenate(out))
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
