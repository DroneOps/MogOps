"""Embed every SCUT face with CLIP, using the app's MediaPipe crop.

Usage: uv run 03_embed.py      (~6 min on an M1)

Writes features/clip_b16.npz (git-ignored).
"""

import time

import numpy as np
import torch
from PIL import Image

from common import ARTIFACTS, FEATURES, IMAGES, load_params
from mognet import ClipEmbedder, crop_face, to_tensor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 32


def main() -> None:
    crop = load_params()["crop"]
    lm = np.load(ARTIFACTS / "landmarks.npz")
    names, points = lm["names"], lm["points"]
    embedder = ClipEmbedder().to(DEVICE)
    out, start = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(names), BATCH):
            faces = [
                crop_face(Image.open(IMAGES / n), p, crop)
                for n, p in zip(names[i : i + BATCH], points[i : i + BATCH])
            ]
            out.append(embedder(to_tensor(faces).to(DEVICE)).float().cpu().numpy())
            if (i // BATCH) % 40 == 0:
                print(f"{i + len(faces)}/{len(names)} ({time.time() - start:.0f}s)", flush=True)
    FEATURES.mkdir(exist_ok=True)
    np.savez(FEATURES / "clip_b16.npz", names=names, features=np.concatenate(out))
    print(f"done in {time.time() - start:.0f}s -> {FEATURES / 'clip_b16.npz'}")


if __name__ == "__main__":
    main()
