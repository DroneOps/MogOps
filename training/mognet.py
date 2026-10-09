"""MogNet: CLIP ViT-B/16 image encoder + a linear rating head.

The same modules are used to embed the training faces (03_embed.py) and to
export the browser model (05_export_web.py), so the app sees exactly the
preprocessing the head was trained on:

  face photo -> MediaPipe crop (common.face_crop_box) -> 224x224 RGB in [0, 1]
             -> CLIP normalization -> encoder(original) + encoder(mirrored)
             -> average -> L2 normalize -> standardize -> ridge -> SCUT 1-5
"""

import numpy as np
import torch
from PIL import Image
from torch import nn

from common import face_crop_box

SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PAD_GRAY = (128, 128, 128)


def crop_face(image: Image.Image, landmarks: np.ndarray, crop: dict) -> Image.Image:
    """Square crop around the face, gray where it runs past the photo edge.

    Mirrors web/src/lib/crop.ts: canvas drawImage over a gray background.
    """
    cx, cy, side = face_crop_box(landmarks, image.width, image.height, crop)
    left, top = round(cx - side / 2), round(cy - side / 2)
    side = round(side)
    canvas = Image.new("RGB", (side, side), PAD_GRAY)
    canvas.paste(image.convert("RGB"), (-left, -top))
    return canvas.resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def to_tensor(images: list[Image.Image]) -> torch.Tensor:
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class ClipEmbedder(nn.Module):
    """[B, 3, 224, 224] RGB in [0, 1] -> flip-averaged CLIP embedding [B, 512]."""

    def __init__(self):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.visual = model.visual.eval()
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        both = self.visual(torch.cat([x, torch.flip(x, dims=[3])], dim=0))
        original, mirrored = both.chunk(2, dim=0)
        return (original + mirrored) / 2


class MogNet(nn.Module):
    """Embedder + linear head folded into one graph: image -> SCUT score (1-5)."""

    def __init__(self, embedder: ClipEmbedder, mean, scale, coef, intercept):
        super().__init__()
        self.embedder = embedder
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
        self.intercept = float(intercept)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedder(x)
        e = e / e.norm(dim=1, keepdim=True)
        return ((e - self.mean) / self.scale) @ self.coef + self.intercept
