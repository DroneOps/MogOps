"""Export MogNet (CLIP encoder + ridge head) to an int8 ONNX file for the browser.

Usage: uv run 05_export_web.py

  models/mognet-fp32.onnx            full-precision export (git-ignored, ~345 MB)
  web/public/models/mognet-int8.onnx dynamically quantized copy the app loads (~90 MB)

Quantization: int8 weights with per-channel scales, but the CLIP projection
and the rating head stay fp32. Rounding in the encoder still shifts scores by
a near-constant amount (~ -0.07 on the 1-5 scale), so we fit a linear
correction (fp32 ~ scale * int8 + offset) on training faces, store it in
model-params.json ("mognet.calibration") and apply it in the app. Agreement is
then checked on held-out test faces.
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from scipy.stats import pearsonr

from common import ARTIFACTS, IMAGES, MODELS, WEB_MODELS, load_params, read_split, SPLITS, save_params
from mognet import SIZE, ClipEmbedder, MogNet, crop_face, to_tensor

FP32 = MODELS / "mognet-fp32.onnx"
INT8 = WEB_MODELS / "mognet-int8.onnx"
CALIBRATION_FACES = 200
CHECK_FACES = 300
KEEP_FP32 = ["/MatMul", "/embedder/visual/MatMul"]  # rating head + CLIP projection


def export() -> MogNet:
    head = np.load(MODELS / "ridge_head.npz")
    model = MogNet(ClipEmbedder(), head["mean"], head["scale"], head["coef"], head["intercept"]).eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), str(FP32), input_names=["image"], output_names=["score"], opset_version=18, dynamo=False
    )
    # Keep everything in one file (no external .data sidecar).
    onnx.save_model(onnx.load(str(FP32)), str(FP32), save_as_external_data=False)
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=KEEP_FP32)
    print(f"fp32 {FP32.stat().st_size / 1e6:.0f} MB, int8 {INT8.stat().st_size / 1e6:.0f} MB")
    return model


def faces(split_file: str, count: int, crop: dict) -> list:
    lm = np.load(ARTIFACTS / "landmarks.npz")
    index = {n: i for i, n in enumerate(lm["names"])}
    names, _ = read_split(SPLITS / "5_folders_cross_validations_files" / "cross_validation_1" / split_file)
    return [to_tensor([crop_face(Image.open(IMAGES / n), lm["points"][index[n]], crop)]) for n in names[:count]]


def compare(model: MogNet, session: ort.InferenceSession, inputs: list) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        reference = np.array([float(model(x)[0]) for x in inputs])
    quantized = np.array([float(session.run(None, {"image": x.numpy()})[0][0]) for x in inputs])
    return reference, quantized


def calibrate_and_check(model: MogNet) -> dict:
    crop = load_params()["crop"]
    session = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    ref, q = compare(model, session, faces("train_1.txt", CALIBRATION_FACES, crop))
    scale, offset = np.polyfit(q, ref, 1)
    ref, q = compare(model, session, faces("test_1.txt", CHECK_FACES, crop))
    raw_gap = np.abs(q - ref).mean()
    q = scale * q + offset
    result = {
        "calibration": {"scale": round(float(scale), 6), "offset": round(float(offset), 6)},
        "quantization": {
            "agreementPC": round(float(pearsonr(ref, q)[0]), 4),
            "meanAbsDiff": round(float(np.abs(q - ref).mean()), 4),
            "meanAbsDiffUncalibrated": round(float(raw_gap), 4),
        },
    }
    print(f"int8 vs fp32 on {CHECK_FACES} held-out faces: {result}")
    return result


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    result = calibrate_and_check(export())
    params = load_params()["mognet"]
    params.update(result)
    params["bytes"] = INT8.stat().st_size
    save_params("mognet", params)


if __name__ == "__main__":
    main()
