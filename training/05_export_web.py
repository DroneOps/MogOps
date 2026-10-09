"""Export MogNet (CLIP encoder + ridge head) to an int8 ONNX file for the browser.

Usage: uv run 05_export_web.py

  models/mognet-fp32.onnx            full-precision export (git-ignored, ~345 MB)
  web/public/models/mognet-int8.onnx dynamically quantized copy the app loads (~90 MB)

Then checks that the int8 model agrees with the PyTorch model on real faces.
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
CHECK_FACES = 300


def export() -> MogNet:
    head = np.load(MODELS / "ridge_head.npz")
    model = MogNet(ClipEmbedder(), head["mean"], head["scale"], head["coef"], head["intercept"]).eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), str(FP32), input_names=["image"], output_names=["score"], opset_version=18, dynamo=False
    )
    # Keep everything in one file (no external .data sidecar).
    onnx.save_model(onnx.load(str(FP32)), str(FP32), save_as_external_data=False)
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    print(f"fp32 {FP32.stat().st_size / 1e6:.0f} MB, int8 {INT8.stat().st_size / 1e6:.0f} MB")
    return model


def check(model: MogNet) -> dict:
    crop = load_params()["crop"]
    lm = np.load(ARTIFACTS / "landmarks.npz")
    index = {n: i for i, n in enumerate(lm["names"])}
    names, y = read_split(SPLITS / "5_folders_cross_validations_files" / "cross_validation_1" / "test_1.txt")
    names, y = names[:CHECK_FACES], y[:CHECK_FACES]
    session = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    torch_scores, onnx_scores = [], []
    with torch.inference_mode():
        for n in names:
            x = to_tensor([crop_face(Image.open(IMAGES / n), lm["points"][index[n]], crop)])
            torch_scores.append(float(model(x)[0]))
            onnx_scores.append(float(session.run(None, {"image": x.numpy()})[0][0]))
    torch_scores, onnx_scores = np.array(torch_scores), np.array(onnx_scores)
    result = {
        "agreementPC": round(float(pearsonr(torch_scores, onnx_scores)[0]), 4),
        "meanAbsDiff": round(float(np.abs(torch_scores - onnx_scores).mean()), 4),
    }
    print(f"int8 vs fp32 on {CHECK_FACES} faces: {result}")
    return result


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    quantization = check(export())
    params = load_params()["mognet"]
    params["quantization"] = quantization
    params["bytes"] = INT8.stat().st_size
    save_params("mognet", params)


if __name__ == "__main__":
    main()
