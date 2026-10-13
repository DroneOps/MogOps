"""Export MogNet's encoder (CLIP ViT-B/16, flip-averaged, L2-normalized) to int8 ONNX.

Usage: uv run 05_export_web.py     (after 04_teacher.py; only needed when the
                                    encoder or crop changes, not for a new head)

  models/mognet-encoder-fp32.onnx            full precision (git-ignored, ~345 MB)
  web/public/models/mognet-encoder-int8.onnx int8-weight copy the app loads (~89 MB)

The rating head (w . e + b) lives in model-params.json, applied by the app.

Quantization is weight-only: every MatMul/Gemm weight is stored as int8 with
one scale per output channel and expanded back to float in the graph
(DequantizeLinear), so activations stay fp32. onnxruntime's quantize_dynamic
also quantizes activations, and CLIP's activation outliers made that ~10x
noisier (head output off by 0.14 sd per face, up to 2.5 points on the display
curve, vs 0.015 sd / 0.3 points here) at the same file size and speed.
A small linear correction (fp32 ~ scale * int8 + offset) is still fitted on
training faces and stored as "mognet.calibration"; agreement is then checked
on held-out test faces.
"""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from PIL import Image
from scipy.stats import pearsonr

from common import ARTIFACTS, IMAGES, MODELS, WEB_MODELS, load_params, read_split, SPLITS, save_params
from mognet import SIZE, ClipEmbedder, MogNetEncoder, crop_face, to_tensor

FP32 = MODELS / "mognet-encoder-fp32.onnx"
INT8 = WEB_MODELS / "mognet-encoder-int8.onnx"
CALIBRATION_FACES = 200
CHECK_FACES = 300


def export() -> MogNetEncoder:
    model = MogNetEncoder(ClipEmbedder()).eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        model, (dummy,), str(FP32), input_names=["image"], output_names=["embedding"], opset_version=18, dynamo=False
    )
    # Keep everything in one file (no external .data sidecar).
    onnx.save_model(onnx.load(str(FP32)), str(FP32), save_as_external_data=False)
    quantize_weights(str(FP32), str(INT8))
    print(f"fp32 {FP32.stat().st_size / 1e6:.0f} MB, int8 {INT8.stat().st_size / 1e6:.0f} MB")
    return model


def quantize_weights(src: str, dst: str) -> None:
    """Weight-only int8: symmetric, per output channel, dequantized in-graph."""
    m = onnx.load(src)
    inits = {i.name: i for i in m.graph.initializer}
    dequantize, replaced = [], set()
    for node in m.graph.node:
        if node.op_type not in ("MatMul", "Gemm") or len(node.input) < 2:
            continue
        name = node.input[1]
        if name not in inits or len(inits[name].dims) != 2:
            continue
        W = numpy_helper.to_array(inits[name]).astype(np.float32)
        trans_b = node.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in node.attribute)
        axis = 0 if trans_b else 1  # the output-channel axis of the weight
        scale = np.abs(W).max(axis=1 - axis) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(W / np.expand_dims(scale, 1 - axis)), -127, 127).astype(np.int8)
        m.graph.initializer.extend(
            [
                numpy_helper.from_array(q, f"{name}_q"),
                numpy_helper.from_array(scale.astype(np.float32), f"{name}_scale"),
                numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), f"{name}_zero"),
            ]
        )
        dequantize.append(
            helper.make_node("DequantizeLinear", [f"{name}_q", f"{name}_scale", f"{name}_zero"], [f"{name}_dq"], axis=axis)
        )
        node.input[1] = f"{name}_dq"
        replaced.add(name)
    kept = [i for i in m.graph.initializer if i.name not in replaced]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    nodes = dequantize + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save_model(m, dst, save_as_external_data=False)


def faces(split_file: str, count: int, crop: dict) -> list:
    lm = np.load(ARTIFACTS / "landmarks.npz")
    index = {n: i for i, n in enumerate(lm["names"])}
    names, _ = read_split(SPLITS / "5_folders_cross_validations_files" / "cross_validation_1" / split_file)
    return [to_tensor([crop_face(Image.open(IMAGES / n), lm["points"][index[n]], crop)]) for n in names[:count]]


def compare(model: MogNetEncoder, session: ort.InferenceSession, inputs: list) -> tuple[np.ndarray, np.ndarray]:
    """Head output (w . e + b) from the fp32 and the int8 encoder."""
    head = load_params()["mognet"]["head"]
    w, b = np.array(head["w"]), head["b"]
    with torch.inference_mode():
        reference = np.array([float(model(x)[0].numpy() @ w + b) for x in inputs])
    quantized = np.array([float(session.run(None, {"image": x.numpy()})[0][0] @ w + b) for x in inputs])
    return reference, quantized


def calibrate_and_check(model: MogNetEncoder) -> dict:
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
