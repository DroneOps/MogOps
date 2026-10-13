"""Shared paths, dataset helpers, geometry metrics and the face-crop rule.

Everything in here has a TypeScript twin in web/src/lib so the browser app
measures faces exactly the way the models were trained:
  - geometry_metrics()  <->  web/src/lib/geometry.ts  computeMetrics()
  - face_crop_box()     <->  web/src/lib/crop.ts      faceCropBox()
web/src/lib/geometry.test.ts checks that both sides agree.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data" / "SCUT-FBP5500_v2"
IMAGES = DATA / "Images"
SPLITS = DATA / "train_test_files"
ARTIFACTS = ROOT / "artifacts"
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
WEB_MODELS = REPO / "web" / "public" / "models"
WEB_PARAMS = REPO / "web" / "src" / "lib" / "model-params.json"
LANDMARKER_TASK = WEB_MODELS / "face_landmarker.task"


# ---------------------------------------------------------------------------
# Dataset splits


def read_split(path: Path) -> tuple[list[str], np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])


def all_labels() -> tuple[list[str], np.ndarray]:
    return read_split(SPLITS / "All_labels.txt")


def official_splits():
    """Yield (setting, (train_names, train_y), (test_names, test_y)).

    The five 80/20 cross-validation folds and the 60/40 split shipped with
    SCUT-FBP5500, so our numbers compare directly with the paper's tables.
    """
    folds = SPLITS / "5_folders_cross_validations_files"
    for k in range(1, 6):
        yield "5fold", read_split(folds / f"cross_validation_{k}" / f"train_{k}.txt"), read_split(
            folds / f"cross_validation_{k}" / f"test_{k}.txt"
        )
    split = SPLITS / "split_of_60%training and 40%testing"
    yield "60/40", read_split(split / "train.txt"), read_split(split / "test.txt")


# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker indices (478-point mesh; "right" = subject's right,
# which appears on the left of an unmirrored photo).

FOREHEAD = 10
GLABELLA = 9
NASION = 168
NOSE_TIP = 1
SUBNASALE = 2
UPPER_LIP = 0
LOWER_LIP = 17
CHIN = 152

R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM = 33, 133, 159, 145
L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOTTOM = 263, 362, 386, 374
R_BROW, L_BROW = 105, 334
R_ZYGION, L_ZYGION = 234, 454
R_GONION, L_GONION = 172, 397

MIDLINE = [FOREHEAD, GLABELLA, NASION, NOSE_TIP, SUBNASALE, UPPER_LIP, LOWER_LIP, CHIN]

# Left/right mirror pairs used for symmetry.
SYMMETRY_PAIRS = [
    (R_EYE_OUTER, L_EYE_OUTER),
    (R_EYE_INNER, L_EYE_INNER),
    (R_EYE_TOP, L_EYE_TOP),
    (R_EYE_BOTTOM, L_EYE_BOTTOM),
    (R_BROW, L_BROW),
    (70, 300),  # outer brow
    (R_ZYGION, L_ZYGION),
    (116, 345),  # cheekbone
    (R_GONION, L_GONION),
    (136, 365),  # lower jaw
    (148, 377),  # chin side
    (61, 291),  # mouth corners
    (98, 327),  # nostril wings
]

METRICS = [
    "symmetry",
    "canthalTilt",
    "eyeAspect",
    "eyeSpacing",
    "midfaceRatio",
    "jawRatio",
    "fwhr",
    "lowerThird",
]


def _aligned(points: np.ndarray) -> np.ndarray:
    """Rotate pixel-space (x, y) points so the eye line is horizontal."""
    eyes_r = (points[R_EYE_OUTER] + points[R_EYE_INNER]) / 2
    eyes_l = (points[L_EYE_OUTER] + points[L_EYE_INNER]) / 2
    dx, dy = eyes_l - eyes_r
    angle = np.arctan2(dy, dx)
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    return (points - eyes_r) @ rot.T


def geometry_metrics(landmarks: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Raw facial ratios from normalized MediaPipe landmarks (N x 2 or N x 3).

    All values are scale-free so they do not depend on image size or distance.
    """
    p = _aligned(landmarks[:, :2] * np.array([width, height], dtype=np.float64))
    dist = lambda a, b: float(np.hypot(*(p[a] - p[b])))  # noqa: E731

    face_width = dist(R_ZYGION, L_ZYGION)
    face_height = dist(FOREHEAD, CHIN)

    # Symmetry: mirror every right-side point across the facial midline and
    # measure how far it lands from its left-side partner (lower = more symmetric).
    mid_x = float(np.median(p[MIDLINE, 0]))
    residuals = []
    for r, l in SYMMETRY_PAIRS:
        residuals.append(abs((p[r, 0] - mid_x) + (p[l, 0] - mid_x)) / face_width)
        residuals.append(abs(p[r, 1] - p[l, 1]) / face_height)
    asymmetry = float(np.mean(residuals))

    # Canthal tilt: degrees the outer eye corner sits above the inner one.
    # Image y grows downward, so "inner y - outer y" is positive when the
    # outer corner is higher.
    tilt_r = np.degrees(np.arctan2(p[R_EYE_INNER, 1] - p[R_EYE_OUTER, 1], p[R_EYE_INNER, 0] - p[R_EYE_OUTER, 0]))
    tilt_l = np.degrees(np.arctan2(p[L_EYE_INNER, 1] - p[L_EYE_OUTER, 1], p[L_EYE_OUTER, 0] - p[L_EYE_INNER, 0]))
    canthal_tilt = float((tilt_r + tilt_l) / 2)

    eye_w_r, eye_w_l = dist(R_EYE_OUTER, R_EYE_INNER), dist(L_EYE_OUTER, L_EYE_INNER)
    eye_aspect = (dist(R_EYE_TOP, R_EYE_BOTTOM) / eye_w_r + dist(L_EYE_TOP, L_EYE_BOTTOM) / eye_w_l) / 2
    eye_spacing = dist(R_EYE_INNER, L_EYE_INNER) / ((eye_w_r + eye_w_l) / 2)

    eye_center_r = (p[R_EYE_OUTER] + p[R_EYE_INNER]) / 2
    eye_center_l = (p[L_EYE_OUTER] + p[L_EYE_INNER]) / 2
    eye_line_y = (eye_center_r[1] + eye_center_l[1]) / 2
    interpupil = float(np.hypot(*(eye_center_l - eye_center_r)))
    midface_ratio = interpupil / abs(p[UPPER_LIP, 1] - eye_line_y)

    jaw_ratio = dist(R_GONION, L_GONION) / face_width
    brow_y = (p[R_BROW, 1] + p[L_BROW, 1]) / 2
    fwhr = face_width / abs(p[UPPER_LIP, 1] - brow_y)
    lower_third = abs(p[CHIN, 1] - p[SUBNASALE, 1]) / abs(p[SUBNASALE, 1] - p[GLABELLA, 1])

    return {
        "symmetry": asymmetry,
        "canthalTilt": canthal_tilt,
        "eyeAspect": float(eye_aspect),
        "eyeSpacing": float(eye_spacing),
        "midfaceRatio": float(midface_ratio),
        "jawRatio": float(jaw_ratio),
        "fwhr": float(fwhr),
        "lowerThird": float(lower_third),
    }


# ---------------------------------------------------------------------------
# "PSL" ratios: the proportions looksmaxxing forums score faces by, each with
# the community's claimed ideal range. These ideals are forum/clinic lore,
# not peer-reviewed, and assume real anthropometric landmarks; MediaPipe's
# points sit elsewhere for some of them (e.g. its canthi make eyes look
# narrower). 07_psl.py measures how well this agrees with real ratings.

R_IRIS, L_IRIS = 468, 473
R_ALA, L_ALA = 64, 294
R_MOUTH, L_MOUTH = 61, 291
UPPER_LIP_INNER, LOWER_LIP_INNER = 13, 14

# key: (ideal low, ideal high, what it is)
PSL_IDEALS = {
    "canthalTilt": (4.0, 8.0, "outer eye corner above inner, degrees"),
    "fwhr": (1.8, 2.0, "cheekbone width / brow-to-upper-lip height"),
    "midfaceRatio": (0.93, 1.01, "pupil distance / pupils-to-upper-lip height"),
    "esr": (0.45, 0.47, "eye separation ratio: pupil distance / cheekbone width"),
    "chinPhiltrum": (2.0, 2.25, "mouth-to-chin height / nose-to-mouth height"),
    "jawCheek": (0.85, 0.92, "jaw width / cheekbone width"),
    "noseEye": (0.95, 1.05, "nose width / gap between the eyes"),
    "mouthNose": (1.45, 1.62, "mouth width / nose width"),
    "thirds": (0.95, 1.05, "lower third / middle third"),
    "lipRatio": (1.5, 2.0, "lower lip / upper lip height"),
}
PSL_METRICS = list(PSL_IDEALS)


def psl_metrics(landmarks: np.ndarray, width: int, height: int) -> dict[str, float]:
    """The PSL ratios from normalized MediaPipe landmarks (478 points, with irises)."""
    p = _aligned(landmarks[:, :2] * np.array([width, height], dtype=np.float64))
    dist = lambda a, b: float(np.hypot(*(p[a] - p[b])))  # noqa: E731

    face_width = dist(R_ZYGION, L_ZYGION)
    pupils = dist(R_IRIS, L_IRIS)
    pupil_y = (p[R_IRIS, 1] + p[L_IRIS, 1]) / 2
    stomion_y = (p[UPPER_LIP_INNER, 1] + p[LOWER_LIP_INNER, 1]) / 2
    nose_width = dist(R_ALA, L_ALA)
    brow_y = (p[R_BROW, 1] + p[L_BROW, 1]) / 2
    tilt_r = np.degrees(np.arctan2(p[R_EYE_INNER, 1] - p[R_EYE_OUTER, 1], p[R_EYE_INNER, 0] - p[R_EYE_OUTER, 0]))
    tilt_l = np.degrees(np.arctan2(p[L_EYE_INNER, 1] - p[L_EYE_OUTER, 1], p[L_EYE_OUTER, 0] - p[L_EYE_INNER, 0]))

    return {
        "canthalTilt": float((tilt_r + tilt_l) / 2),
        "fwhr": face_width / abs(p[UPPER_LIP, 1] - brow_y),
        "midfaceRatio": pupils / abs(p[UPPER_LIP, 1] - pupil_y),
        "esr": pupils / face_width,
        "chinPhiltrum": abs(p[CHIN, 1] - stomion_y) / abs(stomion_y - p[SUBNASALE, 1]),
        "jawCheek": dist(R_GONION, L_GONION) / face_width,
        "noseEye": nose_width / dist(R_EYE_INNER, L_EYE_INNER),
        "mouthNose": dist(R_MOUTH, L_MOUTH) / nose_width,
        "thirds": abs(p[CHIN, 1] - p[SUBNASALE, 1]) / abs(p[SUBNASALE, 1] - p[GLABELLA, 1]),
        "lipRatio": abs(p[LOWER_LIP, 1] - p[LOWER_LIP_INNER, 1]) / abs(p[UPPER_LIP_INNER, 1] - p[UPPER_LIP, 1]),
    }


def pose_values(landmarks: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Head pose proxies; twin of web/src/lib/pose.ts poseValues().

    roll  : eye-line angle in degrees
    yaw   : nose-tip offset from the cheekbone midpoint / face width
    pitch : where the nose tip sits between eye line (0) and chin (1)
    """
    px = landmarks[:, :2] * np.array([width, height], dtype=np.float64)
    eyes_r = (px[R_EYE_OUTER] + px[R_EYE_INNER]) / 2
    eyes_l = (px[L_EYE_OUTER] + px[L_EYE_INNER]) / 2
    roll = float(np.degrees(np.arctan2(eyes_l[1] - eyes_r[1], eyes_l[0] - eyes_r[0])))
    p = _aligned(px)
    face_width = float(np.hypot(*(p[R_ZYGION] - p[L_ZYGION])))
    yaw = (p[NOSE_TIP, 0] - (p[R_ZYGION, 0] + p[L_ZYGION, 0]) / 2) / face_width
    eye_y = (p[R_EYE_OUTER, 1] + p[R_EYE_INNER, 1] + p[L_EYE_OUTER, 1] + p[L_EYE_INNER, 1]) / 4
    pitch = (p[NOSE_TIP, 1] - eye_y) / (p[CHIN, 1] - eye_y)
    return {"roll": roll, "yaw": float(yaw), "pitch": float(pitch)}


# ---------------------------------------------------------------------------
# Face crop rule (shared by the teacher embeddings and the browser app)


def face_crop_box(landmarks: np.ndarray, width: int, height: int, crop: dict) -> tuple[float, float, float]:
    """Square crop (center_x, center_y, side) in pixels around the landmarks.

    `crop` holds {"scale", "offsetX", "offsetY"} measured from SCUT's own
    framing by 01_landmarks.py, so app crops look like the training photos.
    """
    xy = landmarks[:, :2] * np.array([width, height])
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    size = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) / 2 + crop["offsetX"] * size
    cy = (y0 + y1) / 2 + crop["offsetY"] * size
    return float(cx), float(cy), float(size * crop["scale"])


def load_params() -> dict:
    return json.loads(WEB_PARAMS.read_text()) if WEB_PARAMS.exists() else {}


def save_params(section: str, value: dict) -> None:
    params = load_params()
    params[section] = value
    WEB_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    WEB_PARAMS.write_text(json.dumps(params, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Display scale: every engine's raw score goes through its own rank among
# reference faces, then onto this target curve (the owner's pick, 2026-10-13):
# 10% of faces score under 1, 30% under 3, half reach 6, 25% reach 8, 8% reach
# 9 and 3% reach 9.5. Twin of web/src/lib/display.ts.

TARGET_PERCENTILES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.60, 0.75, 0.85, 0.92, 0.97, 1.0]
TARGET_SCORES = [0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0]
EXTRA_KNOTS = [0.01, 0.02, 0.05, 0.40, 0.70, 0.80, 0.90, 0.95, 0.99]


def display_knots(reference: np.ndarray) -> dict:
    """Piecewise-linear map raw -> 0-10 so `reference` lands on the target curve."""
    q = np.array(sorted(set(TARGET_PERCENTILES) | set(EXTRA_KNOTS)))
    raw = np.quantile(reference, q)
    raw = raw + np.arange(len(raw)) * 1e-9  # strictly increasing for interpolation
    return {"raw": [round(float(v), 6) for v in raw], "score": np.interp(q, TARGET_PERCENTILES, TARGET_SCORES).tolist()}


def display_score(raw, knots: dict):
    return np.interp(raw, knots["raw"], knots["score"])
