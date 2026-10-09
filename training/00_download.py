"""Download the SCUT-FBP5500 dataset and the MediaPipe Face Landmarker model.

Usage: uv run 00_download.py

- SCUT-FBP5500 (~180 MB) comes from the authors' official Google Drive link
  (https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). It is for
  non-commercial research use only, so it is never committed to git.
- face_landmarker.task (~3.6 MB) is Google's MediaPipe model. It lives in
  web/public/models so the browser app and the training scripts share it.
"""

import shutil
import ssl
import urllib.request
import zipfile

import certifi
import gdown

from common import DATA, LANDMARKER_TASK

SCUT_DRIVE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def main() -> None:
    if DATA.exists():
        print(f"dataset already present: {DATA}")
    else:
        DATA.parent.mkdir(parents=True, exist_ok=True)
        archive = DATA.parent / "scut.zip"
        gdown.download(id=SCUT_DRIVE_ID, output=str(archive))
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA.parent)
        archive.unlink()
        print(f"dataset extracted to {DATA}")

    if LANDMARKER_TASK.exists():
        print(f"landmarker already present: {LANDMARKER_TASK}")
    else:
        LANDMARKER_TASK.parent.mkdir(parents=True, exist_ok=True)
        # certifi's CA bundle avoids SSL errors on python.org builds of Python.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(LANDMARKER_URL, context=context) as r, open(LANDMARKER_TASK, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"landmarker saved to {LANDMARKER_TASK}")


if __name__ == "__main__":
    main()
