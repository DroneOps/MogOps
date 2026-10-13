# MogOps

Modelo de visión que califica moggeo en tiempo real.

A browser app that rates a face from 0.0 to 10.0, in the spirit of Omoggle and
Mogwars, with everything running on the user's device. It shows two scores:

| Score | What it is | Agreement with human ratings* |
|---|---|---|
| **MogNet** (headline) | CLIP ViT-B/16 image encoder + linear head trained on SCUT-FBP5500 ratings, run in the browser with ONNX Runtime | **r = 0.93** |
| **Geometry** (breakdown) | 8 facial ratios from MediaPipe landmarks, each scored on a curve learned from the same ratings | r = 0.59 |
| *Mogwars-style flat bands, same ratios (for reference)* | *hand-shaped "ideal band" per ratio* | *r = 0.30* |
| *ResNeXt-50 from the SCUT paper (for reference)* | *fine-tuned CNN* | *r = 0.90* |

\*Pearson correlation with the mean rating of 60 people, on held-out faces,
averaged over SCUT-FBP5500's official 5-fold split.

## Quick start

```bash
cd web
npm install        # also copies the MediaPipe / ONNX Runtime wasm into public/vendor
npm run dev        # http://localhost:5173
```

The trained models are already in `web/public/models/`, so the app runs without
touching Python. `npm test` checks that the browser's geometry and pose math
match the Python training code exactly.

## Sounds (local only)

With **Sound** on, the app loops `web/public/audio/gigachad.mp3` while the
headline score is 8.0 or higher (pausing and resuming below that, never
restarting) and plays `web/public/audio/vine-boom.mp3` when camera mode sees a
raised eyebrow (`web/src/lib/eyebrow.ts`). Both are copyrighted clips, so they
are git-ignored: add your own copies with those names, or the app stays silent.

## Repository layout

```
training/                 Python (uv) pipeline, run the numbered scripts in order
  00_download.py          SCUT-FBP5500 (git-ignored) + MediaPipe face_landmarker.task
  01_landmarks.py         MediaPipe on all 5500 faces; SCUT framing -> crop rule; pose ranges
  02_geometry.py          learn per-trait curves + weights; writes parity fixture
  03_embed.py             CLIP embeddings of the cropped faces (~6 min on an M1)
  04_teacher.py           evaluate + save ridge (browser) and SVR (offline teacher) heads
  05_export_web.py        CLIP + ridge -> int8 ONNX for the browser, with agreement check
  predict.py              score any photo from the command line
  common.py, mognet.py    shared definitions (each has a TypeScript twin in web/src/lib)
  models/                 teacher_svr.joblib (offline teacher), ridge_head.npz
web/                      Vite + React + TypeScript + Tailwind app
  src/lib/geometry.ts     ratios, learned curves, strongest/weakest trait
  src/lib/pose.ts         "face the camera" check (pauses scoring, never penalizes)
  src/lib/crop.ts         SCUT-style face crop fed to MogNet
  src/lib/mognet.ts       ONNX Runtime Web session
  src/lib/model-params.json  every learned number the app uses (written by training/)
  public/models/          face_landmarker.task, mognet-int8.onnx
HANDOFF.md                full project context: decisions, results, next steps
```

## Retraining

```bash
cd training
uv sync
uv run 00_download.py
uv run 01_landmarks.py
uv run 02_geometry.py
uv run 03_embed.py
uv run 04_teacher.py
uv run 05_export_web.py
uv run predict.py some_face.jpg
```

Each script writes its results into `web/src/lib/model-params.json` and/or
`web/public/models/`, so the app picks up the new model on the next reload.

## Deploying

`npm run build` produces a static site in `web/dist/`. For multi-threaded model
inference the host must send these headers (the dev server already does):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them the app still works, with single-threaded inference that is
slower.

## Licensing and responsible use

- **SCUT-FBP5500** is licensed for **non-commercial research only**. The
  dataset is never committed; the trained heads and `mognet-int8.onnx` are
  derived from it, so treat them as non-commercial too. Commercial use needs
  our own labeled data (see HANDOFF.md).
- **CLIP** weights: MIT (OpenAI). **MediaPipe** face model: Apache 2.0.
- The repo has no license file yet, so the team should pick one.
- Scores reflect the average taste of 60 raters looking at Asian and Caucasian
  adult faces. They are less reliable for anyone else and are not an objective
  measure of anything. Similar apps have drawn criticism for reaching minors;
  any public deployment should include an age gate.
