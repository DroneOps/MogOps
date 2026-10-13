# MogOps

Modelo de visión que califica moggeo en tiempo real.

A browser app that rates a face from 0.0 to 10.0, in the spirit of Omoggle and
Mogwars, with everything running on the user's device. It has three engines
(pick the headline in the UI):

| Score | What it is | r on SCUT* | r on MEBeauty* |
|---|---|---|---|
| **MogNet** (default) | CLIP ViT-B/16 image encoder + linear head trained on SCUT-FBP5500 **and** MEBeauty ratings, run in the browser with ONNX Runtime | **0.92** | **0.81** |
| **Geometry** | 8 facial ratios from MediaPipe landmarks, each scored on a curve learned from SCUT ratings | 0.59 | 0.29 |
| **PSL ratios** | the looksmaxxing forums' 10 "ideal" ratios (canthal tilt, FWHR, ESR, …), scored like their face-rating tools | −0.09 | 0.10 |
| *Mogwars-style flat bands (reference)* | *hand-shaped "ideal band" per ratio* | *0.30* | – |
| *ResNeXt-50 from the SCUT paper (reference)* | *fine-tuned CNN* | *0.90* | – |

\*Pearson correlation with the mean human rating on held-out faces: SCUT's
official 5-fold split; MEBeauty's official test split (2,550 in-the-wild
photos, six ethnic groups). The PSL formula has no real link to how people
rate faces; it is in the app because it is fun.

**Display scale.** Every engine's raw output is ranked against ~8,000
reference faces (SCUT + MEBeauty) and mapped onto one target curve: 10% of
faces score under 1, 30% under 3, half reach 6, 25% reach 8, 8% reach 9 and
3% reach 9.5. The big number also shows a **PSL** rating (4 = median face,
~1 point per standard deviation) and tier (normie … Chadlite, Chad, Gigachad).

**Meme mode** (on by default) adds deliberately gameable points from
MediaPipe's expression blendshapes: one raised eyebrow (+1.0), hunter-eye
squint (+0.75), hollow cheeks (+0.75), mewing (+0.5); smiling (−0.75) and
puffed cheeks (−0.5) cost points. Ordinary photos earn about +0.03.

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
headline score (bonuses included) is 8.0 or higher (pausing and resuming below that, never
restarting) and plays `web/public/audio/vine-boom.mp3` when camera mode sees a
raised eyebrow (`web/src/lib/eyebrow.ts`; the resting brow height is
re-learned continuously, so people can take turns without a reload). Both are copyrighted clips, so they
are git-ignored: add your own copies with those names, or the app stays silent.

## Repository layout

```
training/                 Python (uv) pipeline, run the numbered scripts in order
  00_download.py          SCUT-FBP5500 (git-ignored) + MediaPipe face_landmarker.task
  01_landmarks.py         MediaPipe on all 5500 faces; SCUT framing -> crop rule; pose ranges
  02_geometry.py          learn per-trait curves + weights; writes parity fixture
  03_embed.py             CLIP embeddings of the cropped faces (~6 min on an M1)
  04_teacher.py           evaluate + save ridge (browser, SCUT+MEBeauty) and SVR (offline teacher) heads
  05_export_web.py        CLIP encoder -> weight-only int8 ONNX for the browser, with agreement check
  06_mebeauty_prep.py     MEBeauty landmarks + CLIP embeddings (dataset cloned by hand, see its docstring)
  07_psl.py               the looksmaxxing PSL-ratio formula: params + how it holds up
  08_mebeauty_eval.py     does MEBeauty help? cross-dataset report
  predict.py              score any photo from the command line
  common.py, mognet.py    shared definitions (each has a TypeScript twin in web/src/lib)
  models/                 teacher_svr.joblib (offline teacher), ridge_head.npz
web/                      Vite + React + TypeScript + Tailwind app
  src/lib/geometry.ts     ratios, learned curves, strongest/weakest trait
  src/lib/pose.ts         "face the camera" check (pauses scoring, never penalizes)
  src/lib/crop.ts         SCUT-style face crop fed to MogNet
  src/lib/mognet.ts       ONNX Runtime Web session + head
  src/lib/psl.ts          PSL ratios with their "ideal" ranges
  src/lib/display.ts      target display curve, PSL scale and tiers
  src/lib/expressions.ts  meme-mode bonuses (blendshapes)
  src/lib/eyebrow.ts      raised-eyebrow trigger for the vine boom
  src/lib/model-params.json  every learned number the app uses (written by training/)
  public/models/          face_landmarker.task, mognet-encoder-int8.onnx
HANDOFF.md                full project context: decisions, results, next steps
```

## Retraining

```bash
cd training
uv sync
uv run 00_download.py
uv run 01_landmarks.py
# MEBeauty (no download script: clone it, see 06_mebeauty_prep.py)
uv run 06_mebeauty_prep.py
uv run 02_geometry.py
uv run 07_psl.py
uv run 03_embed.py
uv run 04_teacher.py          # --skip-svr to keep the saved offline teacher
uv run 05_export_web.py       # only when the encoder or crop changes
uv run 08_mebeauty_eval.py    # optional report
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

- **SCUT-FBP5500** is licensed for **non-commercial research only**.
  **MEBeauty** (https://github.com/fbplab/MEBeauty-database) has no license
  file at all (default: all rights reserved; many photos look like Pixabay
  originals). Neither dataset is ever committed; the trained heads in
  `model-params.json` are derived from both, so treat them as non-commercial
  research too. Commercial use needs our own labeled data (see HANDOFF.md).
  The ONNX encoder itself is plain CLIP and is not derived from them.
- **CLIP** weights: MIT (OpenAI). **MediaPipe** face model: Apache 2.0.
- The repo has no license file yet, so the team should pick one.
- Scores reflect the average taste of SCUT's 60 raters and MEBeauty's ~300,
  looking at adult faces. They are less reliable for faces unlike those and are
  not an objective measure of anything. Similar apps have drawn criticism for reaching minors;
  any public deployment should include an age gate.
