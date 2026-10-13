# Handoff: MogOps

*As of 2026-10-13 (first written 2026-10-09). Repo: https://github.com/DroneOps/MogOps (public), branch `main`.*

## 1. Goal & context

Recreate **Omoggle** (a viral webcam site that pairs strangers and has an AI rate
whose face "mogs", 0–10) as our own project, using beginner-friendly tech, with a
vision model that rates a face from 0.0 to 10.0. The team wants a self-contained
repo that runs on-device (in the browser), with a Mogwars-style UI and breakdown.

The owner's direction as of 2026-10-13: the app should be **funny and engaging
rather than precise**. Users should be able to game it by pulling "gigachad" /
looksmaxxer faces at the camera, and scores should be spread out generously,
not bunched in the middle.

### What we learned before building

- **Omoggle is closed source.** No public model or code exists. It is under fire
  in the press for exposing minors (age 12+) to appearance scoring.
- **`elijah629/mogged`** (https://github.com/elijah629/mogged) is a hobby project
  that copied Omoggle's and Mogwars' scoring by reading their public JS bundles.
  It has **no license**, so we studied it but **did not copy any of its code or
  its copied constants**. A local clone sits outside this repo at `../mogged`.
- **How Omoggle/Mogwars score:** no ML model. MediaPipe Face Landmarker (478
  points) → 7 ratios (canthal tilt, jaw width, symmetry, midface, cheekbones,
  eye aspect, eye spacing) → a hand-picked "ideal band" per ratio (flat 10
  inside, linear falloff outside) → hand-picked weights (symmetry 24%, harmony
  18%, …). Both sites use the **same formula**. The differences:
  - Omoggle: lenient pose check, multiplier 1.0 at confidence ≥ 0.82, floor
    3.1, top end squeezed. An average face lands at about 9.7, so everyone
    scores about the same.
  - Mogwars: 30-frame trimmed mean, plus a strict pose check where one warning
    multiplies the score by 0.55–0.78. Its wider spread mostly comes from
    **head pose, not the face**.
- **Accuracy of the face points:** MediaPipe is built for AR filters and
  publishes no accuracy figures. One independent study found about 4 mm error
  on pupil distance. It also pulls faces toward a template, which compresses
  differences between people.
- **Benchmark:** SCUT-FBP5500 (5,500 faces, each rated 1–5 by 60 people). The
  paper reports r = 0.59–0.67 for geometry features and 0.90 for a fine-tuned
  ResNeXt-50 (5-fold). Later papers report about 0.92. Raters also disagree
  with each other, so about 0.9 is close to the practical ceiling.
- **"PSL" (looksmaxxing) is not a formula.** PSL (PUAhate/Sluthate/Lookism) is a
  1–8 scale forum users assign by eye (4 = average, roughly 1 point per
  standard deviation). What forums and "face rating" apps compute are ratios with
  claimed ideal ranges (canthal tilt 4–8°, FWHR 1.8–2.0, midface 0.93–1.01, ESR
  0.45–0.47, chin/philtrum 2.0–2.25, …). These numbers are forum/clinic lore, not
  peer-reviewed, and often disagree between sources. We implemented them anyway
  (§3) and measured them: **r = −0.09** with SCUT ratings.

## 2. Current state

- **Status:** working MVP, pushed. Photo mode is verified end to end in dev and
  in a production build (about 1 s per photo). Camera mode was confirmed on the
  owner's webcam before today's changes. **Today's camera features (meme
  bonuses, new eyebrow baseline) are unit-tested but have not been tried on a
  real webcam yet.**
- **Completion:** about 85% of the vision. Not done: the small on-device student
  model, deployment, an age gate, and webcam tuning of the bonus thresholds.
- **What's working (changed 2026-10-13 unless noted)**
  - **Three engines**, chosen as the headline in the UI:
    - **MogNet:** CLIP ViT-B/16 + linear head, now trained on **SCUT + MEBeauty**.
      r = 0.925 on SCUT (5-fold) and **0.815 on MEBeauty** (vs 0.934 / 0.658 for
      the old SCUT-only head).
    - **Geometry** (learned curves, unchanged): r = 0.585 on SCUT, 0.29 on MEBeauty.
    - **PSL ratios** (new): looksmaxxing ideals, r = −0.09. In the app as a meme,
      with each ratio shown next to its "ideal" range.
  - **Target display curve** (owner's pick): every engine is ranked against
    about 8,000 reference faces (SCUT + MEBeauty) and mapped so 10% of faces
    score < 1, 20% < 2, 30% < 3, 50% ≥ 6, 40% ≥ 7, 25% ≥ 8, 15% ≥ 8.5, 8% ≥ 9 and
    3% ≥ 9.5. All three engines match this distribution on the reference faces
    by construction.
  - The headline also shows **PSL x.x · tier** (Low/Mid/High-tier normie,
    Chadlite, Chad, Gigachad) and "Top N% of faces". It is derived from the
    headline's rank (PSL = 4 + z-score).
  - **Meme mode** (toggle, on by default): bonus points from MediaPipe
    blendshapes. The Rock (one eyebrow, +1.0), Hunter eyes (squint, +0.75),
    Hollow cheeks (pucker, +0.75), Mewing (lips pressed / jaw forward, +0.5),
    Smiling (−0.75), Puffed cheeks (−0.5). Each bonus has a live meter. They are
    added to the headline, so they can trigger the Giga Chad theme.
  - The **eyebrow-boom baseline is recalculated every frame** (25th percentile
    of the last 90 frames), so people can swap in front of the camera without a
    reload. It is tested with a simulated person swap.
  - **MogNet head moved out of the ONNX** into `model-params.json`
    (`mognet.head`: 512 weights + bias on the L2-normalized embedding). The
    ONNX is now a pure CLIP encoder, so retraining the head never needs a new
    89 MB file.
  - **Weight-only int8 quantization** (new): the int8 encoder agrees with fp32
    at r = 0.9998 (it was 0.987 with onnxruntime's `quantize_dynamic`).
  - From before: Python↔TS parity tests (now geometry, pose and PSL), pose
    gating that pauses instead of penalizing, the face-point toggle, Giga Chad
    at ≥ 8.0 with pause and resume, and the vine boom on a raised eyebrow.
- **What's broken / incomplete**
  - **Bonus thresholds are untested on deliberate faces.** They were tuned only
    so that *ordinary* photos earn nothing: across 600 SCUT and 600 MEBeauty
    faces, positive bonuses average +0.03 and about 5% of faces get > +0.25. If a
    real squint or pucker can't reach the `from`/`to` range, lower it
    (`web/src/lib/expressions.ts`).
  - **The middle of the display curve is steep.** 3 → 6 covers only 20% of
    faces, so in that band one raw unit is worth about 7 display points. Small
    input differences (lighting, crop, browser vs Python landmarks) move scores
    there by up to about 0.5. Example: one SCUT test face gives 4.1 in the
    browser and 4.6 in Python fp32. This is a direct consequence of the chosen
    curve. Camera mode's EMA smoothing (0.6/0.4) softens it.
  - The 89 MB encoder is committed to git. History now holds two versions (the
    old `mognet-int8.onnx` and the new `mognet-encoder-int8.onnx`, about 180 MB
    total). With the head out of the ONNX, this should rarely change again.
  - No license file, no age gate, not deployed.
- **Blockers / decisions for the team**
  - **Licensing:** SCUT-FBP5500 is non-commercial research only. **MEBeauty has
    no license file at all** (default: all rights reserved; filenames suggest
    Pixabay photos). The head in `model-params.json` is derived from both.
    Commercial use needs our own labels.
  - The repo is **public** and includes those derived weights.
  - Pick a repo license.

## 3. Architecture & key files

Two halves share one parameter file. The **Python pipeline** (`training/`,
numbered scripts) learns every number and writes it to
`web/src/lib/model-params.json` and `web/public/models/`. The **web app** (`web/`,
Vite + React 19 + TypeScript 7 + Tailwind 4) is a static site that runs
MediaPipe and ONNX Runtime Web entirely in the browser. Every Python geometry,
PSL, pose, crop or display function has a TypeScript twin, and the parity tests
enforce that they match.

```
Per frame / photo:
  MediaPipe (478 landmarks + 52 blendshapes)
    ├─ pose.ts checkPose ──── bad pose → pause + message
    ├─ geometry.ts computeMetrics → curves → raw → display curve   (camera: 30-frame trimmed mean)
    ├─ psl.ts pslMetrics → ideal-band scores → mean → display curve (camera: 30-frame trimmed mean)
    ├─ expressions.ts scoreBonuses(blendshapes)                   (camera: EMA-smoothed)
    ├─ eyebrow.ts EyebrowDetector → vine boom                      (camera only)
    └─ crop.ts → 224×224 → mognet.ts int8 CLIP encoder → w·e + b → calibration → display curve
                                                                   (camera: every 1.5 s, EMA)
  headline.ts: headline = engine score (+ bonus total in meme mode), clamped 0–10
  display.ts:  PSL = 4 + probit(rank); tier label
  App.tsx:     Giga Chad loop while headline ≥ 8.0 (sounds.ts)
```

```
training/                      Python 3.12, uv. Run from training/: uv run <script>
  common.py                    paths, official splits, landmark indices,
                               geometry_metrics (:117), PSL_IDEALS (:184) + psl_metrics (:199),
                               pose_values (:227), face_crop_box (:250),
                               TARGET_PERCENTILES/SCORES + display_knots (:281), load/save_params
  mognet.py                    crop_face, ClipEmbedder (flip-averaged CLIP), MogNetEncoder (+ L2 norm)
  00_download.py               SCUT (Google Drive via gdown) + face_landmarker.task (certifi SSL)
  01_landmarks.py              MediaPipe on 5500 faces → artifacts/landmarks.npz; "crop" + "pose" params
  02_geometry.py               learned curves, symmetry display-only, display knots on SCUT+MEBeauty,
                               parity fixture __fixtures__/geometry.json
  03_embed.py                  CLIP embeddings of MediaPipe crops → features/clip_b16.npz
  04_teacher.py                SCUT-only ridge/SVR evaluation + offline SVR teacher (--skip-svr keeps it);
                               browser head on SCUT+MEBeauty (z-scored labels), out-of-fold display knots
  05_export_web.py             torch → ONNX fp32 → weight-only int8 (quantize_weights :52) → calibration
  06_mebeauty_prep.py          MEBeauty landmarks + CLIP embeddings (clone instructions in docstring)
  07_psl.py                    PSL formula: tolerances, report (artifacts/psl_report.json), params, fixture
  08_mebeauty_eval.py          SCUT-only vs MEBeauty-only vs both, by ethnic group → artifacts/mebeauty_report.json
  predict.py                   CLI: MogNet / geometry / PSL like the app, plus the SVR teacher's raw 1–5
  data/ features/ artifacts/*.npz models/*.onnx   (git-ignored)

web/src/
  App.tsx                      layout, stored toggles (mogops.showPoints / .sound / .meme), music + boom wiring
  components/InputCard.tsx     image/camera, engine select (MogNet / Geometry / PSL ratios), toggles, overlay
  components/ScoreCard.tsx     headline + PSL tier + top-N%, engine row, bonus meters, geometry or PSL breakdown
  lib/display.ts               interp, displayScore, percentileOf, pslFromScore, pslTier
  lib/headline.ts              Engine type, headlineScore (engine + bonuses)
  lib/expressions.ts           BONUSES (thresholds at top), BlendshapeSmoother, withBonus
  lib/psl.ts                   pslMetrics, pslSubScores, pslScore, labels
  lib/geometry.ts              computeMetrics (:94), curves, TRAITS, strongestAndWeakest (:188)
  lib/eyebrow.ts               EyebrowDetector (rolling-percentile baseline; RAISE_RATIO :17)
  lib/mognet.ts                ORT session (loaded from public/vendor/ort), head, calibration, display
  lib/landmarker.ts            MediaPipe with blendshapes on; returns {landmarks, blendshapes}
  lib/useFaceAnalysis.ts       the pipeline loop (photo + camera); eyebrow check :177
  lib/sounds.ts                MOG_THRESHOLD = 8.0, Giga Chad loop (pause/resume) + vine boom
  lib/rolling.ts               generic MetricWindow (trimmed mean)
  lib/model-params.json        ALL learned numbers: crop, pose, geometry, psl, mognet (head, display, calibration)
  lib/*.test.ts                parity (geometry, pose, PSL), eyebrow, bonuses — 16 tests
web/public/models/             face_landmarker.task (3.7 MB), mognet-encoder-int8.onnx (89 MB)
web/public/audio/              gigachad.mp3, vine-boom.mp3 — copyrighted, git-ignored
```

**External dependencies:** none at runtime; everything is served from our own
origin. For training: Google Drive (SCUT), GitHub (MEBeauty, cloned by hand),
Google Storage (MediaPipe model), Hugging Face (CLIP weights via open_clip).

**Development setup**
- Web: Node 24, `cd web && npm install && npm run dev` → http://localhost:5173.
  `npm test`, `npm run build`, `npm run preview` (port 4173).
- Training: `uv` installed; `cd training && uv sync`. Python 3.12 is pinned
  because mediapipe has no 3.14 wheels. About 1.2 GB venv and about 0.6 GB CLIP
  weights in `~/.cache/huggingface`. MEBeauty: about 250 MB sparse clone in
  `training/data/mebeauty/repo`.
- Pushing: `git config http.postBuffer 524288000` was needed in this clone.

### Results (Pearson r with mean human rating, held-out faces)

| Model | SCUT 5-fold | SCUT 60/40 | MEBeauty test |
|---|---|---|---|
| CLIP + SVR, SCUT only (offline teacher) | **0.941** | 0.938 | – |
| CLIP + ridge, SCUT only (old browser head) | 0.934 | 0.930 | 0.658 |
| CLIP + ridge, MEBeauty only | 0.778 | – | 0.831 |
| **CLIP + ridge, SCUT + MEBeauty (browser head)** | **0.925** | – | **0.815** |
| CLIP + DINOv2-base concat, ridge | 0.941 | – | – |
| Paper: ResNeXt-50 fine-tuned | 0.900 | 0.878 | – |
| Learned-curve geometry | 0.585 | 0.591 | 0.286 (0.328 refit on MEBeauty) |
| Mogwars-style flat bands | 0.295 | 0.295 | – |
| PSL ratios, community ideals, equal weights | −0.093 | −0.117 | 0.104 |
| Same PSL band scores, NNLS-fitted weights | 0.305 | – | – |

MEBeauty test r by group (SCUT-only → both): female Black 0.29 → 0.63,
male Asian 0.50 → 0.64, male Black 0.59 → 0.73, female Hispanic 0.64 → 0.76,
male Caucasian 0.71 → 0.78, female Mideastern 0.74 → 0.86. Every group improves.
Groups have 24–109 test faces, so treat single numbers loosely.

PSL ratios on SCUT, measured with MediaPipe points (median; share inside the ideal; r of sub-score with rating):
canthal tilt 7.6° (45%, −0.13), FWHR 1.62 (6%, −0.31), midface 0.99 (43%, +0.17),
ESR 0.45 (29%, +0.16), chin/philtrum 2.01 (35%, +0.10), jaw/cheek 0.79 (0.5%, −0.25),
nose/eye gap 1.06 (40%, +0.03), mouth/nose 1.38 (24%, +0.16), lower/middle third 0.92 (27%, −0.25),
lip ratio 1.70 (52%, +0.05). Taken literally, the formula bunches faces at
6.6 ± 0.85 (humans: 5.0 ± 1.7). That is the same "everyone scores about the
same" problem as Omoggle. MediaPipe's jaw and cheek points are not real gonion
or zygion, so almost nobody falls inside the jaw ideal.

Leakage checks (from 2026-10-09) passed: no fold overlap, shuffled labels give
r = −0.07, and dropping near-duplicate test faces leaves r unchanged.

## 4. What we tried and what failed

❌ **Mogwars-style flat "ideal bands" (even with data-picked edges)** → r = 0.295.
   **Replaced by** per-trait learned curves + NNLS weights (r = 0.585).

❌ **The looksmaxxing PSL formula as an actual rater** → r = −0.09 (SCUT), +0.10
   (MEBeauty). Even with fitted weights it only reaches 0.305. Kept as an
   explicitly meme engine. Don't use it as a rating signal.

❌ **SCUT-only MogNet on diverse faces** → r drops from 0.93 to 0.66 on
   MEBeauty (0.29 for Black women). **Fixed** by training on both datasets with
   each dataset's labels z-scored (they use 1–5 vs 1–10 scales and different raters).

❌ **onnxruntime `quantize_dynamic` (int8 activations)** → head output noise of
   0.14 sd per face. On the new steep display curve that is up to 2.5 points
   (mean 0.44). **Replaced by** weight-only int8 (`quantize_weights` in
   `05_export_web.py`: int8 weights per output channel + DequantizeLinear;
   activations stay fp32). Same 89 MB file and speed, error 0.015 sd (mean
   0.04 display points, max 0.32). Runs in onnxruntime-web 1.30.

❌ **Symmetry as a scored trait** → it correlates the wrong way (+0.23). It is
   display-only.

❌ **Blendshape bonus thresholds picked by intuition** (squint 0.25–0.6, pucker
   0.3–0.75) → 86% of *neutral* SCUT faces got the hunter-eyes bonus.
   MediaPipe reads a resting face as about 0.4 "squint", and "mouthPucker" has a
   long tail (p95 about 0.8 on neutral faces). **Fixed** by setting thresholds
   from the measured distributions (squint 0.7–0.9, pucker 0.7–0.95 and only
   without a smile, brow asymmetry 0.3–0.65). `jawForward` is essentially never
   above 0.01 on photos, so "Mewing" mostly relies on pressed lips.

❌ **Eyebrow baseline that only learned from resting faces** → if a person with
   naturally higher brows stepped in, they read as "raised" forever and the
   baseline never updated. **Fixed** with a rolling 25th-percentile baseline over
   the last 3 s, always updated.

❌ **Separate male/female geometry calibration** → +0.007 r. Not worth it.

❌ **CLIP + DINOv2 for the browser** → +0.005 r for twice the size.

❌ **`mp.Image.create_from_file`** (mediapipe 1.1.0) crashes → decode with Pillow.

❌ **urllib download on python.org Python 3.12** → SSL error; use certifi.

❌ **onnxruntime-web import variants** (default import, folder `wasmPaths`, bundling)
   all broke. **Current fix:** load `public/vendor/ort/ort.wasm.bundle.min.mjs` by
   full URL with `@vite-ignore` (`mognet.ts`).

⚠️ **`teacher_svr.joblib` unpickling needs `l2_normalize` on `__main__`**: it was
   pickled while 04_teacher.py ran as a script. `predict.py` re-exports it at top
   level. Refit the SVR from an importable module if this keeps biting.

⚠️ **Audio autoplay:** browsers block `play()` until the user clicks on the page.

⚠️ **The dev server dies with the session that started it.** Run it from your own
   terminal (`npm --prefix web run dev`).

⚠️ **The dev machine has very little disk** (0.7–3 GB free during this work). The
   fp32 export (345 MB) is deleted after each run; `npm run build` copies the
   89 MB model into `web/dist/`.

⚠️ **The camera can't be tested from the agent's built-in browser** (camera is
   blocked there). Photo mode works.

## 5. Next steps (priority order)

1. **[HIGH] Tune meme bonuses on a real webcam.** Pull each face and watch its
   meter. Adjust `from`/`to` in `web/src/lib/expressions.ts`. Keep resting faces
   near 0; the measured resting distributions are in §4 and in the file's
   comments. Consider adding a geometric hollow-cheek cue (cheek contour width vs
   zygion width) if pucker alone feels wrong.
2. **[HIGH] Tune the eyebrow boom on a real webcam** (`RAISE_RATIO` 1.15,
   `HOLD_FRAMES` 3, `BASELINE_FRAMES` 90, `BASELINE_PERCENTILE` 0.25 at the top of
   `eyebrow.ts`). Also check that two people swapping works as the test simulates.
3. **[HIGH] Check the feel of the new display curve with the owner.** If the
   middle is too jumpy, soften it (e.g. 30% < 3.5, 50% ≥ 5.5) in
   `TARGET_PERCENTILES/SCORES` in **both** `training/common.py` and
   `web/src/lib/display.ts`. Then rerun `02_geometry.py`, `07_psl.py` and
   `04_teacher.py --skip-svr` to refit the knots.
4. **[HIGH] Distill a small on-device student:** label SCUT + MEBeauty + extra
   unlabeled faces with the teacher (consider refitting the SVR on both
   datasets first), then train MobileNetV3/EfficientNet-Lite at 2–6 MB.
   This replaces the 89 MB encoder.
5. **[HIGH] Team decisions:** repo visibility given the derived head weights
   (SCUT + MEBeauty), repo license, and an age gate before any public deploy.
6. **[MEDIUM] Deploy** to a static host with COOP/COEP headers. The audio isn't in
   git, so a deploy needs licensed sounds.
7. **[MEDIUM] Own labels for commercial use:** in-app pairwise "who mogs?" votes →
   Bradley–Terry scores.
8. **[LOW] Head-to-head mode** (Omoggle-style battle UI); show the face crop the model sees.
9. **[LOW] Cleanup:** the Python `faces()` in `05_export_web.py` reloads landmarks
   per call; `../teacher` and `../mogged` can be deleted to free disk.

## 6. Quick reference

- **Run app:** `npm --prefix web run dev`, then `open -a "Google Chrome" http://localhost:5173`.
  Tests: `npm --prefix web test` (16). Build: `npm --prefix web run build`.
- **Toggles (stored in localStorage):** Meme mode (`mogops.meme`, default on),
  Face points (`mogops.showPoints`, off), Sound (`mogops.sound`, on).
- **Retrain order:** `00 → 01 → 06 → 02 → 07 → 03 → 04 → (05 only if the encoder or crop changed) → 08 (report)`.
  A head-only retrain is `uv run 04_teacher.py --skip-svr` (about 1 min); no ONNX change is needed.
- **Score photos from the CLI:** `cd training && uv run predict.py face.jpg`.
- **Test a face in the browser without a file dialog:** put a JPEG in
  `web/public/vendor/` (git-ignored), then from the console build a `File` with
  `DataTransfer`, assign it to the file input, and dispatch `change`. Wait for the
  status "Done" (don't regex for "MogNet \d": the footer matches it).
- **Score scale:** all engines are mapped through `display_knots` (raw → rank
  among SCUT + MEBeauty faces → target curve). MogNet's raw output is in
  z-units of the pooled ratings.
- **Sources:** SCUT-FBP5500 https://github.com/HCIILAB/SCUT-FBP5500-Database-Release ·
  MEBeauty https://github.com/fbplab/MEBeauty-database (paper: Lebedeva et al., Neural
  Computing and Applications 2021) · mogged https://github.com/elijah629/mogged ·
  MediaPipe Face Landmarker https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker ·
  PSL ratio lore: looksmax.org threads ("Ideal facial ratios and proportions"), perfectcorp /
  brainbound blog posts on chin/philtrum; neoclassical canons (nose = intercanthal width,
  mouth = 1.5 × nose).
- **Secrets:** none are needed. GitHub pushes use the developer's own `gh` login.
