# Handoff: MogOps

*As of 2026-10-13 (first written 2026-10-09). Repo: https://github.com/DroneOps/MogOps (public), branch `main`.*

## 1. Goal & context

Recreate **Omoggle** (a viral webcam site that pairs strangers and has an AI rate
whose face "mogs", 0–10) as our own project, using beginner-friendly tech, with a
vision model that rates a face from 0.0 to 10.0. The team wants a self-contained
repo that runs on-device (in the browser), with a Mogwars-style UI and breakdown.

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

## 2. Current state

- **Status:** working MVP, pushed, in active use by the project owner. Photo mode is
  verified end to end in dev and production builds. Camera mode works on the
  owner's webcam (their feedback: "nice", but **"the scores are too centric"**;
  see Next steps #1). The eyebrow-raise trigger has not been confirmed on a real
  webcam yet.
- **Completion:** about 80% of the vision. Both scoring engines, the UI, sounds,
  training pipeline and tests are done. Not done: score spread, the small
  on-device student model, deployment, and an age gate.
- **What's working**
  - Learned score "MogNet" in the browser: **r = 0.934** with human ratings
    (5-fold), about 1.1 s per photo with 4 WASM threads on an M1.
  - Geometry breakdown with learned per-trait curves: **r = 0.585** (vs 0.295
    for the Mogwars-style flat bands on the same ratios).
  - The browser and Python agree exactly: `npm test` checks metrics, sub-scores,
    overall and pose to 6 decimal places. For one test face, the browser gave
    MogNet 5.1 / geometry 6.0 and Python gave 5.08 / 6.03.
  - Pose problems **pause** scoring with a message ("Face the camera straight
    on") instead of shrinking the score like Mogwars does.
  - Toggleable face-point overlay (off by default; tiny, about 28% opacity when on).
  - Sounds (added 2026-10-13): Giga Chad theme while headline ≥ 8.0
    (`MOG_THRESHOLD` in `web/src/lib/sounds.ts`; pause and resume verified), and a vine boom
    on a raised eyebrow (`web/src/lib/eyebrow.ts`: brow height above the eye
    corners / eye distance vs. a per-user adapting baseline, fires at +15% held
    3 frames, re-arms below +6%, 1.2 s cooldown; unit-tested on synthetic faces,
    **not yet tuned on a real webcam**). MP3s are copyrighted, git-ignored in
    `web/public/audio/`.
- **What's broken / incomplete**
  - **Scores bunch in the middle.** Measured 2026-10-13 on held-out SCUT faces:
    human ratings on 0–10 have mean 4.98, sd 1.72, 5th–95th percentile 2.5–8.0,
    and only 5.8% are ≥ 8.0. MogNet's held-out predictions are nearly as spread
    (sd 1.60, 4.0% ≥ 8.0). So this is mostly **how the people rating the faces
    used the scale**, not the model shrinking toward the mean. Webcam
    frames (lighting, angle) probably push live scores further toward the middle.
    Result: the 8.0 Giga Chad threshold is rarely reached.
  - The int8 model drifts from full precision by an average of 0.076 on the
    1–5 scale (about 0.19 on 0–10) after calibration. Individual faces can be
    off by 0.4.
  - The 89 MB model is committed directly to git (GitHub warns above 50 MB; the
    hard limit is 100 MB). Every retrain adds 89 MB to repo history.
  - No license file, no age gate, not deployed.
- **Blockers / decisions for the team**
  - **Licensing:** SCUT-FBP5500 is non-commercial research only, and the trained
    heads and ONNX are derived from it. Commercial use needs our own labeled data.
  - The repo is **public** and includes those derived weights. Decide whether
    that's acceptable or the repo should be private.
  - Pick a repo license.

## 3. Architecture & key files

Two halves share one parameter file. The **Python pipeline** (`training/`,
numbered scripts) learns every number and writes it to
`web/src/lib/model-params.json` and `web/public/models/`. The **web app** (`web/`,
Vite + React 19 + TypeScript 7 + Tailwind 4) is a static site that runs
MediaPipe and ONNX Runtime Web entirely in the browser. Every Python geometry,
pose or crop function has a TypeScript twin, and the parity test enforces that
they match.

```
Per frame / photo:
  MediaPipe landmarks (478 pts)
    ├─ pose.ts checkPose ──── bad pose → pause + message
    ├─ geometry.ts computeMetrics → subScores (curves) → geometryScore
    │                                (camera: rolling.ts 30-frame trimmed mean)
    └─ crop.ts faceCropBox → cropToTensor 224×224
         → mognet.ts: int8 ONNX (CLIP ViT-B/16 + flip-average + ridge head)
         → calibration (scale·x + offset) → (r−1)/4·10   (camera: every 1.5 s, smoothed)
```

```
training/                      Python 3.12, uv. Run from training/: uv run <script>
  common.py                    paths, official splits, landmark indices,
                               geometry_metrics (:117), pose_values (:171),
                               face_crop_box (:194), load/save_params
  mognet.py                    crop_face, ClipEmbedder (flip-averaged CLIP), MogNet (encoder+head)
  00_download.py               SCUT (Google Drive via gdown) + face_landmarker.task (certifi SSL)
  01_landmarks.py              MediaPipe on 5500 faces → artifacts/landmarks.npz;
                               save_framing (:57) writes "crop" + "pose" params;
                               --framing-only redoes just that from saved landmarks
  02_geometry.py               learned curves (fit_curve :41, calibrate :60),
                               DISPLAY_ONLY symmetry (:37), flat-band baseline,
                               writes "geometry" params + web parity fixture
  03_embed.py                  CLIP embeddings of MediaPipe crops → features/clip_b16.npz
  04_teacher.py                ridge + SVR heads on official splits; saves
                               models/teacher_svr.joblib (offline teacher, 18 MB)
                               models/ridge_head.npz (browser head); "mognet" params
  05_export_web.py             torch → ONNX fp32 → int8 (per-channel, KEEP_FP32 :31)
                               → linear calibration on train faces → check on test faces
  predict.py                   CLI: score any photo (SVR teacher + geometry)
  data/ features/ artifacts/*.npz models/*.onnx   (git-ignored)

web/
  scripts/copy-wasm.mjs        postinstall: copies MediaPipe wasm + ORT bundle/wasm to public/vendor (git-ignored)
  vite.config.ts               COOP/COEP headers (needed for ORT threads), base "./"
  src/App.tsx                  layout, stored toggles (localStorage "mogops.showPoints"/"mogops.sound"), music + boom wiring
  src/components/InputCard.tsx image/camera toggle, headline engine select, Face points / Sound toggles,
                               preview + overlay canvas (points off by default), "Eyebrow raise" badge
  src/components/ScoreCard.tsx big score, MogNet/geometry, strongest/weakest, 8 trait bars, accuracy footer
  src/lib/useFaceAnalysis.ts   the pipeline loop (photo + camera), throttling, smoothing; calls onEyebrowRaise
  src/lib/eyebrow.ts           EyebrowDetector (raised-brow trigger, constants at top) + eyebrow.test.ts
  src/lib/sounds.ts            MOG_THRESHOLD = 8.0, Giga Chad loop (pause/resume) + vine boom
  public/audio/                gigachad.mp3, vine-boom.mp3 — copyrighted, git-ignored, see its README.md
  src/lib/geometry.ts          computeMetrics (:93), curves, TRAITS labels, strongestAndWeakest (:200)
  src/lib/pose.ts / crop.ts / landmarker.ts / rolling.ts / mognet.ts
  src/lib/model-params.json    ALL learned numbers (crop, pose, geometry, mognet+calibration)
  src/lib/geometry.test.ts     Python↔TS parity (fixture: synthetic faces, never real people)
  public/models/               face_landmarker.task (3.7 MB, Apache-2.0), mognet-int8.onnx (89 MB)
```

**External dependencies:** none at runtime; everything is served from our own
origin. For training: Google Drive (SCUT download), Google Storage (MediaPipe
model), Hugging Face (CLIP weights via open_clip).

**Development setup**
- Web: Node 24, `cd web && npm install && npm run dev` → http://localhost:5173.
  `npm test`, `npm run build`, `npm run preview` (port 4173).
- Training: `uv` installed; `cd training && uv sync`. Python 3.12 is pinned
  because mediapipe has no 3.14 wheels. About 1.2 GB venv and about 0.6 GB CLIP
  weights in `~/.cache/huggingface`.
- Pushing: `git config http.postBuffer 524288000` was needed in this clone
  (the default gives HTTP 400 on larger pushes).

### Results table (official SCUT-FBP5500 splits, Pearson r with mean human rating)

| Model | 5-fold | 60/40 | MAE (1–5) |
|---|---|---|---|
| CLIP ViT-B/16 + SVR (offline teacher, MediaPipe crop) | **0.941** | 0.938 | 0.179 |
| CLIP ViT-B/16 + ridge (browser MogNet, fp32) | **0.934** | 0.930 | 0.191 |
| CLIP + DINOv2-base concat, ridge (original crops) | 0.941 | – | – |
| DINOv2-base + ridge (original crops) | 0.900 | 0.896 | 0.230 |
| Paper: ResNeXt-50 fine-tuned | 0.900 | 0.878 | 0.229 |
| Learned-curve geometry (ours, 7 scored ratios) | 0.585 | 0.591 | – |
| Unconstrained ridge on the 8 ratios (reference) | 0.594 | 0.599 | – |
| Mogwars-style flat bands, data-picked edges | 0.295 | 0.295 | – |

Per-subset r for CLIP+SVR (5-fold, original crops): AF 0.939, AM 0.951, CF 0.936, CM 0.931.
Leakage checks passed: train/test folds don't overlap, shuffled labels give
r = −0.07, and dropping the 471 test faces that are near-duplicates of training
faces leaves r unchanged (0.937 vs 0.936). SCUT does contain near-duplicate
photos (cosine > 0.99).

Learned geometry weights: eye spacing 0.39, jaw ratio 0.33, fWHR 0.14, canthal
tilt 0.09, lower third 0.03, eye aspect 0.01, midface 0.00, symmetry 0 (display only).

## 4. What we tried and what failed

❌ **Mogwars-style flat "ideal bands" (even with data-picked edges)** → r = 0.295.
   Most ratios relate to ratings in one direction (e.g. narrower jaw rated
   higher), not with a sweet spot. **Replaced by** per-trait curves: mean rating
   in 10 quantile bins, lightly smoothed, then non-negative least squares
   weights. r = 0.585, the same as an unconstrained linear model.

❌ **Symmetry as a scored trait** → it correlates **+0.23 the wrong way** (more
   asymmetric faces rated higher), which is almost certainly a data or
   mesh-smoothing quirk. Rewarding asymmetry would look broken, so it's
   display-only: a fixed line (10 at the 5th percentile of asymmetry, 0 at the
   95th) with zero weight. `02_geometry.py:37`.

❌ **Separate male/female geometry calibration** → only +0.007 r. Not worth a
   gender toggle in the UI.

❌ **CLIP + DINOv2 for the browser** → +0.005 r for twice the size and compute.
   CLIP only. The combination is still an option for an offline teacher.

❌ **`mp.Image.create_from_file`** (mediapipe 1.1.0) → crashes with
   `std::length_error`. Decode with Pillow and wrap with `mp.Image(...)`
   (`01_landmarks.py`).

❌ **urllib download on python.org Python 3.12** → SSL certificate error. Use
   certifi's CA bundle (`00_download.py`).

❌ **Plain int8 dynamic quantization** → constant offset of −0.095 (1–5 scale)
   vs fp32. Per-channel quantization fails on the 1-D ridge head ("b zero point
   is not valid"). **Fix:** per-channel, with the head and CLIP projection kept
   fp32, plus a linear calibration fitted on 200 training faces. The gap drops
   from 0.105 to 0.076 on held-out faces. The rest is rounding noise in the
   encoder.

❌ **onnxruntime-web default import** → wants the WebGPU "jsep" runtime files.
   ❌ `onnxruntime-web/wasm` with `wasmPaths` set to a folder → Vite dev appends
   `?import` to the runtime's dynamic import and fails on `public/` files.
   ❌ Bundling ORT via a dynamic `import("onnxruntime-web/wasm")` → the ORT chunk
   imports Vite helpers from the main bundle, so ORT's worker threads load the
   whole app and crash ("document is not defined"); MogNet hangs in production
   only. **Fix (current):** copy `ort.wasm.bundle.min.mjs` + wasm into
   `public/vendor/ort` and `import()` it by full URL with `@vite-ignore`
   (`mognet.ts:27`). Works in dev and production.

❌ **Wiring the theme to "start when ≥ 8"** would restart it every time the score
   crossed the line. `sounds.ts` keeps one `Audio` element: it only calls
   `play()` when paused and `pause()` when playing, so it resumes where it
   stopped. Verified in the browser: paused at 2.36 s, resumed from 2.36 s.

⚠️ **Audio autoplay:** browsers block `play()` until the user has clicked on the
   page. Choosing Camera or a photo counts as that click; errors are swallowed.

⚠️ **The dev server dies with the session that started it.** On 2026-10-11 the
   owner found localhost:5173 down. Run it from your own terminal
   (`npm --prefix web run dev`).

⚠️ **Moving `data/` while background jobs ran** broke two evaluation runs.
   Don't move the dataset mid-job.

⚠️ **The 8 GB M1 swaps** if embedding (MPS) runs alongside other heavy work.
   The 6-minute CLIP embedding pass took 34 minutes.

⚠️ **The dev machine has very little disk** (1–5 GB free during this work). The fp32 ONNX
   (345 MB, git-ignored) can be deleted after export.

## 5. Next steps (priority order)

1. **[HIGH] Spread the scores (owner's main complaint).** This is a display
   decision, so ask the owner which feel they want before building. Options:
   - *Stretch:* `display = 5 + k·(score − 5)`, e.g. k = 1.5, clamped to 0–10.
     Simplest. Every score keeps its order.
   - *Percentile:* show where the face ranks among SCUT faces (e.g. "top 8%"),
     or map that rank onto a chosen 0–10 curve. Most honest.
   - *Target curve:* pick how often each score should appear (e.g. 10% of faces
     ≥ 8) and fit a monotonic map to it in `05_export_web.py` /
     `model-params.json`.
   Apply it in `toTen()` (`web/src/lib/mognet.ts:44`) and in `geometryScore`
   (`geometry.ts`), then retune `MOG_THRESHOLD` (`sounds.ts`) so the theme plays
   at the intended rarity.
2. **[HIGH] Tune the eyebrow trigger on a real webcam.** Constants are at the top
   of `web/src/lib/eyebrow.ts`: `RAISE_RATIO` 1.15 (lower it if raises are
   missed, raise it if talking triggers it), `HOLD_FRAMES` 3, `COOLDOWN_MS` 1200.
   Consider logging `browHeights` ratios while raising and resting to choose a
   value from data.
3. **[HIGH] Distill a small on-device student** (the plan from the start):
   use `models/teacher_svr.joblib` to label SCUT plus extra unlabeled faces,
   then train MobileNetV3-Small or EfficientNet-Lite0 on the teacher's scores.
   Target 2–6 MB and real-time speed. It replaces the 89 MB int8 CLIP and fixes
   the drift issue. Add `06_distill.py`. Watch dataset licenses (FFHQ/CelebA
   are non-commercial too).
4. **[HIGH] Team decisions:** repo visibility given the SCUT-derived weights,
   repo license, and an age gate before any public deploy.
5. **[MEDIUM] Model file hosting:** move ONNX files to GitHub Releases or Git
   LFS before the next retrain to avoid bloating git history.
6. **[MEDIUM] Deploy:** static host that can send COOP/COEP headers (e.g.
   Netlify/Vercel/Cloudflare Pages `_headers`). Without them inference falls
   back to a single thread (not timed yet; expect it to be noticeably slower).
   The audio files are not in git, so a deploy needs its own licensed sounds.
7. **[MEDIUM] Own labels for commercial use:** pairwise "who mogs?" votes →
   Bradley–Terry scores, collected in-app. This fits the battle format and
   removes the SCUT dependency.
8. **[LOW] Head-to-head mode** (two faces, the Omoggle-style battle UI), and
   showing the face crop the model sees.
9. **[LOW] Cleanup:** `MOGNET_MB` in `mognet.ts` is exported but unused; the
   Python `faces()` in `05_export_web.py` reloads landmarks on each call; the old
   experiment folders `../teacher` and `../mogged` and the 345 MB
   `training/models/mognet-fp32.onnx` can be deleted to free disk.

## 6. Quick reference

- **Run app:** `cd web && npm install && npm run dev` (or `npm --prefix web run dev`
  from the repo root), then `open -a "Google Chrome" http://localhost:5173`.
  Tests `npm test` (6 tests: geometry/pose parity + eyebrow detector); build
  `npm run build && npm run preview`.
- **Sounds:** put `gigachad.mp3` and `vine-boom.mp3` in `web/public/audio/` (not in
  git). The **Sound** toggle in the UI mutes both. Threshold: `MOG_THRESHOLD`.
- **Retrain:** `cd training && uv sync && uv run 00_download.py && uv run 01_landmarks.py && uv run 02_geometry.py && uv run 03_embed.py && uv run 04_teacher.py && uv run 05_export_web.py`
  (00/01 about 1 min each, 03 about 6 min, 04 about 8 min, 05 about 5 min on an M1)
- **Score photos from the CLI:** `cd training && uv run predict.py face.jpg`
- **Test a face in the browser without a file dialog:** put a JPEG in
  `web/public/vendor/` (git-ignored), then from the console build a `File` with
  `DataTransfer`, assign it to the file input, and dispatch `change`.
- **Score scale:** MogNet = (SCUT rating − 1) / 4 × 10, so an average SCUT face
  (rating about 3.0) is about 5.0. Geometry is spread to match SCUT's rating
  distribution (mean 4.98, sd 1.72 on 0–10).
- **Sources:** SCUT-FBP5500 https://github.com/HCIILAB/SCUT-FBP5500-Database-Release ·
  paper https://arxiv.org/abs/1801.06345 · mogged https://github.com/elijah629/mogged ·
  MediaPipe Face Landmarker https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker ·
  Omoggle coverage: AUT News https://tewahanui.aut.ac.nz/culture/experts-warn-parents-as-omoggle-website-appears-in-nz-schools2
- **Secrets:** none are needed. GitHub pushes use the developer's own `gh` login.
