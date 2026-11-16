# Presentation Script — Image Forgery Detection Using Deep Learning
## 10-Minute College Presentation: Sections 7 & 8

---

> **Speaker Notes Format:**
> - Each slide section begins with the slide title.
> - _[Stage direction or timing note]_ in italics.
> - Spoken words are in plain text.
> - Keep each slide to roughly 60–90 seconds.

---

## SLIDE 1 — Title Slide

**Title:** Image Forgery Detection Using Deep Learning Techniques
**Subtitle:** Implementation, Integration, Deployment & Validation — Sections 7 & 8
**Team / Name / Roll Number**

_[Pause. Make eye contact. Speak clearly.]_

"Good morning, everyone. Today I'll be walking you through the implementation and validation of our Image Forgery Detection system — specifically Sections 7 and 8 of the project report. I'll cover how we built the system, how the different modules connect, how we deployed it, and what the evaluation results actually show."

---

## SLIDE 2 — What We Built (System Overview)

**Slide content:** One-line system description + a simple block diagram showing: User → Web App → 3 AI Models → GradCAM + Trust Score → User

_[~30 seconds]_

"At a high level, we built a web application where a user uploads an image, our deep learning models analyse it for signs of forgery, and the system returns a prediction, a visual explanation using GradCAM, and a risk level — Low, Suspicious, or High Risk. The whole thing runs as a Flask web app that can run locally on your laptop or deployed to the cloud."

---

## SLIDE 3 — Section 7: The Thirteen Modules

**Slide content:** A table or list of the 13 modules with one-word roles

_[~60 seconds — speak about the most important ones]_

"The system is built from thirteen distinct modules. I won't go through all of them in detail, but the key ones are:

- The **three AI models** — ResNet50 as our baseline, EfficientNet-B4 for efficiency, and our own **Hybrid model**, which is our main contribution.
- The **Flask web app** — 473 lines of Python that ties everything together: handles uploads, runs the models, generates heatmaps, extracts EXIF metadata, and sends back a JSON response.
- The **GradCAM module** — which creates visual explanations of what the model is looking at.
- The **Batch processor** — which handles up to 100 images at once, including ZIP file uploads.

The key design principle was: one file, one job. Every module has a single responsibility, which makes the code easier to maintain and test."

---

## SLIDE 4 — The Hybrid Model Architecture

**Slide content:** A diagram showing two branches merging: EfficientNet spatial (1792-dim) + Laplacian frequency CNN (256-dim) → 2048-dim → Classifier

_[~90 seconds — this is the most technical slide]_

"Let me explain our most important module — the Hybrid model. It has two branches that run on the same input image simultaneously.

The **first branch** is a standard EfficientNet-B4 — a state-of-the-art image classifier — which produces a 1,792-dimensional feature vector capturing the spatial content of the image. Things like edges, textures, object regions.

The **second branch** is our frequency domain branch. We first apply a **Laplacian filter** to the image. The Laplacian is a mathematical operation that highlights abrupt changes in pixel intensity — essentially, it shows you the edges and high-frequency residuals. We then run a small convolutional neural network on this Laplacian image, producing a 256-dimensional frequency feature vector.

We concatenate both vectors — 1,792 plus 256 — giving us 2,048 dimensions, and pass that through a classifier to get the final prediction.

The intuition is this: image forgeries disturb the natural frequency statistics of an image. Even after a forged image is shared on WhatsApp and JPEG-compressed, those frequency discontinuities at the forgery boundary don't fully disappear — and our frequency branch is specifically trained to detect them."

---

## SLIDE 5 — Integration Pipeline

**Slide content:** A simplified flowchart (5–6 boxes): Upload → Validate → Preprocess → Inference → GradCAM + EXIF + Trust → JSON Response → Display

_[~60 seconds]_

"When a user submits an image, here's what happens step by step inside the application:

1. The image is validated — we check the file type. If it's not a valid image, we return an error immediately.
2. It's preprocessed — resized to 224 by 224 pixels and normalised to ImageNet statistics.
3. The model runs inference — by default, we use the Hybrid model.
4. In parallel, we generate a GradCAM heatmap, extract EXIF metadata, compute the trust score, run the compression simulation for all four social-media platforms, and compare all three models.
5. Everything is packed into a JSON response and displayed in the browser.

This whole pipeline runs in one request-response cycle, typically in 3 to 6 seconds on a standard laptop."

---

## SLIDE 6 — EXIF Metadata and Trust Scoring

**Slide content:** Show the 16 EXIF features listed briefly, and the three trust levels with colours

_[~45 seconds]_

"Two smaller but important features worth mentioning:

First, **EXIF metadata extraction**. Every JPEG photo taken by a camera or smartphone contains hidden metadata — the focal length, ISO, flash setting, GPS presence, and more. We extract 16 of these features and display them alongside the prediction. In Phase 2 of this project, these features would actually feed into the model itself to improve accuracy.

Second, the **trust score**. We convert the model's raw forgery probability into one of three simple categories: if the forgery probability is below 30%, it's LOW RISK — shown in green. Between 30 and 70%, it's SUSPICIOUS — shown in amber. Above 70%, it's HIGH RISK — shown in red. This makes the system usable by non-experts."

---

## SLIDE 7 — Section 8: Deployment Environment

**Slide content:** Two-column table: Local Dev vs. Cloud Production

_[~45 seconds]_

"Moving to Section 8 — Deployment.

Locally, we developed on Windows 11 with Python 3.14.2, using PyTorch running entirely on CPU — no GPU was needed. The app runs with Flask's built-in development server.

For cloud deployment, we've configured the app for both **Render.com** and **Railway.app** — two popular free-tier cloud platforms. On the cloud, we use Python 3.11 — the stable version — and **gunicorn** as the production-grade web server. The app listens on whatever port the cloud platform assigns via an environment variable.

We also added a `/health` endpoint that returns a simple okay status, which is standard practice for cloud deployments so the platform knows the service is running."

---

## SLIDE 8 — Test Results (TC-001 to TC-015)

**Slide content:** A condensed version of the test table — group by category

_[~60 seconds]_

"We ran 15 functional test cases on the running application. Let me summarise:

For single-image analysis — tests TC-001 through TC-008 — all passed. The API returns correctly structured JSON with the prediction, confidence score, GradCAM heatmap URL, EXIF block, trust level, model comparison, and platform results.

For batch processing — TC-009 and TC-010 — both passed. We successfully processed multiple images in a single request and also tested ZIP file uploads.

For error handling — TC-011 — also passed. Uploading a text file correctly returned a 400 error.

For page loading and health check — TC-013 through TC-015 all pass. TC-012, the health check test, is pending because the endpoint was added during the deployment-fix phase and needs one final re-test.

Overall: 14 out of 15 tests passed. 1 is pending."

---

## SLIDE 9 — Model Evaluation Results

**Slide content:** Table with clean accuracy and average accuracy drop for all 3 models

_[~90 seconds — the core academic result]_

"Now the most important results. We evaluated all three models on clean images and after simulating compression by four social-media platforms: Facebook, Instagram, WhatsApp, and Twitter.

**ResNet50** achieves the highest clean accuracy at 82.1%. But under compression, it falls sharply — an average drop of 14.1 percentage points across the four platforms. On WhatsApp, which compresses images twice, it drops to just 67.1%.

**EfficientNet-B4** starts lower at 77% clean accuracy, but drops only 2.3 percentage points under compression. Much more robust.

**Our Hybrid model** achieves 78.2% clean accuracy — slightly above EfficientNet-B4 — and shows the best robustness, with just a 1.7 percentage point average drop.

Why is the Hybrid model the most robust? Because its Laplacian frequency branch detects forgery artifacts that survive JPEG compression. ResNet50 relies on pixel-level texture, which JPEG compression destroys. The frequency discontinuities at a forgery boundary, however, survive even heavy compression — and that's exactly what our second branch is trained on.

One note: an earlier draft of the paper cited 92.2% for the Hybrid model's clean accuracy. The final verified evaluation from our results file shows 78.2%. We report the verified figure throughout."

---

## SLIDE 10 — Limitations and Future Work

**Slide content:** Bullet list of 4–5 limitations

_[~45 seconds]_

"To conclude, let me be transparent about the current limitations:

1. The Hybrid model has slightly **lower clean accuracy than ResNet50** — 78.2% vs 82.1%. There's a trade-off between peak accuracy and robustness.
2. The **EXIF-aware model (Phase 2)** is architecturally defined but not yet trained. The EXIF metadata branch exists in code but isn't active in inference yet.
3. All inference is **CPU-only**. GPU acceleration would reduce latency from several seconds to sub-second.
4. The system hasn't been benchmarked on **standard public datasets** like CASIA or Columbia — that's a recommended next step.

Thank you. I'm happy to take any questions."

---

## TIME GUIDE

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | ~20 s |
| 2 | System Overview | ~30 s |
| 3 | Thirteen Modules | ~60 s |
| 4 | Hybrid Architecture | ~90 s |
| 5 | Integration Pipeline | ~60 s |
| 6 | EXIF + Trust Scoring | ~45 s |
| 7 | Deployment Environment | ~45 s |
| 8 | Test Results | ~60 s |
| 9 | Model Evaluation | ~90 s |
| 10 | Limitations + Q&A | ~45 s |
| **Total** | | **~9 min 45 s** |

---

> [!TIP]
> Practice with a timer. Slides 4 and 9 are the most important and can expand if time permits. If you're running short, Slide 6 (EXIF/Trust) can be condensed to 20 seconds.
