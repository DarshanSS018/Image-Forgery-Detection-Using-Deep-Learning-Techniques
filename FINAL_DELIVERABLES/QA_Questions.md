# QA Questions — Image Forgery Detection Project
## 15 Likely Questions from Project Guide (Sections 7 & 8)

---

### Q1. ResNet50 achieves the highest clean accuracy (82.1%) but shows the worst degradation under social-media compression (−14.1 pp). Why?

**Answer:**
ResNet50 is a purely spatial model fine-tuned on ImageNet. Its feature representations are dominated by high-frequency texture patterns — fine-grained pixel-level statistics that distinguish authentic images from forged ones under ideal conditions. JPEG compression (the core mechanism behind all four platform simulations) works by discarding high-frequency DCT coefficients via quantisation. This directly destroys the texture cues ResNet50 relies on, causing its recall (sensitivity to forged images) to collapse most sharply — dropping to 45.2% under WhatsApp's double-compression. The fundamental mismatch is that ResNet50's discriminative features and JPEG's lossy mechanism operate in the same frequency band.

---

### Q2. Why does the Hybrid model maintain robustness under social-media compression?

**Answer:**
The Hybrid model's frequency branch operates on the **Laplacian of the input image** — a second-order derivative filter that amplifies abrupt spatial discontinuities. Forgery operations (copy-move, splicing, inpainting) introduce statistical breaks in the natural frequency continuity of an image, particularly at the forgery boundary. These breaks manifest as localised, structurally significant high-frequency discontinuities — not global texture energy. JPEG compression attenuates global high-frequency texture uniformly across the image but cannot fully erase localised boundary discontinuities, because those are abrupt statistical breaks rather than smooth, globally distributed texture. The CNN trained on Laplacian residuals learns to detect precisely these compression-resistant boundary artifacts, giving the Hybrid model its robustness advantage (−1.7 pp average drop vs. −14.1 pp for ResNet50).

---

### Q3. What is Grad-CAM and why is it used in this project?

**Answer:**
**Gradient-weighted Class Activation Mapping (Grad-CAM)** is an explainability technique that produces a spatial heatmap highlighting which regions of an input image were most influential for a model's prediction. It works by computing the gradient of the target class score with respect to the feature maps of a chosen convolutional layer, globally average-pooling those gradients to produce channel-wise importance weights, and computing a weighted sum of the feature maps followed by a ReLU activation (to retain only positively contributing regions). The result is upsampled and overlaid on the original image.

In this project, GradCAM is applied to `spatial_features[-1]` — the last feature map of the EfficientNet-B4 spatial branch within the Hybrid model. It serves two purposes: (1) **trust building** — users can verify the model is highlighting semantically meaningful regions (copy-move areas, splicing boundaries) rather than background; (2) **forensic utility** — analysts can use the heatmap as a spatial pointer to investigate regions of interest within a suspicious image.

---

### Q4. What is the EXIF metadata branch doing, and why might it help forgery detection?

**Answer:**
The EXIF (Exchangeable Image File Format) branch extracts a 16-dimensional numeric feature vector from the JPEG metadata of an uploaded image. The 16 features include focal length, ISO, flash mode, orientation, GPS presence flag, exposure time, F-number, colour space tag, and others. The underlying hypothesis is that authentic images captured by real cameras exhibit consistent, physically plausible EXIF signatures, while forged or synthetically generated images often have missing, inconsistent, or anomalous EXIF data (e.g., GPS absent when flash is on, mismatched software tags, missing focal length).

In Phase 1 (current), EXIF data is extracted and displayed to the user for transparency but not fed into the model. In Phase 2 (`HybridModelV2`, architecturally defined but not yet trained), the EXIF vector passes through a 3-layer MLP producing a 32-dimensional embedding that is concatenated with the 2,048-dimensional spatial+frequency fusion to form a 2,080-dimensional input to the final classifier. This would allow the model to jointly reason about pixel-level and metadata-level evidence.

---

### Q5. How does the frequency domain branch work? What is a Laplacian filter?

**Answer:**
The **Laplacian filter** is a discrete approximation of the second-order spatial derivative of an image. It is computed by convolving the image with a kernel such as:

```
 0  -1   0
-1   4  -1
 0  -1   0
```

This amplifies regions where pixel intensity changes rapidly (edges, boundaries, texture transitions) and suppresses uniform regions. When applied to an image, the Laplacian residual highlights not just edges but all abrupt frequency-domain discontinuities. For a forged image, the forgery boundary introduces a statistical anomaly — a sharp transition between two regions with different frequency statistics — which appears prominently in the Laplacian. A lightweight CNN is then trained on these Laplacian residuals to learn forgery-specific frequency patterns. This CNN produces a 256-dimensional feature vector that is concatenated with the EfficientNet spatial branch output.

---

### Q6. Why not use a full Transformer / Vision Transformer (ViT) instead of this hybrid CNN approach?

**Answer:**
Several practical reasons informed this choice:

1. **Data scale** — Vision Transformers require large-scale pre-training (ImageNet-21k or JFT-300M) and large fine-tuning datasets to match CNN performance. Our dataset is comparatively smaller; CNNs generalise better in this regime.
2. **Computational cost** — ViTs are significantly more computationally expensive at inference time, particularly without GPU hardware. Our system runs on CPU-only hardware, making transformer inference latency impractical for a real-time web app.
3. **Inductive bias for local forensics** — CNNs have a local receptive field that is well-suited to detecting local forgery artifacts (copy-move seams, splicing boundaries). ViTs' global self-attention aggregates over the entire image, potentially diluting local artifact signals.
4. **Explicit frequency modelling** — Our Laplacian branch provides an explicit frequency-domain inductive bias that neither a standard CNN nor a ViT provides by default. This proved important for compression robustness.

A hybrid CNN + lightweight attention mechanism is a reasonable future extension, but it would not replace the explicit frequency branch.

---

### Q7. How was the dataset split? Why stratified splitting?

**Answer:**
The dataset was partitioned into train (70%), validation (15%), and test (15%) subsets using **stratified splitting** — meaning the class ratio (authentic vs. forged) is preserved identically across all three splits. This prevents the model from benefiting from an imbalanced test set where, for example, the majority class dominates. The evaluation metrics (accuracy, F1, precision, recall) are all computed on the held-out test partition that the model has never seen during training or hyperparameter tuning. The same test partition was used for the compression-robustness evaluation to ensure a valid comparison.

---

### Q8. What is the trust score and how were the thresholds chosen?

**Answer:**
The trust score converts the model's raw forgery probability (`fp`, a float in [0, 1]) into a categorical risk label displayed to the user:
- `fp < 0.3` → **LOW RISK** (green)
- `0.3 ≤ fp < 0.7` → **SUSPICIOUS** (amber)
- `fp ≥ 0.7` → **HIGH RISK** (red)

The 0.3 and 0.7 thresholds were chosen to bracket the model's uncertain decision region symmetrically around 0.5. The amber "SUSPICIOUS" band is intentional: for borderline predictions where the model lacks confidence, the system escalates to human review rather than making an automated binary decision. This follows a conservative design principle appropriate for a forensic tool where false negatives (missing a forgery) have higher cost than false positives.

---

### Q9. What platforms were tested in the compression simulation, and how were they simulated?

**Answer:**
Four platforms were simulated:

| Platform  | Simulation Method |
|-----------|------------------|
| Facebook  | JPEG re-encode at quality 75, downsample to max 2,048 px |
| Instagram | JPEG re-encode at quality 78, downsample to max 1,080 px |
| WhatsApp  | JPEG re-encode at quality 60, then again at quality 55, downsample to max 1,600 px |
| Twitter   | JPEG re-encode at quality 82, downsample to max 1,200 px |

Simulation is performed using Python's Pillow library (`Image.save(..., quality=N)`) with downsampling applied before the first JPEG encode if the image exceeds the platform's maximum dimension. WhatsApp is the most aggressive case because it applies two sequential lossy compressions, which causes the greatest cumulative information loss. These parameters were derived from published analysis of platform compression behaviour.

---

### Q10. What deployment platforms are supported, and why were Render.com and Railway.app chosen?

**Answer:**
Both **Render.com** and **Railway.app** were chosen because:
1. They offer **free-tier web service hosting** with no credit card required for initial deployment.
2. Both support **Python / gunicorn** natively with auto-detected build configuration.
3. They accept `Procfile`-based configuration (Heroku-compatible), making migration straightforward.
4. They support **environment variable injection** (`$PORT`), which the app reads to bind gunicorn dynamically.

The free tier limitations (512 MB RAM, cold-start latency, single worker) are acceptable for a prototype demonstration. For production use, a higher-tier plan or a VM-based deployment (e.g., Google Cloud Run, AWS Fargate) with GPU support would be recommended.

---

### Q11. The clean accuracy of the Hybrid model (78.2%) is lower than ResNet50 (82.1%). Doesn't that make ResNet50 the better model overall?

**Answer:**
No — the choice depends on the deployment context. ResNet50's accuracy advantage on clean images disappears entirely under real-world usage patterns, where images are almost always shared through social media before analysis. On compressed images, ResNet50's accuracy drops to 67.1–71.8%, well below the Hybrid model's 75–76.5%. For a system designed to detect forgeries in social-media content (the stated use case), the Hybrid model is strictly better in practice. ResNet50's higher clean accuracy reflects its over-reliance on high-frequency texture that is not compression-invariant, not a generalisation advantage.

---

### Q12. How does the GradCAM implementation differ between `gradcam.py` and the inline version in `app.py`?

**Answer:**
`explainability/gradcam.py` is a standalone importable module that takes any model and target layer as arguments — designed for offline batch analysis and experimentation without running the full web app. It uses the same `register_full_backward_hook` mechanism but is model-agnostic.

The inline GradCAM in `app.py` is tightly coupled to the Hybrid model's architecture: it specifically hooks onto `model.spatial_features[-1]` — the last convolutional feature map of the EfficientNet-B4 spatial branch — because this layer has the best trade-off between spatial resolution and semantic abstraction for forgery localisation. The heatmap is blended with the original image at 60% opacity using OpenCV's Jet colormap and saved as a PNG to the `static/results/` directory, where it's served directly to the browser.

---

### Q13. What would happen if someone uploads a PNG image instead of JPEG? Can the system still extract EXIF?

**Answer:**
PNG images can be processed for forgery detection without issue — the preprocessing pipeline (resize, normalise, model inference, GradCAM) is format-agnostic through Pillow's `Image.open()`. However, **EXIF metadata is only available in JPEG files** (and some TIFF files). PNG does not embed EXIF by standard. For PNG uploads, `exif_utils.extract_exif_features()` returns a sentinel vector of −1 values for all 16 features, and the EXIF block in the response includes `"available": false`. The forgery prediction is unaffected; only the EXIF-based Phase 2 enhancement would be non-functional.

---

### Q14. If the HybridModelV2 (Phase 2) is not trained, why is `models_v2.py` included in the submission?

**Answer:**
`models_v2.py` is included as an **architectural specification** — it documents the Phase 2 design decision in executable code rather than prose alone. It demonstrates that the EXIF integration point is well-defined (the 32-dimensional MLP branch, the 2,080-dimensional fusion layer, the classifier head) and that the transition from Phase 1 to Phase 2 requires only training the model and providing the checkpoint — no architectural changes to `app.py` are needed beyond the checkpoint load path. This is a software engineering best practice: the interface is defined before the implementation is complete, allowing parallel work (training the model vs. building the web UI).

---

### Q15. How confident are you in the evaluation results given that an earlier paper draft cited 92.2% for the Hybrid model?

**Answer:**
We are confident in the **78.2% figure** from `evaluation/results/results.json` for the following reasons:

1. The evaluation script (`evaluate_all.py`) loads the model from the final saved checkpoint (`hybrid_model.pth`) and evaluates on the held-out test partition.
2. The results JSON was generated by running the evaluation after all training was complete — not during development.
3. The discrepancy is likely explained by the draft evaluating on a training or validation subset (or a smaller, easier test sample) rather than the full held-out test partition. This is a common source of optimistic preliminary numbers in research workflows.

We document this discrepancy explicitly in both the report and the evaluation output: *"Paper-reported figures differ from final evaluation output; we report verified results.json values throughout."* Academic integrity requires reporting the verified reproducible number, even if it is less impressive than the preliminary figure.
