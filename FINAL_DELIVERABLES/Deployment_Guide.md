# Deployment Guide — Image Forgery Detection Web App

> **Project:** Image Forgery Detection in Social Media Images Using Deep Learning  
> **Team:** Lekha R · Athul Krishna · Darshan S S · Kanagala Sri Harshitha  
> **Guide:** Ms. Sandhya B R, NMIT

---

## Table of Contents

1. [Local Development](#1-local-development)
2. [Production Requirements](#2-production-requirements)
3. [Deploy on Render.com (Recommended — Free Tier)](#3-deploy-on-rendercom-recommended--free-tier)
4. [Deploy on Railway.app](#4-deploy-on-railwayapp)
5. [Deploy on Heroku](#5-deploy-on-heroku)
6. [Environment Variables](#6-environment-variables)
7. [Health Check](#7-health-check)
8. [Troubleshooting](#8-troubleshooting)
9. [Known Limitations](#9-known-limitations)

---

## 1. Local Development

### Prerequisites

- Python 3.11+ (3.14 also works; tested on 3.14.2)
- All model checkpoint files in `saved_models/`
- Dependencies installed

### Quick Start

```bash
# Clone the repository
git clone https://github.com/nmit-1NT23CS120/22CSP69_57-Image-Forgery-Detection.git
cd "Image Forgery Detection Using Deep Learning Techniques"

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Option A — startup.py wrapper (recommended for local dev)
python startup.py

# Option B — run from web_app directory directly
cd web_app
python app.py
```

Open your browser at **http://localhost:5000**

### Model Checkpoints

Ensure all three checkpoints are present before starting:

```
saved_models/
├── resnet_baseline.pth      (~94 MB)
├── efficientnet_model.pth   (~71 MB)
└── hybrid_model.pth         (~75 MB)
```

> `hybrid_model_v2_exif.pth` is optional — the app gracefully falls back to the Phase 1 hybrid model if it is absent.

---

## 2. Production Requirements

| Item | Minimum | Notes |
|------|---------|-------|
| Python | 3.11.9 | Specified in `runtime.txt` and `.python-version` |
| RAM | 2 GB | Models load ~240 MB total; inference adds temporary overhead |
| Storage | 500 MB | Models + app code + uploads cache |
| CPU | Any | GPU not required; inference runs on CPU in ~3–8 s/image |
| Workers | 1 | More workers = more RAM copies of models |

> [!WARNING]
> Free-tier cloud platforms (Render, Railway) typically provide 512 MB–1 GB RAM.  
> With 3 models (~240 MB) loaded at startup, free tiers may OOM.  
> Use **1 worker** and test with the smallest plan available.

---

## 3. Deploy on Render.com (Recommended — Free Tier)

Render.com provides a straightforward Python web service with automatic HTTPS.

### Steps

1. **Push your code** to a GitHub repository (models committed or handled via LFS).

2. **Create a new Web Service** on [render.com](https://render.com):
   - Connect your GitHub repository
   - Render auto-detects `render.yaml` — accept the detected settings

3. **Manual settings** (if `render.yaml` is not auto-detected):
   | Field | Value |
   |-------|-------|
   | Environment | Python |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `cd web_app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120` |
   | Python Version | 3.11.0 |

4. **Environment Variables** — none required; `PORT` is injected automatically by Render.

5. Click **Deploy**. The build takes 2–5 minutes (pip install).

6. Access your app at the assigned `*.onrender.com` URL.

### render.yaml (already in project root)

```yaml
services:
  - type: web
    name: image-forgery-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: cd web_app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

---

## 4. Deploy on Railway.app

Railway uses NIXPACKS for automatic environment detection.

### Steps

1. Install the Railway CLI:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. From the project root:
   ```bash
   railway init
   railway up
   ```

3. Railway reads `railway.toml` and `nixpacks.toml` automatically:
   - NIXPACKS installs `libGL` and `libglib` (required by OpenCV)
   - Starts gunicorn on the assigned `$PORT`

4. Set a custom domain in the Railway dashboard if desired.

### railway.toml (already in project root)

```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "cd web_app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120"
restartPolicyType = "ON_FAILURE"
```

### nixpacks.toml (already in project root)

```toml
[phases.setup]
nixPkgs = ["python311", "libGL", "libglib"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "cd web_app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120"
```

---

## 5. Deploy on Heroku

### Steps

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and log in.

2. From the project root:
   ```bash
   heroku create image-forgery-detection
   git push heroku main
   ```

3. Heroku reads `Procfile` automatically:
   ```
   web: cd web_app && gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
   ```

4. Set the Python version via `runtime.txt` (already present):
   ```
   python-3.11.9
   ```

> [!NOTE]
> Heroku's free dyno tier was discontinued. Use Eco dynos ($5/month) or higher.

---

## 6. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Port the app listens on (injected by cloud platforms) |

No API keys or secrets are required. The app uses only local model checkpoints.

---

## 7. Health Check

The app exposes a `/health` endpoint for platform liveness probes:

```
GET /health
→ 200 OK
→ {"status": "ok"}
```

Configure your platform's health-check path to `/health`.

---

## 8. Troubleshooting

### `ImportError: libGL.so.1: cannot open shared object file`

OpenCV requires system-level OpenGL libraries. On Railway with NIXPACKS, `nixpacks.toml` already includes `libGL` and `libglib`. On other Linux environments:

```bash
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

Or switch to the headless OpenCV variant:

```bash
pip install opencv-python-headless
```

> Update `requirements.txt` to `opencv-python-headless>=4.8.0` for headless servers.

### `OSError: [Errno 28] No space left on device`

Model files total ~240 MB. Ensure your platform has at least 500 MB persistent storage.

### App starts but returns 502 / times out

- Increase gunicorn `--timeout` (already 120 s)
- Reduce workers to 1 (already set)
- Check RAM: model loading requires ~400 MB peak

### Models not found on startup

Ensure `saved_models/` is committed to the repository or mounted as a persistent disk. `BASE_DIR` in `app.py` resolves as:

```
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

This is relative to `web_app/app.py`, so `saved_models/` must be one level up from `web_app/`.

### EXIF model not available warning

```
INFO: hybrid_model_v2_exif.pth not found — EXIF-aware model not loaded
```

This is expected. The EXIF-aware model (Phase 2) checkpoint is not included. The app falls back to the Phase 1 hybrid model for all predictions. No action needed.

---

## 9. Known Limitations

| Limitation | Detail |
|-----------|--------|
| CPU-only inference | No GPU required, but analysis takes 3–8 s per image |
| Model loading time | ~10–20 s cold start to load all 3 checkpoints |
| Free-tier RAM | 512 MB free tiers may OOM; test on 1 GB+ plans |
| No persistent upload storage | Uploads are stored in `static/uploads/` in-container; they reset on dyno restart |
| Max batch size | Capped at 100 images per batch (`MAX_BATCH_IMAGES` in `batch_utils.py`) |
| EXIF model | `hybrid_model_v2_exif.pth` not available; falls back to Phase 1 hybrid |

---

*Generated for the Image Forgery Detection Major Capstone Project — NMIT, VTU, 2025–26*
