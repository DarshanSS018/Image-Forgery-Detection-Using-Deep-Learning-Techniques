"""
=============================================================
Core Inference and Processing Engine for Image Forgery Detection
=============================================================
This module provides independent, reusable, and cached components for:
  1. Safe Hugging Face model checkpoint retrieval with Streamlit secrets fallback.
  2. Neural network model architectures (ResNet50, EfficientNet-B4, Hybrid, Hybrid V2).
  3. Explainable AI via Grad-CAM with proper hook lifecycle management.
  4. Social media compression simulation (Facebook, Instagram, WhatsApp, Twitter).
  5. EXIF metadata extraction and forensic feature vector analysis.
  6. In-memory single-image and batch prediction pipelines (zero file accumulation).
=============================================================
"""

import os
import sys
import io
import zipfile
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
WEB_APP_DIR = os.path.join(BASE_DIR, 'web_app')

if WEB_APP_DIR not in sys.path:
    sys.path.insert(0, WEB_APP_DIR)

# Import existing EXIF extraction and V2 architecture
try:
    from exif_utils import extract_exif_features, EXIF_FEATURE_NAMES, EXIF_FEATURE_DIM
except ImportError:
    # Fallback if imported from different working dir
    sys.path.append(os.path.join(BASE_DIR, 'web_app'))
    from exif_utils import extract_exif_features, EXIF_FEATURE_NAMES, EXIF_FEATURE_DIM

try:
    from models_v2 import HybridModelV2
except ImportError:
    HybridModelV2 = None

# Hardware device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard ImageNet normalization transform
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

HF_REPO_ID = "Phoenix0181/image-forgery-detection-models"


# ─────────────────────────────────────────────────────────────
# 1. HUGGING FACE TOKEN & CHECKPOINT RETRIEVAL
# ─────────────────────────────────────────────────────────────

def get_hf_token():
    """
    Safely retrieves the Hugging Face token.
    Checks Streamlit secrets first, then environment variables.
    Never exposes or logs token values.
    """
    token = None
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "HF_TOKEN" in st.secrets:
                token = st.secrets["HF_TOKEN"]
            elif "hf_token" in st.secrets:
                token = st.secrets["hf_token"]
    except Exception:
        pass

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    return token


def ensure_model_file(filename: str, repo_id: str = HF_REPO_ID) -> str:
    """
    Ensures that the required .pth checkpoint exists locally in saved_models/.
    If not present or 0 bytes, downloads it from Hugging Face using get_hf_token().
    Returns the absolute path to the verified local file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    token = get_hf_token()

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=MODELS_DIR
        )
        return downloaded_path
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "403" in err_msg or token is None:
            raise RuntimeError(
                f"Model checkpoint '{filename}' is missing locally, and downloading from "
                f"private repo '{repo_id}' requires authentication.\n\n"
                f"Please add your Hugging Face token in Streamlit Community Cloud:\n"
                f"  Settings -> Secrets -> add: HF_TOKEN = \"hf_...\"\n"
                f"Or set the HF_TOKEN environment variable."
            ) from e
        else:
            raise RuntimeError(
                f"Failed to retrieve checkpoint '{filename}' from Hugging Face: {err_msg}"
            ) from e


# ─────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────

class ResNetBaseline(nn.Module):
    """ResNet50 baseline model with custom binary classification head."""
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=None)
        backbone_layers = list(b.children())[:-1]
        self.model = nn.Sequential(
            nn.Sequential(*backbone_layers),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)

    def load_ckpt(self, state):
        remapped = {f'model.{k}': v for k, v in state.items()}
        self.load_state_dict(remapped, strict=False)


class EfficientNetModel(nn.Module):
    """EfficientNet-B4 spatial feature extractor with classification head."""
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b4(weights=None)
        self.features = b.features
        self.avgpool = b.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


class FrequencyBranch(nn.Module):
    """Laplacian high-pass filtering + CNN branch for frequency-domain analysis."""
    def __init__(self):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                         dtype=x.dtype, device=x.device)
        f = F.conv2d(g, k, padding=1)
        f = torch.abs(f) / (torch.abs(f).max() + 1e-8)
        return self.fc(self.freq_cnn(f))


class HybridModel(nn.Module):
    """
    Two-branch fusion model:
      - Spatial: EfficientNet-B4 (1792-dim)
      - Frequency: Laplacian CNN (256-dim)
      - Total Fused: 2048-dim -> Classifier Head
    """
    def __init__(self):
        super().__init__()
        b = models.efficientnet_b4(weights=None)
        self.spatial_features = b.features
        self.spatial_pool = b.avgpool
        self.freq_branch = FrequencyBranch()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1792 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        sp = self.spatial_pool(self.spatial_features(x)).squeeze(-1).squeeze(-1)
        fr = self.freq_branch(x)
        return self.classifier(torch.cat([sp, fr], dim=1))


# ─────────────────────────────────────────────────────────────
# 3. LAZY MODEL LOADERS
# ─────────────────────────────────────────────────────────────

def load_resnet_model(device=DEVICE) -> ResNetBaseline:
    path = ensure_model_file('resnet_baseline.pth')
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state', ck)
    m = ResNetBaseline().to(device)
    m.load_ckpt(state)
    m.eval()
    return m


def load_efficientnet_model(device=DEVICE) -> EfficientNetModel:
    path = ensure_model_file('efficientnet_model.pth')
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state', ck)
    m = EfficientNetModel().to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def load_hybrid_model(device=DEVICE) -> HybridModel:
    path = ensure_model_file('hybrid_model.pth')
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get('model_state', ck)
    m = HybridModel().to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def load_hybrid_v2_model(device=DEVICE):
    if HybridModelV2 is None:
        return None
    try:
        path = ensure_model_file('hybrid_model_v2_exif.pth')
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck.get('model_state', ck)
        m = HybridModelV2().to(device)
        m.load_state_dict(state, strict=True)
        m.eval()
        return m
    except Exception as e:
        print(f"Hybrid V2 could not be loaded: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# 4. EXPLAINABLE AI (GRAD-CAM)
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM implementation with explicit hook removal to prevent memory leaks."""
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts = None
        self.fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach())
        )
        self.bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach())
        )

    def remove(self):
        try:
            self.fh.remove()
        except Exception:
            pass
        try:
            self.bh.remove()
        except Exception:
            pass

    def generate(self, tensor):
        self.model.eval()
        out = self.model(tensor)
        ci = out.argmax(1).item()
        self.model.zero_grad()
        oh = torch.zeros_like(out)
        oh[0][ci] = 1.0
        out.backward(gradient=oh, retain_graph=True)

        if self.grads is None or self.acts is None:
            return np.zeros((tensor.shape[2], tensor.shape[3])), ci

        pg = self.grads.mean(dim=[0, 2, 3])
        a = self.acts[0].clone()
        for i in range(a.shape[0]):
            a[i] *= pg[i]
        hm = a.mean(0).cpu().numpy()
        hm = np.maximum(hm, 0)
        if hm.max() > 0:
            hm /= hm.max()
        return hm, ci


def make_heatmap(img_np, heatmap):
    """Blends Jet colormap with RGB image."""
    h, w = img_np.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hmu = np.uint8(255 * hm)
    hmc = cv2.cvtColor(cv2.applyColorMap(hmu, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np.astype(np.uint8), 0.55, hmc, 0.45, 0)


# ─────────────────────────────────────────────────────────────
# 5. SOCIAL MEDIA COMPRESSION SIMULATORS
# ─────────────────────────────────────────────────────────────

def jpeg_compress_pil(pil_img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).copy()


def compress_facebook(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > 2048:
        s = 2048 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return jpeg_compress_pil(img, 75)


def compress_instagram(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > 1080:
        s = 1080 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return jpeg_compress_pil(img, 78)


def compress_whatsapp(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > 1600:
        s = 1600 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = jpeg_compress_pil(img, 60)
    return jpeg_compress_pil(img, 55)


def compress_twitter(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > 1200:
        s = 1200 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return jpeg_compress_pil(img, 82)


COMPRESSORS = {
    'facebook': compress_facebook,
    'instagram': compress_instagram,
    'whatsapp': compress_whatsapp,
    'twitter': compress_twitter,
}


# ─────────────────────────────────────────────────────────────
# 6. EXIF HUMAN READABLE DETAILS
# ─────────────────────────────────────────────────────────────

def get_detailed_exif_dict(pil_img: Image.Image) -> dict:
    """Extracts human-readable metadata tags and tamper flags."""
    details = {
        'has_exif': False,
        'make': 'Unknown',
        'model': 'Unknown',
        'software': 'None / Clean',
        'datetime_original': 'Not specified',
        'datetime_modified': 'Not specified',
        'resolution': 'Unknown',
        'gps': 'No coordinates embedded',
        'flags': []
    }

    try:
        exif = pil_img.getexif()
        if not exif or len(exif) == 0:
            details['flags'].append("Stripped EXIF (Common for social media or edited files)")
            return details

        details['has_exif'] = True
        raw_tags = {}
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            raw_tags[name] = value

        details['make'] = str(raw_tags.get('Make', 'Unknown'))
        details['model'] = str(raw_tags.get('Model', 'Unknown'))
        software = str(raw_tags.get('Software', '')).strip()
        details['software'] = software if software else 'None / Clean'

        # Check editing keywords
        editor_keywords = ['photoshop', 'gimp', 'lightroom', 'picsart', 'snapseed',
                           'facetune', 'canva', 'affinity', 'paint.net', 'pixlr']
        if any(k in software.lower() for k in editor_keywords):
            details['flags'].append(f"Suspicious Software tag detected: '{software}'")

        dt_orig = raw_tags.get('DateTimeOriginal')
        dt_mod = raw_tags.get('DateTime')
        if dt_orig:
            details['datetime_original'] = str(dt_orig)
        if dt_mod:
            details['datetime_modified'] = str(dt_mod)

        if dt_orig and dt_mod and str(dt_orig) != str(dt_mod):
            details['flags'].append("DateTime mismatch (Modified after camera capture)")

        x_res = raw_tags.get('XResolution')
        y_res = raw_tags.get('YResolution')
        if x_res and y_res:
            details['resolution'] = f"{x_res} x {y_res} DPI"

        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps_ifd:
                details['gps'] = "GPS metadata present"
        except Exception:
            pass

    except Exception as e:
        details['flags'].append(f"EXIF parsing warning: {e}")

    return details


# ─────────────────────────────────────────────────────────────
# 7. INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

def predict_single(model, pil_img: Image.Image, device=DEVICE,
                   generate_heatmap: bool = False, model_name: str = "hybrid") -> dict:
    """
    Runs prediction on a single PIL Image.
    Returns verdict, confidence, probabilities, trust level, and optional heatmap array.
    """
    model.eval()
    orig_np = np.array(pil_img.convert('RGB').resize((224, 224)))
    tensor = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]

    fp = float(probs[1])
    ap = float(probs[0])

    if fp < 0.3:
        trust, color = 'LOW RISK', '#22c55e'
    elif fp < 0.7:
        trust, color = 'SUSPICIOUS', '#f59e0b'
    else:
        trust, color = 'HIGH RISK', '#ef4444'

    result = {
        'predicted': 'FORGED' if fp > 0.5 else 'AUTHENTIC',
        'forged_prob': round(fp * 100, 1),
        'auth_prob': round(ap * 100, 1),
        'confidence': round((fp if fp > 0.5 else ap) * 100, 1),
        'trust': trust,
        'trust_color': color,
        'heatmap_overlay': None,
        'raw_heatmap': None,
    }

    if generate_heatmap and hasattr(model, 'spatial_features'):
        gcam = None
        try:
            target_layer = model.spatial_features[-1]
            gcam = GradCAM(model, target_layer)
            t2 = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(device)
            t2.requires_grad_(True)
            hm, _ = gcam.generate(t2)
            overlay = make_heatmap(orig_np, hm)
            result['heatmap_overlay'] = overlay
            result['raw_heatmap'] = hm
        except Exception as e:
            print(f"GradCAM computation notice: {e}")
        finally:
            if gcam is not None:
                gcam.remove()

    return result


def predict_hybrid_v2(hybrid_v2_model, pil_img: Image.Image,
                      img_input_for_exif, device=DEVICE) -> dict:
    """Runs the 3-branch Hybrid V2 model (Spatial + Frequency + EXIF MLP)."""
    if hybrid_v2_model is None:
        return {'available': False, 'note': 'Hybrid V2 model not available'}

    exif_vec = extract_exif_features(img_input_for_exif)
    exif_tensor = torch.from_numpy(exif_vec).unsqueeze(0).to(device)
    tensor = TRANSFORM(pil_img.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        out = hybrid_v2_model(tensor, exif_tensor)
        probs = torch.softmax(out, dim=1)[0]

    fp = float(probs[1])
    ap = 1.0 - fp

    return {
        'available': True,
        'predicted': 'FORGED' if fp > 0.5 else 'AUTHENTIC',
        'forged_prob': round(fp * 100, 1),
        'auth_prob': round(ap * 100, 1),
        'confidence': round((fp if fp > 0.5 else ap) * 100, 1),
        'exif_vector': exif_vec,
        'feature_names': EXIF_FEATURE_NAMES,
    }


def simulate_compression_suite(model, pil_img: Image.Image, device=DEVICE) -> dict:
    """Tests the given model across Facebook, Instagram, WhatsApp, and Twitter simulations."""
    results = {}
    for platform_name, compressor in COMPRESSORS.items():
        compressed_img = compressor(pil_img.copy())
        pred = predict_single(model, compressed_img, device=device, generate_heatmap=False)
        results[platform_name] = {
            'predicted': pred['predicted'],
            'confidence': pred['confidence'],
            'forged_prob': pred['forged_prob'],
            'auth_prob': pred['auth_prob'],
            'trust': pred['trust'],
            'compressed_preview': compressed_img
        }
    return results


def extract_images_from_zip_bytes(zip_bytes: bytes, max_files: int = 50):
    """Safely extracts image tuples (name, PIL.Image) from uploaded ZIP in memory."""
    images = []
    allowed_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith('.'):
                continue
            base = os.path.basename(info.filename)
            ext = os.path.splitext(base)[1].lower()
            if ext in allowed_ext:
                try:
                    data = zf.read(info)
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                    images.append((base, img, data))
                except Exception:
                    continue
            if len(images) >= max_files:
                break
    return images
