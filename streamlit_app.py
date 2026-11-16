"""
=============================================================
ForensicAI — Image Forgery Detection System
Streamlit Community Cloud Deployment
=============================================================
Entry point: streamlit_app.py
Run locally: streamlit run streamlit_app.py
=============================================================
"""

import os
import sys
import io
import time
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="ForensicAI — Image Forgery Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core detector engine
import detector_core as dc

# ─────────────────────────────────────────────────────────────
# CUSTOM STYLING (CSS)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric and Card styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    .verdict-authentic {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .verdict-forged {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .verdict-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    .badge-authentic {
        color: #065f46;
        font-weight: 700;
    }
    .badge-forged {
        color: #991b1b;
        font-weight: 700;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stat-box {
        background: #f8fafc;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .platform-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LAZY CACHED MODEL LOADERS
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Hybrid Model (Spatial + Frequency)...")
def get_hybrid_model():
    return dc.load_hybrid_model(dc.DEVICE)

@st.cache_resource(show_spinner="Loading ResNet50 Baseline Model...")
def get_resnet_model():
    return dc.load_resnet_model(dc.DEVICE)

@st.cache_resource(show_spinner="Loading EfficientNet-B4 Model...")
def get_efficientnet_model():
    return dc.load_efficientnet_model(dc.DEVICE)

@st.cache_resource(show_spinner="Loading EXIF-Aware Hybrid V2 Model...")
def get_hybrid_v2_model():
    return dc.load_hybrid_v2_model(dc.DEVICE)


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & SETTINGS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ **ForensicAI**")
    st.caption("Deep Learning Image Forgery Detection")
    st.markdown("---")

    mode = st.radio(
        "Navigation",
        ["🔍 Single Image Forensics",
         "📁 Batch / ZIP Forensics",
         "📊 Model Benchmark & Architecture",
         "ℹ️ System & Cloud Status"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Primary Model Selection")
    selected_model_type = st.selectbox(
        "Detection Model",
        [
            "Hybrid Model (Spatial + Frequency - Recommended)",
            "EfficientNet-B4 (Spatial Only)",
            "ResNet50 (Baseline)",
            "Hybrid V2 (Spatial + Frequency + EXIF)"
        ],
        index=0,
        help="Select which neural network architecture evaluates the image by default."
    )

    st.markdown("---")
    st.markdown("### 💾 Memory & Free Tier Guard")
    device_label = "CUDA GPU" if dc.DEVICE.type == 'cuda' else "CPU"
    st.info(f"Execution Device: **{device_label}**\n\nStreamlit Free Tier budget: ~1 GB RAM. Models are loaded lazily on demand.")

    if st.button("🧹 Clear Memory Cache"):
        st.cache_resource.clear()
        st.success("Model memory cache cleared!")
        st.rerun()


# ─────────────────────────────────────────────────────────────
# HELPER: GET CURRENT PRIMARY MODEL INSTANCE
# ─────────────────────────────────────────────────────────────

def get_active_model(model_choice_str):
    try:
        if "Hybrid V2" in model_choice_str:
            m = get_hybrid_v2_model()
            if m is None:
                st.warning("Hybrid V2 model weights not found or failed to load. Falling back to Hybrid V1.")
                return get_hybrid_model(), "hybrid"
            return m, "hybrid_v2"
        elif "ResNet50" in model_choice_str:
            return get_resnet_model(), "resnet"
        elif "EfficientNet-B4" in model_choice_str:
            return get_efficientnet_model(), "efficientnet"
        else:
            return get_hybrid_model(), "hybrid"
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        st.info("If deploying on Streamlit Cloud, ensure your Hugging Face read token is configured in App Settings -> Secrets.")
        return None, None


# ─────────────────────────────────────────────────────────────
# VIEW 1: SINGLE IMAGE FORENSICS
# ─────────────────────────────────────────────────────────────

if mode == "🔍 Single Image Forensics":
    st.markdown("## 🔍 Single Image Forensics Analysis")
    st.write("Upload an image to inspect for manipulation, splicing, copy-move forgery, and social media compression artifacts.")

    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPEG, PNG, BMP, WEBP"
        )

    with upload_col2:
        st.markdown("**Or load a sample image:**")
        sample_dir = os.path.join(dc.BASE_DIR, 'web_app', 'static', 'uploads')
        sample_options = ["None"]
        if os.path.exists(sample_dir):
            samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            sample_options.extend(samples)

        selected_sample = st.selectbox("Sample image", sample_options, index=0)

    # Resolve target image
    pil_image = None
    image_bytes = None
    filename_display = "uploaded_image"

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))
        filename_display = uploaded_file.name
    elif selected_sample != "None":
        sample_path = os.path.join(sample_dir, selected_sample)
        with open(sample_path, 'rb') as f:
            image_bytes = f.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        filename_display = selected_sample

    if pil_image is not None:
        model, model_key = get_active_model(selected_model_type)

        if model is not None:
            with st.spinner("Analyzing image through deep neural network and frequency filters..."):
                t_start = time.time()
                # Run main inference
                if model_key == "hybrid_v2":
                    pred_v2 = dc.predict_hybrid_v2(model, pil_image, image_bytes)
                    pred = {
                        'predicted': pred_v2['predicted'],
                        'forged_prob': pred_v2['forged_prob'],
                        'auth_prob': pred_v2['auth_prob'],
                        'confidence': pred_v2['confidence'],
                        'trust': 'HIGH RISK' if pred_v2['forged_prob'] >= 70 else ('SUSPICIOUS' if pred_v2['forged_prob'] >= 30 else 'LOW RISK'),
                        'trust_color': '#ef4444' if pred_v2['forged_prob'] >= 70 else ('#f59e0b' if pred_v2['forged_prob'] >= 30 else '#22c55e'),
                        'heatmap_overlay': None,
                        'raw_heatmap': None
                    }
                else:
                    pred = dc.predict_single(model, pil_image, device=dc.DEVICE, generate_heatmap=True, model_name=model_key)

                elapsed_time = round(time.time() - t_start, 2)

            st.markdown("---")

            # TOP DASHBOARD ROW: Image View & Verdict Card
            col_img, col_verdict = st.columns([1, 1.3])

            with col_img:
                st.markdown("#### 🖼️ Source Image")
                st.image(pil_image, caption=f"{filename_display} ({pil_image.width}×{pil_image.height} px)", use_container_width=True)

            with col_verdict:
                st.markdown("#### ⚖️ Forensic Verdict")
                is_forged = pred['predicted'] == 'FORGED'

                if is_forged:
                    st.markdown(f"""
                    <div class="verdict-forged">
                        <div class="verdict-title badge-forged">⚠️ FORGED IMAGE DETECTED</div>
                        <p style="color: #7f1d1d; font-size: 1.1rem; margin-bottom: 12px;">
                            Manipulated or spliced regions detected in pixel & frequency domains.
                        </p>
                        <span class="risk-badge" style="background-color: {pred['trust_color']}; color: white;">
                            RISK LEVEL: {pred['trust']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-authentic">
                        <div class="verdict-title badge-authentic">✅ AUTHENTIC IMAGE</div>
                        <p style="color: #064e3b; font-size: 1.1rem; margin-bottom: 12px;">
                            Consistent spatial patterns and uniform frequency spectrum verified.
                        </p>
                        <span class="risk-badge" style="background-color: {pred['trust_color']}; color: white;">
                            RISK LEVEL: {pred['trust']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                # Probabilities & Confidence
                st.markdown(f"**Confidence Score:** `{pred['confidence']}%` (Inference time: `{elapsed_time}s`)")

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Forgery Probability", f"{pred['forged_prob']}%")
                    st.progress(pred['forged_prob'] / 100.0)
                with col_p2:
                    st.metric("Authentic Probability", f"{pred['auth_prob']}%")
                    st.progress(pred['auth_prob'] / 100.0)

            # DEEP DIVE TABS
            st.markdown("---")
            tab_gcam, tab_exif, tab_comp, tab_compare = st.tabs([
                "🔬 Grad-CAM Explainability",
                "🏷️ EXIF Metadata Forensics",
                "📱 Compression Robustness",
                "⚖️ Three-Model Comparison"
            ])

            # ── TAB 1: GRAD-CAM ──
            with tab_gcam:
                st.markdown("### Explainable AI: Grad-CAM Localization")
                st.write(
                    "Grad-CAM (Gradient-weighted Class Activation Mapping) calculates the gradient "
                    "of the predicted class score with respect to the final convolutional feature maps of the spatial backbone. "
                    "Bright red and yellow areas highlight suspected manipulated regions or tampering seams."
                )

                if pred['heatmap_overlay'] is not None:
                    gcam_col1, gcam_col2 = st.columns(2)
                    with gcam_col1:
                        st.markdown("**Original Image (Resized 224×224)**")
                        st.image(pil_image.resize((224, 224)), use_container_width=True)
                    with gcam_col2:
                        st.markdown("**Grad-CAM Heatmap Overlay**")
                        st.image(pred['heatmap_overlay'], use_container_width=True)
                else:
                    st.info("Grad-CAM visualization is generated for models with spatial convolutional backbones (Hybrid and EfficientNet). Switch to Hybrid Model in the sidebar to view heatmap overlays.")

            # ── TAB 2: EXIF METADATA ──
            with tab_exif:
                st.markdown("### EXIF Metadata Analysis & Integrity Checks")
                st.write("Extracts embedded camera metadata, software tags, timestamps, and GPS information to cross-reference forensic consistency.")

                exif_dict = dc.get_detailed_exif_dict(pil_image)

                if exif_dict['flags']:
                    st.warning("⚠️ **Forensic Metadata Alerts:**\n- " + "\n- ".join(exif_dict['flags']))
                else:
                    st.success("✅ No suspicious metadata flags detected.")

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("**Device & Capture Details**")
                    st.write(f"- **Camera Make:** {exif_dict['make']}")
                    st.write(f"- **Camera Model:** {exif_dict['model']}")
                    st.write(f"- **Resolution:** {exif_dict['resolution']}")
                    st.write(f"- **GPS Info:** {exif_dict['gps']}")

                with col_e2:
                    st.markdown("**Software & Timestamps**")
                    st.write(f"- **Software:** {exif_dict['software']}")
                    st.write(f"- **Original Date/Time:** {exif_dict['datetime_original']}")
                    st.write(f"- **Modified Date/Time:** {exif_dict['datetime_modified']}")
                    st.write(f"- **EXIF Block Present:** {'Yes' if exif_dict['has_exif'] else 'No (Stripped)'}")

                st.markdown("#### 16-Dimensional Normalized Forensic Vector")
                vec = dc.extract_exif_features(image_bytes)
                df_vec = pd.DataFrame({
                    "Feature Name": dc.EXIF_FEATURE_NAMES,
                    "Normalized Value": vec
                })
                st.dataframe(df_vec, use_container_width=True, height=260)

            # ── TAB 3: SOCIAL MEDIA COMPRESSION SIMULATION ──
            with tab_comp:
                st.markdown("### Social Media Compression Simulation Suite")
                st.write(
                    "Social media platforms heavily re-encode images upon upload, destroying subtle pixel artifacts. "
                    "Our system tests how the image holds up under simulated compression algorithms from 4 major platforms."
                )

                with st.spinner("Simulating multi-platform compression and running inference..."):
                    comp_results = dc.simulate_compression_suite(model, pil_image, device=dc.DEVICE)

                comp_cols = st.columns(4)
                platform_info = [
                    ("Facebook", "facebook", "Max 2048px | Sharpness 1.2 | Q=75"),
                    ("Instagram", "instagram", "Max 1080px | Lanczos | Q=78"),
                    ("WhatsApp", "whatsapp", "Max 1600px | 2-Stage Q=60,55"),
                    ("Twitter", "twitter", "Max 1200px | Lanczos | Q=82"),
                ]

                for idx, (display_name, key, spec) in enumerate(platform_info):
                    res = comp_results[key]
                    with comp_cols[idx]:
                        st.markdown(f"""
                        <div class="platform-card">
                            <h4 style="margin: 0;">{display_name}</h4>
                            <p style="font-size: 0.75rem; color: #64748b; margin-bottom: 8px;">{spec}</p>
                            <h3 style="color: {'#ef4444' if res['predicted'] == 'FORGED' else '#10b981'}; margin: 4px 0;">
                                {res['predicted']}
                            </h3>
                            <p style="font-size: 0.85rem; font-weight: 600; margin: 0;">
                                Confidence: {res['confidence']}%
                            </p>
                            <span class="risk-badge" style="background-color: {'#ef4444' if res['predicted'] == 'FORGED' else '#22c55e'}; color: white; margin-top: 6px;">
                                {res['trust']}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(res['compressed_preview'], use_container_width=True)

            # ── TAB 4: THREE-MODEL COMPARISON ──
            with tab_compare:
                st.markdown("### Cross-Model Comparison: Baseline vs Improved vs Hybrid")
                st.write(
                    "Compare predictions across all 3 trained architectures. "
                    "To conserve memory on free cloud hosting, the secondary models are loaded only when requested."
                )

                if st.button("🚀 Run Multi-Model Comparison on this Image"):
                    with st.spinner("Loading models and evaluating..."):
                        m_resnet = get_resnet_model()
                        m_eff = get_efficientnet_model()
                        m_hyb = get_hybrid_model()

                        r_res = dc.predict_single(m_resnet, pil_image, device=dc.DEVICE)
                        r_eff = dc.predict_single(m_eff, pil_image, device=dc.DEVICE)
                        r_hyb = dc.predict_single(m_hyb, pil_image, device=dc.DEVICE)

                        comp_table = pd.DataFrame([
                            {
                                "Model": "ResNet50 (Baseline)",
                                "Backbone": "ResNet50",
                                "Features": "Spatial Only (Clean Trained)",
                                "Predicted": r_res['predicted'],
                                "Confidence": f"{r_res['confidence']}%",
                                "Forgery Prob": f"{r_res['forged_prob']}%",
                                "Risk Score": r_res['trust']
                            },
                            {
                                "Model": "EfficientNet-B4",
                                "Backbone": "EfficientNet-B4",
                                "Features": "Spatial Only (Augmented)",
                                "Predicted": r_eff['predicted'],
                                "Confidence": f"{r_eff['confidence']}%",
                                "Forgery Prob": f"{r_eff['forged_prob']}%",
                                "Risk Score": r_eff['trust']
                            },
                            {
                                "Model": "Hybrid Model (Ours)",
                                "Backbone": "EfficientNet-B4 + Laplacian CNN",
                                "Features": "Spatial (1792) + Frequency (256)",
                                "Predicted": r_hyb['predicted'],
                                "Confidence": f"{r_hyb['confidence']}%",
                                "Forgery Prob": f"{r_hyb['forged_prob']}%",
                                "Risk Score": r_hyb['trust']
                            }
                        ])

                        st.dataframe(comp_table, use_container_width=True)

                st.markdown("#### 📈 Verified Research Benchmark Comparison")
                st.caption("Standardized test set performance under simulated social media compression (from results.json):")

                benchmark_df = pd.DataFrame([
                    {"Model": "ResNet50 (Baseline)", "Clean Acc": "82.1%", "Facebook": "71.8%", "Instagram": "71.4%", "WhatsApp": "67.1%", "Twitter": "71.8%", "Avg Drop": "-14.1%"},
                    {"Model": "EfficientNet-B4", "Clean Acc": "77.0%", "Facebook": "75.0%", "Instagram": "74.5%", "WhatsApp": "74.9%", "Twitter": "74.4%", "Avg Drop": "-2.3%"},
                    {"Model": "Hybrid Model (Ours)", "Clean Acc": "78.2%", "Facebook": "75.0%", "Instagram": "76.0%", "WhatsApp": "75.2%", "Twitter": "76.5%", "Avg Drop": "-1.7% (Most Robust)"}
                ])
                st.dataframe(benchmark_df, use_container_width=True)
    else:
        st.info("👆 Please upload an image or choose a sample image above to start analysis.")


# ─────────────────────────────────────────────────────────────
# VIEW 2: BATCH / ZIP FORENSICS
# ─────────────────────────────────────────────────────────────

elif mode == "📁 Batch / ZIP Forensics":
    st.markdown("## 📁 Batch & ZIP Image Forensics")
    st.write("Upload multiple images or a single ZIP archive to run high-throughput forgery screening in memory.")

    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        batch_type = st.radio("Upload Method", ["Multiple Images", "ZIP Archive (.zip)"], horizontal=True)

        if batch_type == "Multiple Images":
            batch_files = st.file_uploader(
                "Choose images for batch analysis",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True
            )
        else:
            zip_file = st.file_uploader(
                "Upload a ZIP archive containing images",
                type=["zip"]
            )
            batch_files = None

    with b_col2:
        max_batch_count = st.slider("Max images to process", 5, 50, 20, help="Prevents RAM exhaustion on free hosting.")

    # Process batch
    items_to_process = []

    if batch_type == "Multiple Images" and batch_files:
        for f in batch_files[:max_batch_count]:
            try:
                img_bytes = f.getvalue()
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                items_to_process.append((f.name, img, img_bytes))
            except Exception:
                continue

    elif batch_type == "ZIP Archive (.zip)" and zip_file:
        with st.spinner("Extracting images from ZIP in memory..."):
            items_to_process = dc.extract_images_from_zip_bytes(zip_file.getvalue(), max_files=max_batch_count)

    if items_to_process:
        st.write(f"Loaded **{len(items_to_process)}** images ready for analysis.")
        if st.button("🚀 Execute Batch Forensic Analysis"):
            model, model_name = get_active_model(selected_model_type)

            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, img, raw_bytes) in enumerate(items_to_process):
                status_text.text(f"Processing ({i+1}/{len(items_to_process)}): {name}")
                pred = dc.predict_single(model, img, device=dc.DEVICE, generate_heatmap=False, model_name=model_name)
                results_list.append({
                    "Filename": name,
                    "Predicted": pred['predicted'],
                    "Confidence": f"{pred['confidence']}%",
                    "Forgery Prob (%)": pred['forged_prob'],
                    "Authentic Prob (%)": pred['auth_prob'],
                    "Risk Level": pred['trust'],
                })
                progress_bar.progress((i + 1) / len(items_to_process))

            status_text.text(f"Batch completed: {len(items_to_process)} images evaluated.")

            # Summary Metrics
            df_batch = pd.DataFrame(results_list)
            num_forged = (df_batch["Predicted"] == "FORGED").sum()
            num_auth = (df_batch["Predicted"] == "AUTHENTIC").sum()
            avg_conf = df_batch["Forgery Prob (%)"].mean()

            st.markdown("### 📊 Batch Summary")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Total Analyzed", len(df_batch))
            m_col2.metric("Authentic Images", num_auth)
            m_col3.metric("Forged Images", num_forged)
            m_col4.metric("Avg Forgery Risk", f"{round(avg_conf, 1)}%")

            st.markdown("### 📋 Results Table")
            st.dataframe(df_batch, use_container_width=True)

            # Export CSV
            csv_buf = io.StringIO()
            df_batch.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buf.getvalue(),
                file_name="forensic_batch_results.csv",
                mime="text/csv"
            )


# ─────────────────────────────────────────────────────────────
# VIEW 3: MODEL BENCHMARK & ARCHITECTURE
# ─────────────────────────────────────────────────────────────

elif mode == "📊 Model Benchmark & Architecture":
    st.markdown("## 📊 Research Architecture & Evaluation")
    st.write(
        "A deep learning framework specifically designed for detecting image forgeries that have undergone "
        "social media compression across platforms like Facebook, Instagram, WhatsApp, and Twitter."
    )

    st.markdown("### 🏗️ Architecture: Hybrid Dual-Branch Network")
    st.markdown("""
    ```
                      ┌───────────────────────────────────────┐
                      │          Input Image (224×224)        │
                      └──────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         ┌─────────────────────┐                   ┌─────────────────────┐
         │   Spatial Branch    │                   │  Frequency Branch   │
         │  (EfficientNet-B4)  │                   │ (Laplacian Filtering│
         │                     │                   │      + 4-layer CNN) │
         └──────────┬──────────┘                   └──────────┬──────────┘
                    │ 1792-dim                                │ 256-dim
                    └────────────────────┬────────────────────┘
                                         ▼
                            ┌────────────────────────┐
                            │ Feature Fusion (2048)  │
                            └───────────┬────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │ Classifier Head (512)  │
                            └───────────┬────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │ Softmax [Authentic,    │
                            │          Forged]       │
                            └────────────────────────┘
    ```
    """)

    st.markdown("### 🔬 Why the Hybrid Approach Wins Under Compression")
    st.markdown("""
    - **Spatial Artifact Destruction:** Standard CNN backbones (like ResNet50) rely primarily on pixel-level edges, chromatic seams, and noise boundaries. When social media platforms resize and apply lossy JPEG compression, these spatial cues get heavily degraded, causing ResNet50 accuracy to collapse from **82.1%** down to **67.1%** on WhatsApp.
    - **Frequency-Domain Resiliency:** The frequency branch applies a fixed Laplacian operator $k = [[0,1,0],[1,-4,1],[0,1,0]]$ that suppresses image content and emphasizes high-frequency energy. This exposes differences in camera sensor noise (PRNU) and non-aligned secondary JPEG grid blocks.
    - **Minimal Degradation:** The Hybrid model exhibits an average accuracy drop of only **1.7%** across all platforms, compared to **14.1%** for the standard baseline.
    """)

    st.markdown("### 📑 Full Evaluation Matrix (Verified from evaluation/results/results.json)")
    bench_data = {
        "Metric": ["Clean Accuracy", "Facebook Acc", "Instagram Acc", "WhatsApp Acc", "Twitter Acc", "Average Drop", "Clean F1", "Clean Precision", "Clean Recall"],
        "ResNet50 Baseline": ["82.1%", "71.8%", "71.4%", "67.1%", "71.8%", "-14.1%", "83.1%", "78.7%", "88.0%"],
        "EfficientNet-B4": ["77.0%", "75.0%", "74.5%", "74.9%", "74.4%", "-2.3%", "77.5%", "76.0%", "79.0%"],
        "Hybrid Model (Ours)": ["78.2%", "75.0%", "76.0%", "75.2%", "76.5%", "-1.7% (Best)", "78.9%", "76.5%", "81.5%"]
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# VIEW 4: SYSTEM & CLOUD STATUS
# ─────────────────────────────────────────────────────────────

elif mode == "ℹ️ System & Cloud Status":
    st.markdown("## ℹ️ Deployment Environment & Cloud Hosting Status")
    st.write("System health, dependency versions, and cloud configuration check.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📦 Environment Versions")
        st.write(f"- **Python Version:** `{sys.version.split()[0]}`")
        st.write(f"- **PyTorch:** `{dc.torch.__version__}`")
        st.write(f"- **Streamlit:** `{st.__version__}`")
        st.write(f"- **Pillow:** `{Image.__version__}`")
        st.write(f"- **OpenCV:** `{dc.cv2.__version__}`")
        st.write(f"- **Compute Device:** `{dc.DEVICE}`")

    with col2:
        st.markdown("### ☁️ Hugging Face & Checkpoints Status")
        token = dc.get_hf_token()
        st.write(f"- **HF_TOKEN Configured:** `{'Yes' if token else 'No (Not configured)'}`")
        st.write(f"- **HF Model Repo:** `{dc.HF_REPO_ID}`")

        model_files = [
            "resnet_baseline.pth",
            "efficientnet_model.pth",
            "hybrid_model.pth",
            "hybrid_model_v2_exif.pth"
        ]
        st.markdown("**Local Checkpoint Status:**")
        for f in model_files:
            pth = os.path.join(dc.MODELS_DIR, f)
            exists = os.path.exists(pth) and os.path.getsize(pth) > 0
            size_mb = round(os.path.getsize(pth) / (1024*1024), 1) if exists else 0
            status_icon = "✅ Present" if exists else "☁️ Requires Download"
            st.write(f"- `{f}`: {status_icon} ({size_mb} MB)")

    st.markdown("---")
    st.markdown("### 📋 Streamlit Community Cloud Free Hosting Setup Instructions")
    st.markdown("""
    1. **Fork/Push repository to GitHub:** `https://github.com/DarshanSS018/Image-Forgery-Detection-Using-Deep-Learning-Techniques`
    2. **Log in to [share.streamlit.io](https://share.streamlit.io/)** with your GitHub account.
    3. **Click 'New app':**
       - **Repository:** `DarshanSS018/Image-Forgery-Detection-Using-Deep-Learning-Techniques`
       - **Branch:** `main`
       - **Main file path:** `streamlit_app.py`
    4. **Add Secret in Advanced Settings:**
       - Click **Advanced Settings** -> **Secrets**
       - Add: `HF_TOKEN = "hf_your_token_here"`
    5. **Deploy!**
    """)
