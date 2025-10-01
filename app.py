# Streamlit demo for CXR pneumonia flagger (research-only)
# --------------------------------------------------------
# Run:  streamlit run app.py
#
# Fixed folders:
#   outputs/         -> best.ckpt (preferred) or model_traced.pt, plus optional curves/conf mats/metrics.json
#   sample_images/   -> small, fixed gallery of public/de-identified CXRs (no uploads)

import os, json, time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import streamlit as st


# -----------------------------
# Model defs (same as your training script)
# -----------------------------
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.pool(x)
            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.pool(x)
            flattened = x.view(1, -1).shape[1]
        self.fc = nn.Linear(flattened, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))); x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x))); x = self.dropout(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool(x); x = self.dropout(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.pool(x); x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_model(name: str, num_classes=2, pretrained=True):
    name = name.lower()
    if name == "deeper":
        return DeeperCNN(num_classes=num_classes)
    elif name == "resnet18":
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name} (use 'deeper' or 'resnet18')")


# -----------------------------
# Preprocess (match eval transform)
# -----------------------------
def get_eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def predict_one(model, image: Image.Image, device, transform, class_names):
    x = transform(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pos_idx = 1 if (len(class_names) >= 2 and class_names[1].lower().startswith('pneum')) else int(np.argmax(probs))
    p_pos = float(probs[pos_idx])
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if class_names else str(pred_idx)
    return p_pos, pred_label, probs, x


def load_model_from_outputs(outputs_dir: Path, device):
    best_ckpt = outputs_dir / "best.ckpt"
    traced_pt = outputs_dir / "model_traced.pt"

    class_names = ['NORMAL', 'PNEUMONIA']
    model = None
    arch = "deeper"
    loaded_from = None

    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu")
        args = state.get("args", {})
        arch = args.get("model", "deeper")
        class_names = state.get("class_names", class_names)
        model = build_model(arch, num_classes=len(class_names), pretrained=(arch=="resnet18"))
        model.load_state_dict(state["model"])
        loaded_from = str(best_ckpt.name)
    elif traced_pt.exists():
        model = torch.jit.load(str(traced_pt), map_location="cpu")
        loaded_from = str(traced_pt.name)
    else:
        raise FileNotFoundError("No model found in outputs/ (best.ckpt or model_traced.pt).")

    model.eval().to(device)
    return model, class_names, arch, loaded_from


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Chest X-Ray Classification Demo", layout="wide")
st.title("🧪 Chest X-Ray Classification CNN — Research Demo (Not for Clinical Use)")

# Fixed locations
outputs_dir = Path("outputs")
gallery_dir = Path("sample_images")

# Device (auto; no UI control)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load model
try:
    t0 = time.time()
    model, class_names, arch, loaded_from = load_model_from_outputs(outputs_dir, device)
    load_ms = int((time.time() - t0) * 1000)
    st.success(f"Model loaded from `{loaded_from}` (arch: {arch}) on **{device}** in {load_ms} ms. Classes: {class_names}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load gallery file list
if not gallery_dir.exists():
    st.warning(f"Gallery folder `{gallery_dir}` not found. Create it and add public sample CXRs (JPG/PNG).")
    st.stop()
image_paths = sorted([p for p in gallery_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if not image_paths:
    st.warning(f"No images found in `{gallery_dir}`. Add a few JPG/PNG files (public/de-identified).")
    st.stop()

# -----------------------------
# Settings (TOP, not sidebar)
# -----------------------------
st.markdown("### Settings")
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    preset = st.radio(
        "Threshold preset (τ)",
        options=["Sensitive (τ=0.50)", "Balanced (τ≈0.999)", "Custom"],
        index=1,
        horizontal=True
    )

with c2:
    # Build list for selector
    img_options = [p.relative_to(gallery_dir).as_posix() for p in image_paths]
    sel_name = st.selectbox("Select an example image", options=img_options)

# Custom threshold slider (if chosen)
if preset == "Custom":
    tau = st.slider("Custom τ", min_value=0.0, max_value=1.0, value=0.999, step=0.001)
elif preset.startswith("Sensitive"):
    tau = 0.50
else:
    tau = 0.999

st.divider()

# Resolve selected image path
sel_path = gallery_dir / sel_name

# Layout for image + prediction
col_img, col_out = st.columns([1.1, 1.0], gap="large")

with col_img:
    st.subheader("Selected image")
    img = Image.open(sel_path).convert("RGB")
    st.image(img, use_column_width=True, caption=sel_name)

with col_out:
    st.subheader("Prediction")
    transform = get_eval_transform()
    p_pos, pred_label, probs, _ = predict_one(model, img, device, transform, class_names)

    positive_name = class_names[1] if len(class_names) > 1 else "Positive"
    negative_name = class_names[0] if len(class_names) > 0 else "Negative"
    decision = positive_name if p_pos >= tau else negative_name

    st.metric(label=f"Model probability: {positive_name}", value=f"{p_pos:.3f}")
    st.write(f"**Threshold (τ) = {tau:.3f}** → Predict **{positive_name}** if probability ≥ τ; else **{negative_name}**.")
    st.markdown(f"**Decision:** `{decision}`")

    st.write("Class probabilities:")
    prob_rows = {cn: f"{probs[i]:.3f}" for i, cn in enumerate(class_names)}
    st.table(prob_rows)

    st.caption("This is a research demo. Not a medical device and not for diagnostic use.")

st.divider()

# Metrics & curves
st.subheader("Metrics & Curves (from your training run)")
curve_cols = st.columns(2)
roc_path = outputs_dir / "roc_curve.png"
pr_path = outputs_dir / "pr_curve.png"
with curve_cols[0]:
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curve (ranking quality across thresholds)", use_column_width=True)
    else:
        st.info("Put `roc_curve.png` in outputs/ to display.")
with curve_cols[1]:
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision–Recall Curve (class-imbalance friendly)", use_column_width=True)
    else:
        st.info("Put `pr_curve.png` in outputs/ to display.")

cm_cols = st.columns(2)
cm_default = outputs_dir / "confusion_matrix_default.png"
cm_best = outputs_dir / "confusion_matrix_bestF1.png"
with cm_cols[0]:
    if cm_default.exists():
        st.image(str(cm_default), caption="Confusion Matrix @ τ=0.50 (Sensitive)", use_column_width=True)
    else:
        st.info("Put `confusion_matrix_default.png` in outputs/ to display.")
with cm_cols[1]:
    if cm_best.exists():
        st.image(str(cm_best), caption="Confusion Matrix @ Best-F1 τ", use_column_width=True)
    else:
        st.info("Put `confusion_matrix_bestF1.png` in outputs/ to display.")

# metrics.json (optional)
metrics_json = outputs_dir / "metrics.json"
if metrics_json.exists():
    st.subheader("Summary metrics (from metrics.json)")
    try:
        data = json.loads(metrics_json.read_text())
        st.json(data)
    except Exception as e:
        st.info(f"metrics.json present but could not be parsed: {e}")

st.divider()
with st.expander("Model card & notes"):
    st.markdown("""
**Intended use is research**  
Demonstration of how operating thresholds affect precision/recall for chest X-ray pneumonia detection.

**Training & evaluation**  
- Architectures: DeeperCNN or ResNet18.  
- Metrics: AUROC/PR AUC (ranking); precision/recall/F1 at chosen thresholds.  
- Thresholds: Sensitive (τ=0.50) vs Balanced (τ≈0.999).  
- Recommend probability calibration (e.g., temperature scaling) before any real-world use.

**Limitations**  
Test-set results may not reflect real-world prevalence or domain shifts.

**Contact**  
Ping me on LinkedIn/email if you want to collaborate or pilot.
""")

st.caption("© Research demo. Built for transparency and discussion, not diagnosis.")
