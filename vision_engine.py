"""
AgriShield AI — Vision Engine
Computer Vision & Deep Learning pipeline for crop disease diagnostics.

Pipeline:
  1. Load & normalise image (PIL → NumPy)
  2. RGB → HSV colour-space conversion for chlorosis quantification
  3. Contour / edge detection for necrotic lesion density
  4. Composite severity scoring (0–100 %)
  5. Optional: MobileNetV3 backbone feature extraction (CPU/GPU, graceful fallback)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── optional deep-learning backbone ──────────────────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.info("PyTorch not found — deep feature extraction disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Data contract returned to the caller
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionReport:
    chlorosis_pct: float          # 0.0 – 100.0
    lesion_count: int             # raw contour count
    lesion_density: float         # lesions per 1 000 px²
    severity_score: float         # composite 0–100
    severity_label: str           # Healthy / Mild / Moderate / Severe / Critical
    annotated_image: Image.Image  # PIL image with visual overlays
    deep_features: Optional[list[float]] = field(default=None, repr=False)
    backbone_used: str = "none"


# ─────────────────────────────────────────────────────────────────────────────
# Colour-space helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert H×W×3 uint8 RGB array to H×W×3 float32 HSV (H: 0-180, S/V: 0-255)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


def compute_chlorosis(rgb: np.ndarray) -> float:
    """
    Detect chlorosis (yellowing) via HSV thresholding.

    Yellow hue band  : H ∈ [20, 40] in OpenCV (0-180 scale)
    Saturation guard : S > 40  (avoids pale/white noise)
    Value guard      : V > 50  (avoids shadow regions)

    Returns percentage of leaf pixels that fall within the yellow band.
    """
    hsv = _rgb_to_hsv_array(rgb)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Isolate the leaf area (exclude very dark / near-white background)
    leaf_mask = (v > 40) & (s > 20)
    leaf_pixels = int(leaf_mask.sum())
    if leaf_pixels == 0:
        return 0.0

    yellow_mask = (h >= 18) & (h <= 42) & (s > 40) & (v > 50) & leaf_mask
    yellow_pixels = int(yellow_mask.sum())

    return round(min(yellow_pixels / leaf_pixels * 100, 100.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Necrotic lesion detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_lesions(rgb: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Detect necrotic lesions via edge + contour analysis.

    Steps:
      1. Convert to grayscale → Gaussian blur → Canny edge detection
      2. Dilate edges to close small gaps
      3. Find external contours; filter by area to remove noise
      4. Compute density = count / (image_area / 1000)

    Returns (count, density_per_1k_px, annotated_bgr_array).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=40, threshold2=120)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: keep contours whose area is between 30 and 5 000 px² (leaf features)
    valid = [c for c in contours if 30 < cv2.contourArea(c) < 5000]

    h, w = rgb.shape[:2]
    image_area_k = (h * w) / 1000.0
    density = round(len(valid) / image_area_k, 4) if image_area_k > 0 else 0.0

    # Draw contours on a copy for annotation
    annotated_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.drawContours(annotated_bgr, valid, -1, (0, 0, 220), 2)  # red in BGR

    return len(valid), density, annotated_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Composite severity scoring
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_THRESHOLDS = [
    (0,  15,  "Healthy"),
    (15, 35,  "Mild"),
    (35, 55,  "Moderate"),
    (55, 75,  "Severe"),
    (75, 101, "Critical"),
]


def compute_severity(chlorosis_pct: float, lesion_density: float) -> tuple[float, str]:
    """
    Composite score = 0.55 × chlorosis_pct + 0.45 × min(lesion_density × 10, 100).
    Weights reflect that chlorosis is the primary early-warning signal.
    """
    lesion_component = min(lesion_density * 10, 100.0)
    score = round(0.55 * chlorosis_pct + 0.45 * lesion_component, 2)
    score = min(score, 100.0)

    label = "Unknown"
    for lo, hi, lbl in _SEVERITY_THRESHOLDS:
        if lo <= score < hi:
            label = lbl
            break

    return score, label


# ─────────────────────────────────────────────────────────────────────────────
# Optional: MobileNetV3 backbone feature extraction
# ─────────────────────────────────────────────────────────────────────────────

_BACKBONE: Optional[object] = None  # lazy-loaded singleton


def _get_backbone():
    global _BACKBONE
    if not _TORCH_AVAILABLE:
        return None
    if _BACKBONE is None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BACKBONE = model.to(device)
    return _BACKBONE


def extract_deep_features(pil_image: Image.Image) -> Optional[list[float]]:
    """
    Run image through MobileNetV3 backbone to extract a 576-d feature vector
    from the avgpool layer via a forward hook.  Returns None if PyTorch is absent.
    """
    backbone = _get_backbone()
    if backbone is None:
        return None

    device = next(backbone.parameters()).device
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()

    tensor = transform(pil_image).unsqueeze(0).to(device)

    features: list[np.ndarray] = []

    def _hook(_, __, output):
        features.append(output.detach().cpu().numpy())

    handle = backbone.avgpool.register_forward_hook(_hook)  # type: ignore[attr-defined]
    with torch.no_grad():
        backbone(tensor)
    handle.remove()

    if features:
        return features[0].flatten().tolist()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Annotation overlay
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_image(
    base_bgr: np.ndarray,
    chlorosis_pct: float,
    lesion_count: int,
    severity_score: float,
    severity_label: str,
) -> Image.Image:
    """Burn diagnostic metrics as a HUD overlay onto the annotated BGR image."""
    img_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    lines = [
        f"Chlorosis : {chlorosis_pct:.1f}%",
        f"Lesions   : {lesion_count}",
        f"Severity  : {severity_score:.1f}% ({severity_label})",
    ]

    colour_map = {
        "Healthy": (34, 197, 94),
        "Mild": (234, 179, 8),
        "Moderate": (249, 115, 22),
        "Severe": (239, 68, 68),
        "Critical": (185, 28, 28),
    }
    hud_colour = colour_map.get(severity_label, (255, 255, 255))

    x, y = 10, 10
    padding = 4
    line_h = 18
    box_h = len(lines) * line_h + padding * 2
    box_w = 250
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 160))

    for i, line in enumerate(lines):
        draw.text((x + padding, y + padding + i * line_h), line, fill=hud_colour)

    return pil


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(image_bytes: bytes) -> VisionReport:
    """
    Full vision pipeline entry-point.

    Args:
        image_bytes: Raw bytes of an uploaded JPG/PNG image.

    Returns:
        VisionReport dataclass with all computed metrics and annotated image.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to a consistent working resolution (keeps aspect ratio)
    max_dim = 640
    pil_image.thumbnail((max_dim, max_dim), Image.LANCZOS)

    rgb = np.array(pil_image, dtype=np.uint8)

    chlorosis_pct = compute_chlorosis(rgb)
    lesion_count, lesion_density, annotated_bgr = compute_lesions(rgb)
    severity_score, severity_label = compute_severity(chlorosis_pct, lesion_density)

    annotated_pil = _annotate_image(
        annotated_bgr, chlorosis_pct, lesion_count, severity_score, severity_label
    )

    deep_features = extract_deep_features(pil_image)
    backbone_used = "MobileNetV3-Small" if deep_features is not None else "none"

    return VisionReport(
        chlorosis_pct=chlorosis_pct,
        lesion_count=lesion_count,
        lesion_density=lesion_density,
        severity_score=severity_score,
        severity_label=severity_label,
        annotated_image=annotated_pil,
        deep_features=deep_features,
        backbone_used=backbone_used,
    )
