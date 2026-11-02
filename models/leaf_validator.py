"""
Leaf Verification and Out-of-Distribution (OOD) Gatekeeper.
Ensures that uploaded images are authentic plant leaves before passing them
to the 38-class plant disease classification neural network.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

def compute_leaf_metrics(pil_image: Image.Image) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Computes botanical chromaticity, foliage coverage, and texture roughness.
    
    Returns:
        (is_leaf, foliage_ratio, reason_str, detailed_metrics)
    """
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total_pixels = 128 * 128

    # 1. Grayscale / monochrome check
    rg_diff = float(np.abs(r - g).mean())
    gb_diff = float(np.abs(g - b).mean())
    if rg_diff < 5.0 and gb_diff < 5.0:
        return False, 0.0, "Monochromatic or grayscale image detected. Disease diagnosis requires full-spectrum RGB plant images.", {
            "grayscale": True
        }

    # 2. Vectorized RGB to HSV conversion
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c + 1e-6

    v = max_c / 255.0
    s = np.where(max_c > 0, delta / (max_c + 1e-6), 0)

    h = np.zeros_like(r)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    h = h * 60.0

    # 3. Botanical Foliage Masks:
    # A) Green leaf tissue: Hue 35 to 155 deg, S > 0.15, V > 0.12
    green_leaf = (h >= 35) & (h <= 155) & (s >= 0.15) & (v >= 0.12) & (v <= 0.95)

    # B) Diseased / chlorotic / necrotic lesion tissue:
    # Yellowing / orange-brown rust / brown blight: Hue 10 to 35 deg, S >= 0.20, V between 0.12 and 0.85
    necrotic_leaf = (h >= 10) & (h < 35) & (s >= 0.20) & (v >= 0.12) & (v <= 0.85)

    # C) Dark necrotic spots / scab: Low V (< 0.25) with organic warm hue
    dark_necrotic = (h >= 10) & (h <= 50) & (s >= 0.15) & (v < 0.25) & (v > 0.04)

    foliage_mask = green_leaf | necrotic_leaf | dark_necrotic
    foliage_ratio = float(np.sum(foliage_mask) / total_pixels)

    # 4. Spatial texture & Noise check:
    # Synthetic random noise has extreme adjacent pixel fluctuations
    diff_h = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    diff_v = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    spatial_roughness = float((diff_h + diff_v) / 2.0)

    if spatial_roughness > 50.0 and foliage_ratio < 0.50:
        return False, foliage_ratio, f"Image lacks organic leaf texture (High-frequency noise detected, roughness={spatial_roughness:.1f})", {
            "spatial_roughness": spatial_roughness,
            "foliage_ratio": foliage_ratio
        }

    # Lower-bound texture check: reject smooth surfaces (green cars, walls)
    if spatial_roughness < 3.0 and foliage_ratio < 0.50:
        return False, foliage_ratio, f"Image surface too smooth for organic leaf tissue (roughness={spatial_roughness:.1f}). Likely a man-made object.", {
            "spatial_roughness": spatial_roughness,
            "foliage_ratio": foliage_ratio
        }

    # 5. Non-botanical color dominance check (blue, purple, magenta objects)
    inorganic_mask = (h >= 180) & (h <= 330) & (s >= 0.25) & (v >= 0.15)
    inorganic_ratio = float(np.sum(inorganic_mask) / total_pixels)

    if inorganic_ratio > 0.45 and foliage_ratio < 0.25:
        return False, foliage_ratio, f"Inorganic non-botanical colors dominate the image ({inorganic_ratio*100:.1f}% cyan/blue/magenta)", {
            "inorganic_ratio": inorganic_ratio,
            "foliage_ratio": foliage_ratio
        }

    # 6. Minimum foliage threshold
    if foliage_ratio < 0.20:
        return False, foliage_ratio, f"Insufficient plant foliage detected ({foliage_ratio*100:.1f}% foliage coverage; minimum 20% required for reliable diagnosis)", {
            "foliage_ratio": foliage_ratio
        }

    return True, foliage_ratio, f"Valid plant foliage confirmed ({foliage_ratio*100:.1f}% coverage)", {
        "foliage_ratio": foliage_ratio,
        "spatial_roughness": spatial_roughness
    }

def validate_leaf_image(
    pil_image: Image.Image,
    model_predictions: np.ndarray = None,
    min_model_confidence: float = 0.40
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Dual-stage gatekeeper:
    Stage 1: Botanical Chromatic & Texture Analysis
    Stage 2: Model Confidence & Entropy Verification
    """
    is_leaf, foliage_ratio, msg, meta = compute_leaf_metrics(pil_image)
    if not is_leaf:
        return False, msg, meta

    # Stage 2: Softmax confidence verification
    if model_predictions is not None:
        max_prob = float(np.max(model_predictions))
        probs = np.clip(model_predictions, 1e-12, 1.0)
        entropy = -float(np.sum(probs * np.log(probs)))
        meta["max_confidence"] = max_prob
        meta["entropy"] = entropy

        if max_prob < min_model_confidence and foliage_ratio < 0.30:
            return (
                False,
                f"The image does not match any recognized plant pathology (Prediction confidence: {max_prob*100:.1f}%). Please ensure the leaf is clearly focused.",
                meta
            )

    return True, "Valid plant leaf confirmed.", meta
