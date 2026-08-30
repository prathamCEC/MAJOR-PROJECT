"""
Visualization Module for Grad-CAM Heatmaps and SHAP Feature Importance.

Generates high-resolution publication-quality overlays and attribution bar charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_cam_on_image(
    image: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay 2D normalized Grad-CAM activation map onto retinal image.

    Args:
        image: Original RGB or Grayscale image
        cam: 2D numpy array [H, W] normalized in [0.0, 1.0]
        alpha: Transparency blending factor (0.0 to 1.0)
        colormap: Matplotlib colormap name ('jet', 'viridis', 'inferno')

    Returns:
        Tuple of (colored_heatmap_rgb, blended_overlay_rgb)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        if image.ndim == 2:
            img_np = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            img_np = np.repeat(image, 3, axis=-1)
        else:
            img_np = image.copy()

    # Ensure float in [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np.astype(np.float32) / 255.0

    H, W = img_np.shape[:2]
    # Resize CAM to match image exactly if needed
    if cam.shape[:2] != (H, W):
        from PIL import Image as PILImg
        cam_pil = PILImg.fromarray((cam * 255).astype(np.uint8)).resize((W, H), PILImg.BILINEAR)
        cam = np.array(cam_pil).astype(np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_cam = cmap(cam)[:, :, :3]  # [H, W, 3]

    # Alpha blend: (1 - alpha) * img + alpha * heatmap
    overlay = (1.0 - alpha) * img_np + alpha * colored_cam
    overlay = np.clip(overlay, 0.0, 1.0)

    return (colored_cam * 255).astype(np.uint8), (overlay * 255).astype(np.uint8)


def save_gradcam_panel(
    original_img: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Grad-CAM Explanation",
    modality: str = "OCT-A",
    disease_target: str = "Stroke",
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Path:
    """
    Generate and save a 3-panel figure: [Original | Heatmap | Overlay].
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    colored_cam, overlay = overlay_cam_on_image(original_img, cam, alpha=alpha, colormap=colormap)

    if isinstance(original_img, Image.Image):
        orig_np = np.array(original_img.convert("RGB"))
    else:
        orig_np = original_img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} — {disease_target.upper()} ({modality.upper()})", fontsize=14, fontweight="bold")

    axes[0].imshow(orig_np, cmap="gray" if orig_np.ndim == 2 else None)
    axes[0].set_title(f"Original {modality.upper()} Scan", fontsize=11)
    axes[0].axis("off")

    im1 = axes[1].imshow(cam, cmap=colormap, vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Activation Map", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Aligned Retinal Overlay", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_gradcam_components(
    original_img: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    output_dir: Union[str, Path],
    file_prefix: str = "gradcam",
    title: str = "Grad-CAM Explanation",
    modality: str = "OCT-A",
    disease_target: str = "Stroke",
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Dict[str, Path]:
    """
    Save individual components for interactive UI:
    - Original retinal image
    - Pure Grad-CAM heatmap (colored)
    - Alpha-blended retinal overlay
    - 3-Panel comparison figure

    Returns:
        Dict mapping 'original', 'heatmap', 'overlay', 'panel' to their Path objects.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    colored_cam, overlay = overlay_cam_on_image(original_img, cam, alpha=alpha, colormap=colormap)

    if isinstance(original_img, Image.Image):
        orig_np = np.array(original_img.convert("RGB"))
    else:
        orig_np = original_img

    orig_path = out_dir / f"{file_prefix}_original.png"
    heatmap_path = out_dir / f"{file_prefix}_heatmap.png"
    overlay_path = out_dir / f"{file_prefix}_overlay.png"
    legacy_panel_path = out_dir / f"{file_prefix}.png"
    panel_path = out_dir / f"{file_prefix}_panel.png"

    # 1. Save Original Image
    if orig_np.ndim == 2:
        Image.fromarray(orig_np).save(orig_path)
    else:
        if orig_np.dtype != np.uint8:
            orig_to_save = (np.clip(orig_np, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            orig_to_save = orig_np
        Image.fromarray(orig_to_save).save(orig_path)

    # 2. Save Pure Heatmap
    Image.fromarray(colored_cam).save(heatmap_path)

    # 3. Save Overlay
    Image.fromarray(overlay).save(overlay_path)

    # 4. Save 3-Panel Figure
    save_gradcam_panel(
        original_img=original_img,
        cam=cam,
        output_path=legacy_panel_path,
        title=title,
        modality=modality,
        disease_target=disease_target,
        alpha=alpha,
        colormap=colormap,
    )
    if panel_path != legacy_panel_path:
        import shutil
        shutil.copyfile(legacy_panel_path, panel_path)

    return {
        "original": orig_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "panel": legacy_panel_path,
    }



def save_shap_bar_chart(
    shap_summary: List[Dict[str, Any]],
    output_path: Union[str, Path],
    disease_target: str = "Stroke",
    base_value: Optional[float] = None,
) -> Path:
    """
    Generate and save horizontal SHAP feature importance bar chart.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    features = [item["feature"] for item in shap_summary][::-1]
    values = [item["shap_value"] for item in shap_summary][::-1]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))

    ax.barh(y_pos, values, color=colors, align="center", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)

    ax.set_xlabel("SHAP Value (Impact on Model Logit)", fontsize=11)
    title_text = f"Clinical Feature Attributions (SHAP) — {disease_target.upper()}"
    if base_value is not None:
        title_text += f" (Base Expected: {base_value:.3f})"
    ax.set_title(title_text, fontsize=12, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Increases Risk (Positive)"),
        Patch(facecolor="#3498db", label="Decreases Risk (Negative)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
