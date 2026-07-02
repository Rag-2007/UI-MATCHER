"""
preprocess.py — Stage 1: Image preprocessing and alignment.

Responsibilities:
- Load images from various input types (path, PIL Image, numpy array).
- Resize both images to a common reference resolution (letterbox pad, preserve AR).
- Apply ignore-region masks to both images.
- Produce a lightly Gaussian-blurred pair for SSIM (blur never used for detection/color).
"""
from __future__ import annotations

import pathlib
from typing import Union, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_pair(
    expected: Union[str, pathlib.Path, Image.Image, np.ndarray],
    actual: Union[str, pathlib.Path, Image.Image, np.ndarray],
    cfg: dict,
    ignore_regions: Optional[list[dict]] = None,
):
    """Preprocess both images and return six outputs:

    Returns:
        exp_pil, act_pil          — PIL Image (resized, RGB, unblurred)
        exp_arr, act_arr          — numpy uint8 HxWx3 (resized, RGB, unblurred)
        masked_exp, masked_act    — numpy uint8 HxWx3 blurred + masked (for SSIM)
    """
    pre_cfg = cfg.get("preprocess", {})
    ref_w = int(pre_cfg.get("reference_width", 1280))
    ref_h = int(pre_cfg.get("reference_height", 800))
    blur_radius = int(pre_cfg.get("ssim_blur_radius", 1))
    pad_color = tuple(pre_cfg.get("pad_color", [255, 255, 255]))

    exp_pil = _load_and_resize(expected, ref_w, ref_h, pad_color)
    act_pil = _load_and_resize(actual, ref_w, ref_h, pad_color)

    exp_arr = np.array(exp_pil)
    act_arr = np.array(act_pil)

    # Apply ignore-region masks (zero-out / fill with pad color)
    regions = ignore_regions or []
    if regions:
        exp_arr = _apply_mask(exp_arr, regions, fill=pad_color)
        act_arr = _apply_mask(act_arr, regions, fill=pad_color)
        exp_pil = Image.fromarray(exp_arr)
        act_pil = Image.fromarray(act_arr)

    # Blurred copies for SSIM only
    masked_exp = _gaussian_blur(exp_arr, blur_radius)
    masked_act = _gaussian_blur(act_arr, blur_radius)

    return exp_pil, act_pil, exp_arr, act_arr, masked_exp, masked_act


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_image(src: Union[str, pathlib.Path, Image.Image, np.ndarray]) -> Image.Image:
    """Accept path / PIL / numpy and always return an RGB PIL Image."""
    if isinstance(src, (str, pathlib.Path)):
        return Image.open(src).convert("RGB")
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        elif src.shape[2] == 4:
            src = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
        elif src.shape[2] == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) if _looks_bgr(src) else src
        return Image.fromarray(src.astype(np.uint8))
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(src)}")


def _looks_bgr(arr: np.ndarray) -> bool:
    """Heuristic: if the array came from cv2 it's BGR — can't tell definitively,
    so we trust PIL paths and only convert numpy arrays passed from cv2 callers.
    This is best-effort; callers should pass PIL or file paths when possible."""
    return False  # Treat numpy input as RGB by default; cv2-callers should convert before passing.


def _load_and_resize(
    src: Union[str, pathlib.Path, Image.Image, np.ndarray],
    ref_w: int,
    ref_h: int,
    pad_color: tuple,
) -> Image.Image:
    """Resize image to (ref_w, ref_h) using letterbox padding (preserve aspect ratio)."""
    img = _load_image(src)
    return _letterbox(img, ref_w, ref_h, pad_color)


def _letterbox(img: Image.Image, target_w: int, target_h: int, pad_color: tuple) -> Image.Image:
    """Fit image into target dimensions with padding. Never stretches."""
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _apply_mask(arr: np.ndarray, regions: list[dict], fill=(255, 255, 255)) -> np.ndarray:
    """Zero out / fill ignore regions in-place (copy first to avoid mutation)."""
    arr = arr.copy()
    for region in regions:
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        arr[y : y + h, x : x + w] = fill
    return arr


def _gaussian_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Apply Gaussian blur for SSIM pre-processing. Returns new array."""
    if radius <= 0:
        return arr.copy()
    # kernel size must be odd
    ksize = max(1, radius * 2 + 1)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=radius)
    return blurred
