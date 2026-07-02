"""
layers/ssim_layer.py — Layer A: SSIM structural similarity.

Weight: 0.15 (sanity check, low weight).
Uses scikit-image SSIM on lightly-blurred grayscale images to neutralise
anti-aliasing noise. Generates a diff heatmap for report overlay use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


@dataclass
class SSIMResult:
    score: float                           # 0.0 – 1.0
    diff_map: Optional[np.ndarray] = None  # HxW float array, same shape as inputs


class SSIMLayer:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    def run(self, exp_blurred: np.ndarray, act_blurred: np.ndarray) -> SSIMResult:
        """Run SSIM on pre-blurred arrays.

        Args:
            exp_blurred: HxWx3 uint8 — blurred expected image (from preprocess).
            act_blurred: HxWx3 uint8 — blurred actual image.

        Returns:
            SSIMResult with .score (0–1) and .diff_map heatmap.
        """
        # Convert to grayscale for SSIM
        exp_gray = cv2.cvtColor(exp_blurred, cv2.COLOR_RGB2GRAY).astype(np.float64)
        act_gray = cv2.cvtColor(act_blurred, cv2.COLOR_RGB2GRAY).astype(np.float64)

        score, diff_map = ssim(
            exp_gray,
            act_gray,
            full=True,
            data_range=255.0,
        )

        # diff_map is 0–1 where 1 = identical; invert so higher = more different
        diff_map_inv = 1.0 - diff_map

        # Clamp score to [0, 1]
        score = float(np.clip(score, 0.0, 1.0))

        return SSIMResult(score=score, diff_map=diff_map_inv)
