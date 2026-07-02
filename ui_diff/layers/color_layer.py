"""
layers/color_layer.py — Layer D: Color matching (PRIMARY, weight 0.30).

Pipeline:
  1. Per matched element region: k-means dominant color extraction (k=1–3).
  2. Compare expected vs actual dominant color using Delta-E CIE2000 in Lab space.
     (Raw RGB Euclidean distance is NOT used — it doesn't match human perception.)
  3. Global palette: whole-image color histogram comparison to catch theme-level
     mismatches even when element matching is imperfect.
  4. Final score blends per-element Delta-E score with global histogram similarity.

Note: Delta-E CIE2000 is implemented manually using pure NumPy to avoid the
      colormath / numpy.asscalar incompatibility with NumPy >= 1.25.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .position_layer import MatchedPair, BBox


@dataclass
class ColorIssue:
    element_index: int
    expected_rgb: tuple[int, int, int]
    actual_rgb: tuple[int, int, int]
    delta_e: float
    severity: str   # "match" | "drift" | "mismatch"


@dataclass
class ColorResult:
    score: float                                  # 0.0 – 1.0
    per_element_scores: list[float] = field(default_factory=list)
    color_issues: list[ColorIssue] = field(default_factory=list)
    global_histogram_score: float = 0.0


class ColorLayer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        col = cfg.get("color", {})
        self._k = int(col.get("kmeans_k", 3))
        self._delta_e_thresh = float(col.get("delta_e_threshold", 10.0))
        self._hist_bins = int(col.get("histogram_bins", 64))
        self._hist_weight = float(col.get("histogram_weight", 0.3))

    def run(
        self,
        exp_arr: np.ndarray,
        act_arr: np.ndarray,
        matched_pairs: list[MatchedPair],
    ) -> ColorResult:
        """Compute color match score.

        Args:
            exp_arr: HxWx3 uint8 RGB — full expected image.
            act_arr: HxWx3 uint8 RGB — full actual image.
            matched_pairs: from PositionLayer, used to sample per-element colors.

        Returns:
            ColorResult with .score, .per_element_scores, .color_issues.
        """
        # ── Per-element color comparison ──────────────────────────────────────
        per_element_scores: list[float] = []
        color_issues: list[ColorIssue] = []

        for idx, pair in enumerate(matched_pairs):
            exp_rgb = self._dominant_color(exp_arr, pair.expected_bbox)
            act_rgb = self._dominant_color(act_arr, pair.actual_bbox)
            de = self._delta_e(exp_rgb, act_rgb)
            element_score = max(0.0, 1.0 - de / self._delta_e_thresh)
            per_element_scores.append(element_score)

            severity = (
                "match" if de < 2.0 else
                "drift" if de < self._delta_e_thresh * 0.5 else
                "mismatch"
            )
            if severity != "match":
                color_issues.append(ColorIssue(
                    element_index=idx,
                    expected_rgb=exp_rgb,
                    actual_rgb=act_rgb,
                    delta_e=de,
                    severity=severity,
                ))

        per_element_avg = (
            float(np.mean(per_element_scores)) if per_element_scores else 0.0
        )

        # ── Global histogram similarity ────────────────────────────────────────
        hist_score = self._histogram_similarity(exp_arr, act_arr)

        # ── Blend ─────────────────────────────────────────────────────────────
        hw = self._hist_weight
        pw = 1.0 - hw
        final_score = pw * per_element_avg + hw * hist_score

        return ColorResult(
            score=float(np.clip(final_score, 0.0, 1.0)),
            per_element_scores=per_element_scores,
            color_issues=color_issues,
            global_histogram_score=hist_score,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dominant_color(self, arr: np.ndarray, bbox: BBox) -> tuple[int, int, int]:
        """Extract dominant color from a bounding-box region via k-means."""
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        # Clamp to image bounds
        ih, iw = arr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        region = arr[y1:y2, x1:x2]

        if region.size == 0:
            return (128, 128, 128)

        pixels = region.reshape(-1, 3).astype(np.float32)
        if len(pixels) < self._k:
            # Fewer pixels than k — just use mean
            mean = pixels.mean(axis=0).astype(int)
            return (int(mean[0]), int(mean[1]), int(mean[2]))

        k = min(self._k, len(pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Ensure deterministic k-means by setting OpenCV's RNG seed
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        # Pick the cluster with the most pixels
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return (int(dominant[0]), int(dominant[1]), int(dominant[2]))

    def _delta_e(
        self, rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]
    ) -> float:
        """Delta-E CIE2000 between two sRGB colors — pure NumPy implementation.

        Avoids colormath which calls numpy.asscalar (removed in NumPy 1.25).
        """
        lab1 = _rgb_to_lab(rgb1)
        lab2 = _rgb_to_lab(rgb2)
        return _delta_e_cie2000(lab1, lab2)

    def _histogram_similarity(
        self, exp_arr: np.ndarray, act_arr: np.ndarray
    ) -> float:
        """Global BGR histogram correlation (Bhattacharyya distance → score).

        Uses L2 normalization instead of NORM_MINMAX to avoid NaN when
        all histogram bins are zero (e.g. uniform-color image).
        Falls back to correlation metric if Bhattacharyya produces NaN.
        """
        bins = self._hist_bins
        exp_bgr = cv2.cvtColor(exp_arr, cv2.COLOR_RGB2BGR)
        act_bgr = cv2.cvtColor(act_arr, cv2.COLOR_RGB2BGR)

        scores = []
        for ch in range(3):
            h_exp = cv2.calcHist([exp_bgr], [ch], None, [bins], [0, 256])
            h_act = cv2.calcHist([act_bgr], [ch], None, [bins], [0, 256])

            # L2 normalise — safe even when all bins are zero
            norm_exp = np.linalg.norm(h_exp)
            norm_act = np.linalg.norm(h_act)
            if norm_exp > 0:
                h_exp = h_exp / norm_exp
            if norm_act > 0:
                h_act = h_act / norm_act

            # Correlation: 1 = identical, -1 = opposite
            h_exp_f = h_exp.astype(np.float32)
            h_act_f = h_act.astype(np.float32)
            corr = cv2.compareHist(h_exp_f, h_act_f, cv2.HISTCMP_CORREL)
            # Map from [-1, 1] → [0, 1]
            sim = (float(corr) + 1.0) / 2.0

            # Guard against NaN / Inf from degenerate inputs
            if not np.isfinite(sim):
                sim = 0.0
            scores.append(max(0.0, min(1.0, sim)))

        return float(np.mean(scores))


# ── Pure-NumPy Lab conversion & CIE2000 Delta-E ───────────────────────────────
# Avoids colormath which uses numpy.asscalar (removed in NumPy 1.25).

def _rgb_to_lab(rgb: tuple) -> tuple:
    """Convert sRGB (0-255 int) → CIE L*a*b* using D65 illuminant."""
    def _linearize(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _linearize(rgb[0]), _linearize(rgb[1]), _linearize(rgb[2])

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def _f(t: float) -> float:
        delta = 6.0 / 29.0
        return t ** (1.0 / 3.0) if t > delta**3 else t / (3 * delta**2) + 4.0 / 29.0

    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def _delta_e_cie2000(lab1: tuple, lab2: tuple) -> float:
    """CIE2000 Delta-E between two L*a*b* colors — pure Python/math."""
    import math
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    kL = kC = kH = 1.0
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1 - math.sqrt(C_avg7 / (C_avg7 + 25**7)))
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        Hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        Hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        Hp_avg = (h1p + h2p + 360) / 2.0
    else:
        Hp_avg = (h1p + h2p - 360) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(Hp_avg - 30))
        + 0.24 * math.cos(math.radians(2 * Hp_avg))
        + 0.32 * math.cos(math.radians(3 * Hp_avg + 6))
        - 0.20 * math.cos(math.radians(4 * Hp_avg - 63))
    )

    SL = 1 + 0.015 * (Lp_avg - 50)**2 / math.sqrt(20 + (Lp_avg - 50)**2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2 * math.sqrt(Cp_avg7 / (Cp_avg7 + 25**7))
    d_theta = 30 * math.exp(-((Hp_avg - 275) / 25)**2)
    RT = -math.sin(math.radians(2 * d_theta)) * RC

    return math.sqrt(
        (dLp / (kL * SL))**2
        + (dCp / (kC * SC))**2
        + (dHp / (kH * SH))**2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
