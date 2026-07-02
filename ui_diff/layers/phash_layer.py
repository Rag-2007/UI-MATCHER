"""
layers/phash_layer.py — Layer B: Perceptual hash gate.

Weight: 0.05 (very low — cheap early-exit gate).
Uses imagehash average hash; returns normalized similarity 0–1.
"""
from __future__ import annotations

from dataclasses import dataclass

import imagehash
from PIL import Image
import numpy as np


@dataclass
class PHashResult:
    score: float       # 0.0 – 1.0 (1 = identical hash)
    hash_distance: int # raw Hamming distance between hashes


class PHashLayer:
    # Average hash uses a 8x8 = 64-bit hash by default
    _HASH_BITS = 64

    def __init__(self, cfg: dict):
        self._cfg = cfg

    def run(self, exp_pil: Image.Image, act_pil: Image.Image) -> PHashResult:
        """Compute perceptual hash similarity.

        Args:
            exp_pil: PIL Image — expected (already resized to reference resolution).
            act_pil: PIL Image — actual.

        Returns:
            PHashResult with .score (0–1) and raw .hash_distance.
        """
        h_exp = imagehash.average_hash(exp_pil)
        h_act = imagehash.average_hash(act_pil)

        distance = int(h_exp - h_act)  # Hamming distance
        score = 1.0 - (distance / self._HASH_BITS)
        score = float(np.clip(score, 0.0, 1.0))

        return PHashResult(score=score, hash_distance=distance)
