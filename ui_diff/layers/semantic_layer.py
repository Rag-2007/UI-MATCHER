"""
layers/semantic_layer.py — Semantic similarity layer (Phase 2, optional).

Uses CLIP image embeddings (open_clip_torch) for semantic-level comparison.
Not included in default score fusion — placeholder for future enhancement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticResult:
    score: float = 1.0
    enabled: bool = False
    message: str = "Semantic layer (CLIP) is a Phase 2 optional feature."


class SemanticLayer:
    """Phase 2 CLIP-based semantic similarity. Not included in default scoring."""

    def __init__(self, cfg: dict):
        self._enabled = cfg.get("semantic", {}).get("enabled", False)

    def run(self, exp_arr, act_arr) -> SemanticResult:
        if not self._enabled:
            return SemanticResult()
        try:
            import open_clip
            import torch
            from PIL import Image
            import numpy as np

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model.eval()

            def _encode(arr):
                img = Image.fromarray(arr)
                t = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    feat = model.encode_image(t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat

            f_exp = _encode(exp_arr)
            f_act = _encode(act_arr)
            cos_sim = float((f_exp * f_act).sum())
            score = (cos_sim + 1) / 2  # shift from [-1,1] to [0,1]
            return SemanticResult(score=score, enabled=True, message="")
        except ImportError:
            return SemanticResult(
                message="open_clip_torch not installed. Install it to enable semantic layer."
            )
