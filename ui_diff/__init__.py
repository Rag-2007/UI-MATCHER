"""
ui_diff — UI Visual Match Validator
====================================
Public interface: compare two UI screenshots and get a confidence score + diff report.

Usage (Python library):
    from ui_diff import compare
    result = compare("expected.png", "actual.png", config="config/default_weights.yaml")
    assert result.confidence_score >= 85
    assert not result.has_critical_failures
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import numpy as np
from PIL import Image

from .preprocess import preprocess_pair
from .layers.ssim_layer import SSIMLayer
from .layers.phash_layer import PHashLayer
from .layers.position_layer import PositionLayer
from .layers.color_layer import ColorLayer
from .fusion import fuse_scores
from .report import CompareResult, generate_report
from ._config import load_config


def compare(
    expected: Union[str, pathlib.Path, Image.Image, np.ndarray],
    actual: Union[str, pathlib.Path, Image.Image, np.ndarray],
    config: Union[str, pathlib.Path, dict, None] = None,
    ignore_regions: Optional[list[dict]] = None,
    diff_image_path: Optional[Union[str, pathlib.Path]] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
) -> CompareResult:
    """Compare two UI screenshots and return a CompareResult.

    Args:
        expected: Path, PIL Image, or numpy array of the reference/design mock.
        actual: Path, PIL Image, or numpy array of the built UI screenshot.
        config: Path to YAML config, or a dict of config values.
                Defaults to the bundled ``config/default_weights.yaml``.
        ignore_regions: List of ``{x, y, w, h}`` dicts to exclude from scoring.
        diff_image_path: If provided, save the annotated diff overlay here.
        output_path: If provided, save the JSON report here.

    Returns:
        CompareResult with .confidence_score, .layers, .issues, .has_critical_failures
    """
    cfg = load_config(config)

    # Merge runtime ignore_regions with config ones
    cfg_regions = cfg.get("ignore_regions", []) or []
    all_ignore = cfg_regions + (ignore_regions or [])

    # ── Stage 1: Preprocess & Align ──────────────────────────────────────────
    exp_img, act_img, exp_arr, act_arr, masked_exp, masked_act = preprocess_pair(
        expected, actual, cfg, all_ignore
    )

    # ── Fast Path: Exact Match ────────────────────────────────────────────────
    if np.array_equal(exp_arr, act_arr):
        return CompareResult(
            confidence_score=100.0,
            layers={"ssim": 1.0, "phash": 1.0, "position_match": 1.0, "color_match": 1.0},
            issues=[],
            has_critical_failures=False,
            diff_image_path=str(diff_image_path) if diff_image_path else None,
            report_path=str(output_path) if output_path else None,
        )

    # ── Stage 2: Run comparison layers ───────────────────────────────────────
    ssim_result = SSIMLayer(cfg).run(masked_exp, masked_act)
    phash_result = PHashLayer(cfg).run(exp_img, act_img)
    position_result = PositionLayer(cfg).run(exp_arr, act_arr)
    color_result = ColorLayer(cfg).run(exp_arr, act_arr, position_result.matched_pairs)

    # ── Stage 3: Score fusion ─────────────────────────────────────────────────
    fused = fuse_scores(cfg, ssim_result, phash_result, position_result, color_result)

    # ── Stage 4: Report ───────────────────────────────────────────────────────
    result = generate_report(
        cfg=cfg,
        fused=fused,
        ssim_result=ssim_result,
        phash_result=phash_result,
        position_result=position_result,
        color_result=color_result,
        exp_arr=exp_arr,
        act_arr=act_arr,
        diff_image_path=diff_image_path,
        output_path=output_path,
    )

    return result


__all__ = ["compare", "CompareResult"]
