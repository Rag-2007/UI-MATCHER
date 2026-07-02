"""
report.py — Stage 4: Report generation.

Produces:
  - JSON report matching the spec schema.
  - Annotated visual diff overlay image:
      green  = matched elements (good match)
      yellow = matched elements with minor position/color drift
      red    = missing elements / major mismatches
    With centroid arrows for shifted elements.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .layers.ssim_layer import SSIMResult
from .layers.phash_layer import PHashResult
from .layers.position_layer import PositionResult, MatchedPair, BBox
from .layers.color_layer import ColorResult
from .fusion import FusionResult


# ── Public result dataclass ───────────────────────────────────────────────────

@dataclass
class CompareResult:
    """Public result returned by ui_diff.compare()."""
    confidence_score: float
    layers: dict
    issues: list[dict] = field(default_factory=list)
    has_critical_failures: bool = False
    diff_image_path: Optional[str] = None
    report_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "confidence_score": self.confidence_score,
            "layers": self.layers,
            "issues": self.issues,
            "has_critical_failures": self.has_critical_failures,
            "diff_image_path": self.diff_image_path,
            "report_path": self.report_path,
        }

    def __repr__(self) -> str:
        return (
            f"CompareResult(confidence_score={self.confidence_score}, "
            f"issues={len(self.issues)}, "
            f"has_critical_failures={self.has_critical_failures})"
        )


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_report(
    cfg: dict,
    fused: FusionResult,
    ssim_result: SSIMResult,
    phash_result: PHashResult,
    position_result: PositionResult,
    color_result: ColorResult,
    exp_arr: np.ndarray,
    act_arr: np.ndarray,
    diff_image_path: Optional[Union[str, pathlib.Path]] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
) -> CompareResult:
    """Build CompareResult, save JSON report, and generate annotated diff image."""

    layers = {
        "ssim": round(ssim_result.score, 4),
        "phash": round(phash_result.score, 4),
        "position_match": round(position_result.score, 4),
        "color_match": round(color_result.score, 4),
    }

    has_critical = any(
        issue.get("type") == "critical_element_missing"
        for issue in fused.issues
    )

    diff_path_str = None
    if diff_image_path is not None:
        diff_path_str = str(diff_image_path)
        _render_diff_image(
            exp_arr=exp_arr,
            act_arr=act_arr,
            position_result=position_result,
            ssim_diff=ssim_result.diff_map,
            output_path=diff_image_path,
        )

    report_path_str = None
    if output_path is not None:
        report_path_str = str(output_path)
        report_dict = {
            "confidence_score": fused.confidence_score,
            "layers": layers,
            "issues": fused.issues,
            "has_critical_failures": has_critical,
            "score_capped": fused.capped,
            "cap_reason": fused.cap_reason,
            "diff_image_path": diff_path_str,
        }
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_dict, indent=2))

    return CompareResult(
        confidence_score=fused.confidence_score,
        layers=layers,
        issues=fused.issues,
        has_critical_failures=has_critical,
        diff_image_path=diff_path_str,
        report_path=report_path_str,
    )


# ── Diff image rendering ──────────────────────────────────────────────────────

def _render_diff_image(
    exp_arr: np.ndarray,
    act_arr: np.ndarray,
    position_result: PositionResult,
    ssim_diff: Optional[np.ndarray],
    output_path: Union[str, pathlib.Path],
) -> None:
    """Render a side-by-side annotated diff image and save it."""
    h, w = exp_arr.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")
    fig.suptitle(
        "UI Visual Match — Diff Report",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    # ── Left: Expected annotated ──────────────────────────────────────────────
    ax_exp = axes[0]
    ax_exp.imshow(exp_arr)
    ax_exp.set_title("Expected", color="white", fontsize=11)
    ax_exp.axis("off")
    _draw_boxes_on_ax(ax_exp, position_result, side="expected")

    # ── Middle: Actual annotated ──────────────────────────────────────────────
    ax_act = axes[1]
    ax_act.imshow(act_arr)
    ax_act.set_title("Actual", color="white", fontsize=11)
    ax_act.axis("off")
    _draw_boxes_on_ax(ax_act, position_result, side="actual")

    # ── Right: SSIM heatmap or pixel diff ────────────────────────────────────
    ax_diff = axes[2]
    if ssim_diff is not None:
        # Normalize heatmap and apply colormap
        diff_norm = np.clip(ssim_diff, 0, 1)
        ax_diff.imshow(diff_norm, cmap="hot", vmin=0, vmax=1)
        ax_diff.set_title("SSIM Diff Heatmap\n(brighter = more different)", color="white", fontsize=11)
    else:
        diff_rgb = cv2.absdiff(exp_arr, act_arr)
        ax_diff.imshow(diff_rgb)
        ax_diff.set_title("Pixel Diff", color="white", fontsize=11)
    ax_diff.axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#00e676", label="Match"),
        mpatches.Patch(color="#ffea00", label="Drift"),
        mpatches.Patch(color="#ff1744", label="Missing / Mismatch"),
        mpatches.Patch(color="#00b0ff", label="Extra"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        framealpha=0.3,
        labelcolor="white",
        facecolor="#1a1a2e",
        fontsize=9,
    )

    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_boxes_on_ax(ax, position_result: PositionResult, side: str) -> None:
    """Draw bounding boxes on a matplotlib Axes."""
    import matplotlib.patches as patches

    # Matched pairs
    for pair in position_result.matched_pairs:
        bbox = pair.expected_bbox if side == "expected" else pair.actual_bbox
        color = {
            "match": "#00e676",
            "drift": "#ffea00",
            "mismatch": "#ff1744",
        }.get(pair.severity, "#00e676")
        rect = patches.Rectangle(
            (bbox.x, bbox.y), bbox.w, bbox.h,
            linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.85,
        )
        ax.add_patch(rect)

        # Draw shift arrow on actual side
        if side == "actual" and pair.severity in ("drift", "mismatch"):
            eb = pair.expected_bbox
            ab = pair.actual_bbox
            ax.annotate(
                "",
                xy=(ab.cx, ab.cy),
                xytext=(eb.cx, eb.cy),
                arrowprops=dict(arrowstyle="->", color="#ffea00", lw=1.2),
            )

    # Missing elements (only on expected side)
    if side == "expected":
        for bbox in position_result.missing_elements:
            rect = patches.Rectangle(
                (bbox.x, bbox.y), bbox.w, bbox.h,
                linewidth=1.5, edgecolor="#ff1744", facecolor="#ff174422",
                linestyle="--",
            )
            ax.add_patch(rect)

    # Extra elements (only on actual side)
    if side == "actual":
        for bbox in position_result.extra_elements:
            rect = patches.Rectangle(
                (bbox.x, bbox.y), bbox.w, bbox.h,
                linewidth=1.5, edgecolor="#00b0ff", facecolor="#00b0ff22",
                linestyle=":",
            )
            ax.add_patch(rect)
