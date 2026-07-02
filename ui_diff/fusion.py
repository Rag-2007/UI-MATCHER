"""
fusion.py — Stage 3: Score fusion and hard-fail rules.

Combines layer scores using config-driven weights.
Applies hard-fail rules: critical elements missing → cap score at threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .layers.ssim_layer import SSIMResult
from .layers.phash_layer import PHashResult
from .layers.position_layer import PositionResult
from .layers.color_layer import ColorResult


@dataclass
class FusionResult:
    confidence_score: float           # 0–100 (final weighted score)
    raw_score: float                  # 0–1 (pre-100 scaling)
    capped: bool = False              # True if hard-fail cap was applied
    cap_reason: str = ""
    issues: list[dict] = field(default_factory=list)


def fuse_scores(
    cfg: dict,
    ssim_result: SSIMResult,
    phash_result: PHashResult,
    position_result: PositionResult,
    color_result: ColorResult,
) -> FusionResult:
    """Fuse layer scores into a final confidence score.

    confidence = (
        w_ssim  * ssim_score  +   # Layer A
        w_phash * phash_score +   # Layer B
        w_pos   * pos_score   +   # Layer C  (primary)
        w_color * color_score     # Layer D  (primary)
    ) * 100
    """
    weights = cfg.get("weights", {})
    w_ssim = float(weights.get("ssim", 0.15))
    w_phash = float(weights.get("phash", 0.05))
    w_pos = float(weights.get("position", 0.50))
    w_color = float(weights.get("color", 0.30))

    def _safe(v: float, fallback: float = 0.0) -> float:
        """Guard against NaN/Inf — return fallback instead of silently returning 1.0."""
        import math
        return v if math.isfinite(v) else fallback

    raw = (
        w_ssim  * _safe(ssim_result.score)
        + w_phash * _safe(phash_result.score)
        + w_pos   * _safe(position_result.score)
        + w_color * _safe(color_result.score)
    )
    raw = float(max(0.0, min(1.0, raw)))

    # ── Structural Gating ─────────────────────────────────────────────────────
    # If the structural layout is fundamentally broken (Position F1 < 0.6),
    # we penalize the overall score to prevent SSIM/Color from inflating it.
    pos_score = _safe(position_result.score)
    if pos_score < 0.6:
        # Scale down linearly: at 0.6 it's 1.0x, at 0.0 it's 0.0x
        multiplier = pos_score / 0.6
        raw *= multiplier

    # ── Collect issues ────────────────────────────────────────────────────────
    issues: list[dict] = []

    for i, bbox in enumerate(position_result.missing_elements):
        issues.append({
            "type": "missing_element",
            "element_index": i,
            "bbox": {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
            "message": f"Element {i} from expected image not found in actual.",
        })

    for i, bbox in enumerate(position_result.extra_elements):
        issues.append({
            "type": "extra_element",
            "element_index": i,
            "bbox": {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
            "message": f"Extra element {i} found in actual image, not in expected.",
        })

    for pair in position_result.matched_pairs:
        if pair.severity in ("drift", "mismatch"):
            issues.append({
                "type": "position_shift",
                "severity": pair.severity,
                "expected_bbox": {
                    "x": pair.expected_bbox.x, "y": pair.expected_bbox.y,
                    "w": pair.expected_bbox.w, "h": pair.expected_bbox.h,
                },
                "actual_bbox": {
                    "x": pair.actual_bbox.x, "y": pair.actual_bbox.y,
                    "w": pair.actual_bbox.w, "h": pair.actual_bbox.h,
                },
                "delta_x_norm": round(pair.delta_x, 4),
                "delta_y_norm": round(pair.delta_y, 4),
                "message": (
                    f"Element shifted: Δx={pair.delta_x:.3f}, Δy={pair.delta_y:.3f} "
                    f"(normalized to image dims)"
                ),
            })

    for issue in color_result.color_issues:
        issues.append({
            "type": "color_mismatch",
            "element_index": issue.element_index,
            "severity": issue.severity,
            "expected_rgb": issue.expected_rgb,
            "actual_rgb": issue.actual_rgb,
            "delta_e": round(issue.delta_e, 2),
            "message": (
                f"Color mismatch (ΔE={issue.delta_e:.1f}): "
                f"expected RGB{issue.expected_rgb} got RGB{issue.actual_rgb}"
            ),
        })

    # ── Hard-fail: critical elements ──────────────────────────────────────────
    critical_elements: list[str] = cfg.get("critical_elements", []) or []
    score_cap = float(cfg.get("critical_element_score_cap", 60.0))
    capped = False
    cap_reason = ""

    if critical_elements:
        # For now, critical element enforcement is tag-based (future: DOM integration).
        # When using CV detection, we flag if the number of missing elements exceeds
        # the count of critical elements defined.
        n_missing = len(position_result.missing_elements)
        if n_missing >= len(critical_elements):
            raw = min(raw, score_cap / 100.0)
            capped = True
            cap_reason = (
                f"Critical element(s) may be missing — score capped at {score_cap}. "
                f"Missing element count: {n_missing}."
            )
            issues.append({
                "type": "critical_element_missing",
                "severity": "critical",
                "message": cap_reason,
            })

    return FusionResult(
        confidence_score=round(raw * 100, 2),
        raw_score=raw,
        capped=capped,
        cap_reason=cap_reason,
        issues=issues,
    )
