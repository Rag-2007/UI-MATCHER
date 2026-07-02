"""
layers/position_layer.py — Layer C: Element position matching (PRIMARY, weight 0.50).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x + self.w, other.x + other.w)
        iy2 = min(self.y + self.h, other.y + other.h)
        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter = inter_w * inter_h
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def centroid_dist(self, other: "BBox", img_w: int, img_h: int) -> float:
        """Normalized centroid distance (0 = same center, 1 = diagonal distance)."""
        dx = (self.cx - other.cx) / img_w
        dy = (self.cy - other.cy) / img_h
        diag = (1.0**2 + 1.0**2) ** 0.5
        return min((dx**2 + dy**2) ** 0.5 / diag, 1.0)


@dataclass
class MatchedPair:
    expected_bbox: BBox
    actual_bbox: BBox
    iou: float
    delta_x: float      # normalized position shift Δx
    delta_y: float      # normalized position shift Δy
    delta_w: float      # normalized size shift Δw
    delta_h: float      # normalized size shift Δh
    severity: str       # "match" | "drift" | "mismatch"


@dataclass
class PositionResult:
    score: float                              # 0.0 – 1.0
    matched_pairs: list[MatchedPair] = field(default_factory=list)
    missing_elements: list[BBox] = field(default_factory=list)   # in expected, not in actual
    extra_elements: list[BBox] = field(default_factory=list)     # in actual, not in expected
    expected_boxes: list[BBox] = field(default_factory=list)
    actual_boxes: list[BBox] = field(default_factory=list)


class PositionLayer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        det = cfg.get("element_detection", {})
        self._canny_t1 = int(det.get("canny_threshold1", 50))
        self._canny_t2 = int(det.get("canny_threshold2", 150))
        self._min_area = int(det.get("min_contour_area", 500))
        self._max_area_ratio = float(det.get("max_contour_area_ratio", 0.90))
        self._min_ar = float(det.get("min_aspect_ratio", 0.05))
        self._max_ar = float(det.get("max_aspect_ratio", 20.0))
        self._iou_w = float(det.get("iou_weight", 0.6))
        self._cent_w = float(det.get("centroid_weight", 0.4))
        self._cost_thresh = float(det.get("match_cost_threshold", 0.85))
        pen = cfg.get("penalties", {})
        self._miss_pen = float(pen.get("missing_element_penalty", 0.10))
        self._extra_pen = float(pen.get("extra_element_penalty", 0.05))

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, exp_arr: np.ndarray, act_arr: np.ndarray) -> PositionResult:
        """Detect elements in both images and compute position match score."""
        h, w = exp_arr.shape[:2]
        img_area = h * w

        exp_boxes = self._detect_elements(exp_arr, img_area)
        act_boxes = self._detect_elements(act_arr, img_area)

        if not exp_boxes and not act_boxes:
            return PositionResult(
                score=1.0,
                expected_boxes=exp_boxes,
                actual_boxes=act_boxes,
            )

        if not exp_boxes or not act_boxes:
            # Nothing to match against
            missing = exp_boxes if not act_boxes else []
            extra = act_boxes if not exp_boxes else []
            penalty = len(missing) * self._miss_pen + len(extra) * self._extra_pen
            score = max(0.0, 1.0 - penalty)
            return PositionResult(
                score=score,
                missing_elements=missing,
                extra_elements=extra,
                expected_boxes=exp_boxes,
                actual_boxes=act_boxes,
            )

        matched, missing, extra = self._match(exp_boxes, act_boxes, w, h)
        score = self._compute_score(matched, missing, extra, w, h)

        return PositionResult(
            score=score,
            matched_pairs=matched,
            missing_elements=missing,
            extra_elements=extra,
            expected_boxes=exp_boxes,
            actual_boxes=act_boxes,
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_elements(self, arr: np.ndarray, img_area: int) -> list[BBox]:
        """Canny edge detection → contour extraction → bbox filtering."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self._canny_t1, self._canny_t2)

        # Dilate to close small gaps between nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[BBox] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self._min_area:
                continue
            if area > img_area * self._max_area_ratio:
                continue
            ar = w / h if h > 0 else 0
            if ar < self._min_ar or ar > self._max_ar:
                continue
            boxes.append(BBox(x=x, y=y, w=w, h=h))

        # Merge heavily overlapping boxes (IoU > 0.7) via NMS-style pass
        boxes = self._nms(boxes, iou_thresh=0.7)
        return boxes

    def _nms(self, boxes: list[BBox], iou_thresh: float = 0.7) -> list[BBox]:
        """Non-maximum suppression — merge overlapping boxes, keep largest."""
        if not boxes:
            return boxes
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        kept: list[BBox] = []
        suppressed = [False] * len(boxes)
        for i, b in enumerate(boxes):
            if suppressed[i]:
                continue
            kept.append(b)
            for j in range(i + 1, len(boxes)):
                if not suppressed[j] and b.iou(boxes[j]) > iou_thresh:
                    suppressed[j] = True
        return kept

    # ── Matching ──────────────────────────────────────────────────────────────

    def _match(
        self,
        exp_boxes: list[BBox],
        act_boxes: list[BBox],
        img_w: int,
        img_h: int,
    ) -> tuple[list[MatchedPair], list[BBox], list[BBox]]:
        """Hungarian optimal assignment between expected and actual elements."""
        n = len(exp_boxes)
        m = len(act_boxes)

        # Build cost matrix (n x m), cost = 1 - similarity
        cost = np.ones((n, m), dtype=np.float64)
        for i, eb in enumerate(exp_boxes):
            for j, ab in enumerate(act_boxes):
                iou_score = eb.iou(ab)
                cent_score = 1.0 - eb.centroid_dist(ab, img_w, img_h)
                # Heavily penalize matching elements with vastly different sizes, but forgive small jitter
                area_eb = eb.w * eb.h
                area_ab = ab.w * ab.h
                size_ratio = min(area_eb, area_ab) / max(1, max(area_eb, area_ab))
                
                if size_ratio < 0.2:
                    size_penalty = 0.0
                else:
                    size_penalty = min(1.0, size_ratio + 0.3)
                
                similarity = (self._iou_w * iou_score + self._cent_w * cent_score) * size_penalty
                cost[i, j] = 1.0 - similarity

        row_ind, col_ind = linear_sum_assignment(cost)

        matched: list[MatchedPair] = []
        matched_exp = set()
        matched_act = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self._cost_thresh:
                continue  # too dissimilar — treat as unmatched
            eb = exp_boxes[r]
            ab = act_boxes[c]
            iou_val = eb.iou(ab)
            dx = abs(eb.cx - ab.cx) / img_w
            dy = abs(eb.cy - ab.cy) / img_h
            dw = abs(eb.w - ab.w) / img_w
            dh = abs(eb.h - ab.h) / img_h
            pos_delta = (dx + dy) / 2
            severity = (
                "match" if pos_delta < 0.02 else
                "drift" if pos_delta < 0.08 else
                "mismatch"
            )
            matched.append(MatchedPair(
                expected_bbox=eb,
                actual_bbox=ab,
                iou=iou_val,
                delta_x=dx,
                delta_y=dy,
                delta_w=dw,
                delta_h=dh,
                severity=severity,
            ))
            matched_exp.add(r)
            matched_act.add(c)

        missing = [b for i, b in enumerate(exp_boxes) if i not in matched_exp]
        extra = [b for j, b in enumerate(act_boxes) if j not in matched_act]
        return matched, missing, extra

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_score(
        self,
        matched: list[MatchedPair],
        missing: list[BBox],
        extra: list[BBox],
        img_w: int,
        img_h: int,
    ) -> float:
        """Area-Weighted F1-based position score (Precision/Recall).

        Score = Area_F1_Score × position_quality

        By weighting by bounding box area instead of raw counts, we prevent 
        microscopic noise (e.g. anti-aliasing artifacts extracted by Canny) 
        from dragging down the score of an otherwise structurally identical UI.
        """
        matched_exp_area = sum(p.expected_bbox.w * p.expected_bbox.h for p in matched)
        matched_act_area = sum(p.actual_bbox.w * p.actual_bbox.h for p in matched)
        
        missing_area = sum(b.w * b.h for b in missing)
        extra_area = sum(b.w * b.h for b in extra)
        
        total_exp_area = matched_exp_area + missing_area
        total_act_area = matched_act_area + extra_area

        if total_exp_area == 0 and total_act_area == 0:
            return 1.0
        if matched_exp_area == 0 or matched_act_area == 0:
            return 0.0

        recall = matched_exp_area / max(1, total_exp_area)
        precision = matched_act_area / max(1, total_act_area)
        
        f1_score = 2 * (precision * recall) / max(1e-6, precision + recall)

        # Quality of positional agreement for matched elements (area-weighted delta)
        total_matched_area = matched_exp_area + matched_act_area
        if total_matched_area > 0:
            avg_delta = sum(
                ((p.delta_x + p.delta_y) / 2) * (p.expected_bbox.w * p.expected_bbox.h + p.actual_bbox.w * p.actual_bbox.h)
                for p in matched
            ) / total_matched_area
        else:
            avg_delta = 1.0

        position_quality = math.exp(-2.0 * avg_delta)

        score = f1_score * position_quality
        return float(max(0.0, min(1.0, score)))
