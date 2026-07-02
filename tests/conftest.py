"""
tests/conftest.py — Shared fixtures for ui-diff tests.

All images are generated programmatically — no external files required.
Each fixture creates a pair (expected, actual) as numpy uint8 arrays.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Canvas helpers ────────────────────────────────────────────────────────────

def _white_canvas(w: int = 800, h: int = 600) -> np.ndarray:
    """Create a white RGB canvas."""
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _draw_rect(
    canvas: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: tuple[int, int, int] = (30, 80, 200),
) -> np.ndarray:
    """Draw a filled rectangle on a copy of canvas."""
    out = canvas.copy()
    out[y : y + h, x : x + w] = color
    return out


# ── Fixture factories ─────────────────────────────────────────────────────────

@pytest.fixture
def near_identical_pair() -> tuple[np.ndarray, np.ndarray]:
    """A pair of nearly identical UI layouts.

    Expected: white canvas with 3 colored blocks.
    Actual:   same layout with tiny ±1px sub-pixel noise (anti-alias simulation).
    Both should score >= 85.
    """
    expected = _white_canvas()
    expected = _draw_rect(expected, 50, 50, 200, 80, color=(30, 80, 200))   # blue header
    expected = _draw_rect(expected, 50, 160, 120, 60, color=(60, 180, 75))  # green button
    expected = _draw_rect(expected, 50, 250, 300, 150, color=(220, 220, 220))  # grey card

    # Actual: add tiny 1px noise
    rng = np.random.default_rng(42)
    noise = rng.integers(-2, 3, expected.shape, dtype=np.int16)
    actual = np.clip(expected.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return expected, actual


@pytest.fixture
def shifted_element_pair() -> tuple[np.ndarray, np.ndarray]:
    """A pair where two elements are substantially shifted.

    Expected: 3 blocks.
    Actual:   first block shifted 100px right (well beyond IoU match range),
              second block shifted 100px down.
    Should detect position issues and overall confidence < 85.
    """
    expected = _white_canvas(w=400, h=400)
    expected = _draw_rect(expected, 10, 10, 80, 40, color=(30, 80, 200))    # block A at left
    expected = _draw_rect(expected, 10, 70, 60, 30, color=(60, 180, 75))    # block B
    expected = _draw_rect(expected, 10, 120, 120, 60, color=(220, 220, 220)) # block C

    actual = _white_canvas(w=400, h=400)
    # Block A shifted 110px right — zero IoU with expected position
    actual = _draw_rect(actual, 120, 10, 80, 40, color=(30, 80, 200))
    # Block B shifted 110px down — zero IoU with expected position
    actual = _draw_rect(actual, 10, 180, 60, 30, color=(60, 180, 75))
    actual = _draw_rect(actual, 10, 120, 120, 60, color=(220, 220, 220))    # block C same

    return expected, actual


@pytest.fixture
def wrong_color_pair() -> tuple[np.ndarray, np.ndarray]:
    """A pair where one element's color is wrong.

    Expected: blue header (30, 80, 200).
    Actual:   same header position but red (200, 30, 30).
    Should detect a color_mismatch issue.
    """
    expected = _white_canvas()
    expected = _draw_rect(expected, 50, 50, 200, 80, color=(30, 80, 200))   # blue header
    expected = _draw_rect(expected, 50, 160, 120, 60, color=(60, 180, 75))  # green button
    expected = _draw_rect(expected, 50, 250, 300, 150, color=(220, 220, 220))  # grey card

    actual = _white_canvas()
    actual = _draw_rect(actual, 50, 50, 200, 80, color=(200, 30, 30))       # RED header (wrong!)
    actual = _draw_rect(actual, 50, 160, 120, 60, color=(60, 180, 75))      # green button same
    actual = _draw_rect(actual, 50, 250, 300, 150, color=(220, 220, 220))   # grey card same

    return expected, actual


@pytest.fixture
def missing_element_pair() -> tuple[np.ndarray, np.ndarray]:
    """A pair where one element is completely missing in actual.

    Expected: 3 blocks.
    Actual:   only 2 blocks (the green button is absent).
    Should detect a missing_element issue and score < 85.
    """
    expected = _white_canvas()
    expected = _draw_rect(expected, 50, 50, 200, 80, color=(30, 80, 200))   # blue header
    expected = _draw_rect(expected, 50, 160, 120, 60, color=(60, 180, 75))  # green button
    expected = _draw_rect(expected, 50, 250, 300, 150, color=(220, 220, 220))  # grey card

    actual = _white_canvas()
    actual = _draw_rect(actual, 50, 50, 200, 80, color=(30, 80, 200))       # blue header same
    # green button MISSING
    actual = _draw_rect(actual, 50, 250, 300, 150, color=(220, 220, 220))   # grey card same

    return expected, actual
