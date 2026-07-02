"""
dom_capture.py — Optional Playwright ground-truth bounding-box capture.

When the actual UI is a live web page, this provides far more accurate
element bounding boxes than CV-based detection.
Falls back gracefully to CV detection when Playwright is unavailable or
when comparing static images (no live page URL).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .layers.position_layer import BBox


@dataclass
class DOMCaptureResult:
    boxes: list[BBox]
    source: str  # "playwright" | "fallback"


def capture_dom_boxes(
    url: str,
    selector: str = "*",
    timeout_ms: int = 10_000,
) -> DOMCaptureResult:
    """Capture element bounding boxes from a live page via Playwright.

    Args:
        url: URL of the live web page to inspect.
        selector: CSS selector to query (default all elements).
        timeout_ms: Navigation timeout in milliseconds.

    Returns:
        DOMCaptureResult with list of BBoxes and source="playwright".

    Raises:
        ImportError: if playwright is not installed.
        RuntimeError: if page navigation fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        )

    boxes: list[BBox] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=timeout_ms)
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as e:
            browser.close()
            raise RuntimeError(f"Failed to navigate to {url}: {e}")

        elements = page.query_selector_all(selector)
        for el in elements:
            try:
                bb = el.bounding_box()
                if bb and bb["width"] > 0 and bb["height"] > 0:
                    boxes.append(BBox(
                        x=int(bb["x"]),
                        y=int(bb["y"]),
                        w=int(bb["width"]),
                        h=int(bb["height"]),
                    ))
            except Exception:
                continue
        browser.close()

    return DOMCaptureResult(boxes=boxes, source="playwright")
