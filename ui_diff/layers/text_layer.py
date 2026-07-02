"""
layers/text_layer.py — Text/OCR comparison layer.

OFF BY DEFAULT. Text content matching is never part of the default score
(see spec: dynamic text causes constant false mismatches).

This stub exists for future optional use. Enable via config or CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextResult:
    score: float = 1.0    # always 1.0 when disabled — no penalty
    enabled: bool = False
    message: str = "Text layer is disabled by default."


class TextLayer:
    """Stub text/OCR layer. Not included in score fusion by default."""

    def __init__(self, cfg: dict):
        self._enabled = cfg.get("text", {}).get("enabled", False)

    def run(self, exp_arr, act_arr) -> TextResult:
        if not self._enabled:
            return TextResult()
        # Future: integrate easyocr / pytesseract here
        raise NotImplementedError(
            "Text layer is not yet implemented. "
            "Enable it only for static-copy validation where dynamic text is masked."
        )
