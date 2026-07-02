"""
api.py — FastAPI REST interface for ui-diff.

POST /compare
  multipart form: expected_image, actual_image, config (optional JSON string)
  → JSON report + diff image URL (base64 encoded in response)

GET /health
  → {"status": "ok"}
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import tempfile
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ui_diff import compare

app = FastAPI(
    title="UI Visual Match Validator API",
    description=(
        "Compare two UI screenshots for position and color match. "
        "Returns a confidence score (0–100) and a detailed diff report."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "ui-diff"}


@app.post("/compare", tags=["comparison"])
async def compare_images(
    expected_image: UploadFile = File(..., description="Reference / design mock image."),
    actual_image: UploadFile = File(..., description="Built UI screenshot."),
    config: Optional[str] = Form(None, description="JSON config overrides (optional)."),
    ignore_regions: Optional[str] = Form(
        None, description='JSON list of ignore regions: [{"x":0,"y":0,"w":100,"h":60}]'
    ),
    include_diff_image: bool = Form(
        True, description="Include base64-encoded diff image in response."
    ),
) -> JSONResponse:
    """Compare two UI images.

    Returns a JSON body with:
    - ``confidence_score``: 0–100
    - ``layers``: per-layer scores
    - ``issues``: list of detected problems
    - ``diff_image_b64``: base64 PNG diff overlay (if include_diff_image=True)
    """
    # ── Load images from upload ───────────────────────────────────────────────
    try:
        exp_bytes = await expected_image.read()
        act_bytes = await actual_image.read()
        exp_pil = Image.open(io.BytesIO(exp_bytes)).convert("RGB")
        act_pil = Image.open(io.BytesIO(act_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read images: {e}")

    # ── Parse optional config / ignore regions ────────────────────────────────
    cfg_dict = None
    if config:
        try:
            cfg_dict = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

    regions = None
    if ignore_regions:
        try:
            regions = json.loads(ignore_regions)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid ignore_regions JSON: {e}")

    # ── Run comparison ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        diff_path = pathlib.Path(tmpdir) / "diff.png" if include_diff_image else None

        try:
            result = compare(
                expected=exp_pil,
                actual=act_pil,
                config=cfg_dict,
                ignore_regions=regions,
                diff_image_path=diff_path,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

        # ── Build response ────────────────────────────────────────────────────
        response = result.to_dict()
        response.pop("diff_image_path", None)
        response.pop("report_path", None)

        if include_diff_image and diff_path and diff_path.exists():
            diff_b64 = base64.b64encode(diff_path.read_bytes()).decode("utf-8")
            response["diff_image_b64"] = diff_b64
            response["diff_image_mime"] = "image/png"

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
