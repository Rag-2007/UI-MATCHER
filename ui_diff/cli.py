"""
cli.py — Typer-based CLI for ui-diff.

Usage:
    ui-diff compare expected.png actual.png \\
        --config config.yaml \\
        --ignore-regions ignore.json \\
        --output report.json \\
        --diff-image diff.png
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import typer

app = typer.Typer(
    name="ui-diff",
    help="UI Visual Match Validator — compare UI screenshots for position & color match.",
    add_completion=False,
)


@app.command("compare")
def compare_cmd(
    expected: pathlib.Path = typer.Argument(..., help="Path to expected/reference UI image."),
    actual: pathlib.Path = typer.Argument(..., help="Path to actual/built UI screenshot."),
    config: Optional[pathlib.Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML config (defaults to bundled config/default_weights.yaml).",
    ),
    ignore_regions: Optional[pathlib.Path] = typer.Option(
        None, "--ignore-regions", "-i",
        help="Path to JSON file with ignore regions: [{x, y, w, h}, ...]",
    ),
    output: Optional[pathlib.Path] = typer.Option(
        None, "--output", "-o",
        help="Save JSON report to this path.",
    ),
    diff_image: Optional[pathlib.Path] = typer.Option(
        None, "--diff-image", "-d",
        help="Save annotated diff image to this path.",
    ),
    threshold: float = typer.Option(
        85.0, "--threshold", "-t",
        help="Confidence threshold — exit code 1 if score is below this value.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed layer scores."),
) -> None:
    """Compare two UI images and output a confidence score (0–100)."""

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not expected.exists():
        typer.echo(f"[ERROR] Expected image not found: {expected}", err=True)
        raise typer.Exit(code=2)
    if not actual.exists():
        typer.echo(f"[ERROR] Actual image not found: {actual}", err=True)
        raise typer.Exit(code=2)

    # ── Load ignore regions ───────────────────────────────────────────────────
    regions = None
    if ignore_regions is not None:
        if not ignore_regions.exists():
            typer.echo(f"[ERROR] Ignore-regions file not found: {ignore_regions}", err=True)
            raise typer.Exit(code=2)
        with open(ignore_regions) as f:
            regions = json.load(f)

    # ── Run comparison ────────────────────────────────────────────────────────
    from ui_diff import compare

    typer.echo(f"Comparing: {expected.name}  ←→  {actual.name}")
    result = compare(
        expected=expected,
        actual=actual,
        config=config,
        ignore_regions=regions,
        diff_image_path=diff_image,
        output_path=output,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    score = result.confidence_score
    bar_filled = int(score / 2)
    bar = "█" * bar_filled + "░" * (50 - bar_filled)
    color = typer.colors.GREEN if score >= threshold else typer.colors.RED

    typer.echo()
    typer.echo(f"  Confidence Score: ", nl=False)
    typer.secho(f"{score:.1f} / 100", fg=color, bold=True)
    typer.echo(f"  [{bar}]")
    typer.echo()

    if verbose or score < threshold:
        typer.echo("  Layer Scores:")
        for layer, val in result.layers.items():
            typer.echo(f"    {layer:20s}: {val:.4f}")
        typer.echo()

    if result.issues:
        typer.echo(f"  Issues ({len(result.issues)}):")
        for issue in result.issues[:20]:  # cap display at 20
            icon = "⚠" if issue.get("severity") in ("drift", "mismatch") else "✗"
            typer.echo(f"    {icon} [{issue['type']}] {issue.get('message', '')}")
        if len(result.issues) > 20:
            typer.echo(f"    ... and {len(result.issues) - 20} more (see JSON report)")
        typer.echo()

    if output:
        typer.echo(f"  JSON report saved: {output}")
    if diff_image:
        typer.echo(f"  Diff image saved:  {diff_image}")

    if result.has_critical_failures:
        typer.secho("  [CRITICAL] One or more critical elements are missing!", fg=typer.colors.RED, bold=True)

    # Exit code 1 if below threshold, 0 if passing
    if score < threshold:
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
