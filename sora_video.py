"""
OpenAI Videos API (Sora): create an MP4 from a text prompt, optionally guided
by an image that must match the target resolution.

Requires OPENAI_API_KEY in the environment (see .env).
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

VideoModel = Literal["sora-2", "sora-2-pro"]
VideoSize = Literal["720x1280", "1280x720", "1024x1792", "1792x1024"]
VideoSeconds = Literal["4", "8", "12"]

__all__ = [
    "VideoModel",
    "VideoSize",
    "generate_with_sora",
    "seconds_for_duration_slider",
]


def seconds_for_duration_slider(duration_sec: float) -> VideoSeconds:
    """Sora accepts 4, 8, or 12 second clips; pick the closest to the UI slider."""
    allowed = (4, 8, 12)
    d = float(duration_sec)
    best = min(allowed, key=lambda x: abs(x - d))
    return str(best)  # type: ignore[return-value]


def _parse_size(size: VideoSize) -> tuple[int, int]:
    w_s, h_s = size.lower().split("x")
    return int(w_s), int(h_s)


def prepare_input_reference(rgb_uint8: np.ndarray, size: VideoSize) -> tuple[str, io.BytesIO, str]:
    """Resize/cover to exact video dimensions; return a multipart file tuple for the API."""
    w, h = _parse_size(size)
    pil = Image.fromarray(rgb_uint8, mode="RGB").convert("RGB")
    pil.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = (w - pil.width) // 2
    y = (h - pil.height) // 2
    canvas.paste(pil, (x, y))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return ("reference.png", buf, "image/png")


def generate_with_sora(
    *,
    prompt: str,
    image_rgb: np.ndarray | None,
    use_image_reference: bool,
    duration_sec: float,
    model: VideoModel,
    size: VideoSize,
) -> tuple[str | None, str]:
    """
    Run Sora job, poll until complete, save MP4 under outputs/.
    Returns (path or None, status message).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "Set OPENAI_API_KEY (env or .env file) to use Sora."

    text = (prompt or "").strip()
    if not text:
        return None, "Sora needs a text prompt describing the shot, motion, lighting, etc."

    if use_image_reference and image_rgb is None:
        return None, "Enable reference only when you have uploaded an image."

    try:
        from openai import OpenAI
    except ImportError:
        return None, "Install the OpenAI SDK: pip install openai"

    seconds = seconds_for_duration_slider(duration_sec)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"sora_{stamp}.mp4"

    # Jobs can run a long time; downloads may be large.
    client = OpenAI(api_key=api_key, timeout=600.0)
    kwargs: dict = {
        "model": model,
        "prompt": text,
        "seconds": seconds,
        "size": size,
        "poll_interval_ms": 10_000,
    }
    if use_image_reference and image_rgb is not None:
        kwargs["input_reference"] = prepare_input_reference(image_rgb, size)

    try:
        video = client.videos.create_and_poll(**kwargs)
    except Exception as exc:
        return None, f"Sora request failed: {exc}"

    if video.status == "failed":
        err = getattr(video, "error", None)
        msg = getattr(err, "message", None) or str(err) or "generation failed"
        return None, f"Sora failed: {msg}"

    if video.status != "completed":
        return None, f"Unexpected status: {video.status}"

    try:
        client.videos.download_content(video.id, variant="video").write_to_file(str(out_path))
    except Exception as exc:
        return None, f"Video ready but download failed: {exc}"

    return str(out_path), f"Sora saved ({seconds}s, {size}): {out_path}"

