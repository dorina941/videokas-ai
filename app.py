"""
VideoKas AI - simple Gradio demo: turn a still image into a short MP4
with zoom + pan locally, or generate clips with OpenAI Sora (Videos API).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import gradio as gr
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from sora_video import VideoModel, VideoSize, generate_with_sora

# Where generated files go (folder is created if missing)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24
# Magnification at start / end of clip (bigger gap = more visible motion)
MAG_START = 1.22
MAG_END = 1.65


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    Gradio may hand back uint8, float32 [0–255], or float [0–1]. PIL needs uint8 RGB.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("Expected HxWx3 or HxWx4 image")
    rgb = rgb[:, :, :3]
    if rgb.dtype == np.uint8:
        out = rgb
    elif np.issubdtype(rgb.dtype, np.floating) and float(rgb.max()) <= 1.0 + 1e-6:
        out = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    else:
        out = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _motion_frame(rgb_uint8: np.ndarray, progress: float) -> np.ndarray:
    """
    Ken Burns–style motion: scale up, then crop a sliding window so the picture
    really travels across the frame (not a barely-visible nudge).
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    w, h = pil.size
    mag = MAG_START + (MAG_END - MAG_START) * progress
    W = max(w + 1, int(round(w * mag)))
    H = max(h + 1, int(round(h * mag)))
    big = pil.resize((W, H), Image.Resampling.LANCZOS)
    max_x = max(0, W - w)
    max_y = max(0, H - h)
    left = int(round(progress * max_x))
    top = int(round(progress * max_y))
    return np.asarray(big.crop((left, top, left + w, top + h)))


def _generate_local(image, prompt: str, duration_sec: float) -> tuple[str | None, str]:
    """Ken Burns-style clip from a single image (prompt ignored)."""
    if image is None:
        return None, "Please upload an image first."

    _ = prompt

    try:
        base = _as_uint8_rgb(image)
    except ValueError:
        return None, "Image must be RGB or RGBA."

    n_frames = max(1, int(round(duration_sec * FPS)))
    if n_frames == 1:
        progresses = np.array([0.0], dtype=np.float64)
    else:
        progresses = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    frames = [_motion_frame(base, float(p)) for p in progresses]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_mp4 = OUTPUT_DIR / f"video_{timestamp}.mp4"
    path_gif = OUTPUT_DIR / f"video_{timestamp}.gif"

    # MP4 needs FFmpeg (install `imageio-ffmpeg` or system FFmpeg). GIF always works.
    try:
        # macro_block_size avoids some H.264 size warnings; quality=8 is a reasonable default
        imageio.mimsave(
            str(path_mp4),
            frames,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        )
        return str(path_mp4), f"Saved: {path_mp4}"
    except Exception:
        try:
            imageio.mimsave(str(path_gif), frames, fps=FPS)
            return str(path_gif), f"MP4 not available; saved GIF instead: {path_gif}"
        except Exception as exc:
            return None, f"Could not write video: {exc}"


def generate_video(
    image,
    prompt: str,
    duration_sec: float,
    mode: str,
    model: str,
    size: str,
    use_image_reference: bool,
) -> tuple[str | None, str]:
    """
    Local path: zoom + pan on the uploaded image.
    Sora path: OpenAI Videos API (async job, can take minutes; duration mapped to 4/8/12 s).
    """
    if mode.startswith("Local"):
        return _generate_local(image, prompt, duration_sec)

    img_rgb: np.ndarray | None
    if image is not None:
        try:
            img_rgb = _as_uint8_rgb(image)
        except ValueError:
            return None, "Image must be RGB or RGBA."
    else:
        img_rgb = None

    if use_image_reference and img_rgb is None:
        return None, "Upload an image or turn off “Use image as first frame”."

    return generate_with_sora(
        prompt=prompt,
        image_rgb=img_rgb,
        use_image_reference=use_image_reference,
        duration_sec=duration_sec,
        model=cast(VideoModel, model),
        size=cast(VideoSize, size),
    )


def main() -> None:
    with gr.Blocks(title="VideoKas AI") as demo:
        gr.Markdown(
            "# VideoKas AI\n"
            "**Local:** zoom + pan on your image.\n\n"
            "**OpenAI Sora:** text-to-video (and optionally image-conditioned first frame). "
            "Requires `OPENAI_API_KEY` and may take several minutes. "
            "See OpenAI docs: input images with **human faces can be rejected**."
        )

        mode_in = gr.Radio(
            choices=["Local (zoom + pan)", "OpenAI Sora (API)"],
            value="Local (zoom + pan)",
            label="Generation mode",
        )
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload image")
            with gr.Column():
                prompt_in = gr.Textbox(
                    label="Prompt",
                    placeholder="Local mode: optional. Sora: required (shot type, action, light, camera motion…).",
                    lines=3,
                )
                duration = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Duration hint (local: exact seconds · Sora: mapped to 4, 8 or 12 s)",
                )
                model_in = gr.Dropdown(
                    choices=["sora-2", "sora-2-pro"],
                    value="sora-2",
                    label="Sora model (API mode only)",
                )
                size_in = gr.Dropdown(
                    choices=["1280x720", "720x1280", "1024x1792", "1792x1024"],
                    value="1280x720",
                    label="Output size (API mode; reference image is fitted to this)",
                )
                use_ref = gr.Checkbox(
                    value=True,
                    label="Sora: use uploaded image as first frame (input_reference)",
                )
                gen_btn = gr.Button("Generate video", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

        video_out = gr.Video(label="Generated video")

        gen_btn.click(
            fn=generate_video,
            inputs=[image_in, prompt_in, duration, mode_in, model_in, size_in, use_ref],
            outputs=[video_out, status],
        )

    demo.launch()


if __name__ == "__main__":
    main()
