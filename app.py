"""
VideoKas AI - simple Gradio demo: turn a still image into a short MP4
with zoom + pan (Ken Burns) so every frame differs clearly from the next.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import numpy as np
from PIL import Image

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


def generate_video(image, prompt: str, duration_sec: float) -> tuple[str | None, str]:
    """
    Build a video with zoom + pan so motion is obvious in the player and on disk.
    Returns (path to video file or None, status message).
    """
    if image is None:
        return None, "Please upload an image first."

    _ = prompt  # reserved for future AI features

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


def main() -> None:
    with gr.Blocks(title="VideoKas AI") as demo:
        gr.Markdown(
            "# VideoKas AI\n"
            "Turn an image into a short video with **zoom + pan** (clear motion; MVP demo)."
        )

        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload image")
            with gr.Column():
                prompt_in = gr.Textbox(
                    label="Prompt (optional, for future AI)",
                    placeholder="Describe the motion or style you want…",
                    lines=2,
                )
                duration = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Duration (seconds)",
                )
                gen_btn = gr.Button("Generate video", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

        video_out = gr.Video(label="Generated video")

        gen_btn.click(
            fn=generate_video,
            inputs=[image_in, prompt_in, duration],
            outputs=[video_out, status],
        )

    demo.launch()


if __name__ == "__main__":
    main()
