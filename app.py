"""
VideoKas AI - simple Gradio demo: turn a still image into a short MP4
with a gentle zoom-in (Ken Burns) so the clip is not a frozen frame.
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

# Video looks smooth enough for a demo; duration sets how long the clip runs
FPS = 24
# How much we “zoom in” from first to last frame (1.0 = no motion, 1.12 ≈ 12 % tighter crop)
ZOOM_END = 1.12


def _zoom_frame(rgb: np.ndarray, progress: float) -> np.ndarray:
    """
    Zoom into the center: progress 0 = full frame, 1 = strongest zoom (see ZOOM_END).
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    zoom = 1.0 + (ZOOM_END - 1.0) * progress

    pil = Image.fromarray(rgb)
    w, h = pil.size
    nw = max(1, int(round(w * zoom)))
    nh = max(1, int(round(h * zoom)))
    resized = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    cropped = resized.crop((left, top, left + w, top + h))
    return np.asarray(cropped)


def generate_video(image, prompt: str, duration_sec: float) -> tuple[str | None, str]:
    """
    Build a video with a slow center zoom so pixels change over time (not a static hold).
    Returns (path to video file or None, status message).
    """
    if image is None:
        return None, "Please upload an image first."

    _ = prompt  # reserved for future AI features

    # Gradio may pass a numpy array (H, W, C) uint8
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        return None, "Image must be RGB or RGBA."

    # Drop alpha if present — MP4 expects RGB
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    n_frames = max(1, int(round(duration_sec * FPS)))
    if n_frames == 1:
        progresses = np.array([0.0], dtype=np.float64)
    else:
        progresses = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    frames = [_zoom_frame(frame, p) for p in progresses]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_mp4 = OUTPUT_DIR / f"video_{timestamp}.mp4"
    path_gif = OUTPUT_DIR / f"video_{timestamp}.gif"

    # MP4 needs FFmpeg (install `imageio-ffmpeg` or system FFmpeg). GIF always works.
    try:
        imageio.mimsave(str(path_mp4), frames, fps=FPS, codec="libx264")
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
            "Turn an image into a short video with a **slow zoom-in** (MVP demo)."
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
