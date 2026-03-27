"""
Optional step: ask OpenAI image edit to produce an “eyes closed” variant,
then the video pipeline can cross-fade between open and closed for a blink.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Tuple

import numpy as np
from PIL import Image

_EDIT_PROMPT_BASE = (
    "Edit this photograph: keep the exact same scene, camera angle, lighting, and identity. "
    "The subject’s eyes are gently closed as in a calm, natural blink—upper and lower eyelids "
    "meet realistically. Do not change pose, clothes, hair, or background. Photorealistic."
)


def _rgb_to_png_bytes(rgb_uint8: np.ndarray) -> tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return ("source.png", buf, "image/png")


def make_eyes_closed_variant(
    rgb_uint8: np.ndarray,
    user_hint: str = "",
) -> Tuple[np.ndarray | None, str]:
    """
    Calls OpenAI Images API edit (gpt-image-1). Returns (H,W,3) uint8 matching input size,
    or (None, error message).
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None, "Parpadeo IA: falta OPENAI_API_KEY en .env"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "Parpadeo IA: pip install openai"

    prompt = _EDIT_PROMPT_BASE
    hint = (user_hint or "").strip()
    if hint:
        prompt = f"{_EDIT_PROMPT_BASE} Extra direction from user: {hint}"

    h0, w0 = rgb_uint8.shape[0], rgb_uint8.shape[1]
    if max(h0, w0) > 2048:
        scale = 2048 / max(h0, w0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        small = np.asarray(
            Image.fromarray(rgb_uint8, mode="RGB").resize((nw, nh), Image.Resampling.LANCZOS)
        )
    else:
        small = rgb_uint8

    file_tuple = _rgb_to_png_bytes(small)
    client = OpenAI(api_key=key, timeout=180.0)

    try:
        result = client.images.edit(
            model="gpt-image-1",
            image=file_tuple,
            prompt=prompt,
            input_fidelity="high",
            size="auto",
            output_format="png",
            quality="high",
            n=1,
        )
    except Exception as exc:
        return None, f"Parpadeo IA: la API rechazó o falló la edición ({exc}). Prueba otra foto o desactiva esta opción."

    if not result.data:
        return None, "Parpadeo IA: la API no devolvió imagen."

    b64 = getattr(result.data[0], "b64_json", None)
    if not b64:
        return None, "Parpadeo IA: respuesta sin base64."

    try:
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return None, f"Parpadeo IA: no se pudo decodificar la imagen ({exc})."

    if pil.size != (w0, h0):
        pil = pil.resize((w0, h0), Image.Resampling.LANCZOS)

    return np.ascontiguousarray(np.asarray(pil)), "OK"

