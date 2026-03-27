# 🎬 VideoKas AI

Aplicación en Python para generar vídeos cortos a partir de imágenes.

Este proyecto es un MVP (Minimum Viable Product) que permite subir una imagen, definir un prompt y generar un vídeo básico repitiendo frames.

---

## 🚀 Demo

Sube una imagen, selecciona duración y genera un vídeo en segundos.

---

## 🧠 Funcionalidades

- Subida de imagen
- Input de prompt (preparado para IA futura)
- Control de duración del vídeo
- Generación de vídeo (MP4 o GIF)
- Guardado automático en carpeta `outputs`

---

## 🛠️ Tecnologías

- Python 3
- Gradio
- imageio
- numpy
- moviepy
- OpenAI (`openai`) — Sora / Videos API (opcional)
- python-dotenv

### OpenAI Sora (opcional)

1. Crea `.env` en la raíz con `OPENAI_API_KEY=tu_clave`.
2. En la app elige **OpenAI Sora (API)**. El **prompt es obligatorio** (plano, acción, luz, movimiento de cámara…).
3. La duración del slider se mapea al valor permitido más cercano: **4, 8 u 12 s**.
4. Revisa la documentación de OpenAI: las **imágenes de referencia con rostros humanos pueden rechazarse**; hay límites de uso de personajes reales y contenido.

---

## ▶️ Cómo ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone https://github.com/dorina941/videokas-ai.git
cd videokas-ai