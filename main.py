"""Web application for the notebook's U²-Net salient object pipeline."""

from __future__ import annotations

import base64
import io
import ipaddress
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from rembg import new_session, remove
from skimage import data as sample_data

MODEL_NAME = "u2net"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGES = 3
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once per server process and reuse the downloaded weights/session.
    app.state.model_session = new_session(MODEL_NAME)
    yield


app = FastAPI(title="Salient Object Detection using U²-Net", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


class UrlRequest(BaseModel):
    urls: list[str] = Field(min_length=3, max_length=3)


def _decode_image(content: bytes) -> Image.Image:
    if not content:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Each image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(content)) as probe:
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            probe.verify()
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="A supplied file is not a valid image.") from exc


def _sample_images() -> list[tuple[str, Image.Image]]:
    # These are the three built-in samples selected in notebook Method 1.
    # Convert each skimage array into a PIL Image, as in the notebook.
    samples = (
        ("Astronaut", sample_data.astronaut()),
        ("Cat", sample_data.chelsea()),
        ("Coffee", sample_data.coffee()),
    )
    return [(label, Image.fromarray(array).convert("RGB")) for label, array in samples]


def _segment_image(original: Image.Image) -> dict[str, str]:
    """Reuse segment_image's alpha extraction, threshold 128, and white composite."""
    try:
        result_rgba = remove(original, session=app.state.model_session)
        mask = np.array(result_rgba.split()[-1])
        binary_mask = (mask > 128).astype(np.uint8) * 255
        mask_3ch = np.dstack([binary_mask] * 3) / 255.0
        original_np = np.array(original)
        white_bg = np.ones_like(original_np) * 255
        foreground = (original_np * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        return {
            "original": _encode_png(original),
            "mask": _encode_png(Image.fromarray(binary_mask)),
            "foreground": _encode_png(Image.fromarray(foreground)),
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid sending model/runtime stack traces to the browser.
        raise HTTPException(status_code=500, detail="Image processing failed. Please try again.") from exc


def _encode_png(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _process_images(images: list[tuple[str, Image.Image]]) -> dict[str, list[dict[str, str]]]:
    if not images:
        raise HTTPException(status_code=400, detail="Provide at least one image.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail="Process up to three images at a time.")
    return {
        "results": [
            {"label": label, **_segment_image(image)}
            for label, image in images
        ]
    }


@app.post("/process-samples")
def process_samples() -> dict[str, list[dict[str, str]]]:
    """Process the astronaut, cat, and coffee images built into scikit-image."""
    return _process_images(_sample_images())


@app.post("/process-upload")
def process_upload(images: Annotated[list[UploadFile], File()]) -> dict[str, list[dict[str, str]]]:
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail="Choose up to three images.")
    decoded = []
    total_size = 0
    for index, upload in enumerate(images, start=1):
        content = upload.file.read(MAX_IMAGE_BYTES + 1)
        total_size += len(content)
        if total_size > MAX_IMAGE_BYTES * MAX_IMAGES:
            raise HTTPException(status_code=413, detail="The combined upload must be 30 MB or smaller.")
        decoded.append((upload.filename or f"Uploaded image {index}", _decode_image(content)))
    return _process_images(decoded)


def _validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password:
            raise ValueError
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addresses = {entry[4][0] for entry in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)}
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError
    except (ValueError, OSError, socket.gaierror):
        raise HTTPException(status_code=400, detail="Enter a valid public HTTP or HTTPS image URL.")


def _download_image(url: str) -> Image.Image:
    _validate_public_url(url)
    try:
        with requests.get(
            url,
            timeout=(5, 20),
            stream=True,
            allow_redirects=False,
            headers={"User-Agent": "SOD-Image-Processor/1.0"},
        ) as response:
            if 300 <= response.status_code < 400:
                raise HTTPException(status_code=400, detail="Use the direct image URL; redirects are not followed.")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="A URL did not return an image.")
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Each downloaded image must be 10 MB or smaller.")
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Each downloaded image must be 10 MB or smaller.")
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail="Could not download one of the image URLs.") from exc
    return _decode_image(bytes(content))


@app.post("/process-urls")
def process_urls(request: UrlRequest) -> dict[str, list[dict[str, str]]]:
    images = [
        (f"Internet image {index}", _download_image(url))
        for index, url in enumerate(request.urls, start=1)
    ]
    return _process_images(images)
