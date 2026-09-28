"""Streamlit app for salient object detection with rembg/U²-Net."""

from __future__ import annotations

import io
import ipaddress
import os
import socket
from urllib.parse import urlsplit

import numpy as np
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
from rembg import new_session, remove
from skimage import data as sample_data

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGES = 3
try:
    configured_model = st.secrets.get("SOD_MODEL", os.getenv("SOD_MODEL", "u2netp"))
except (FileNotFoundError, KeyError):
    configured_model = os.getenv("SOD_MODEL", "u2netp")
MODEL_NAME = str(configured_model).strip().lower()
if MODEL_NAME not in {"u2net", "u2netp"}:
    MODEL_NAME = "u2netp"

st.set_page_config(page_title="Salient Object Detection", page_icon="✂️", layout="wide")


@st.cache_resource(show_spinner="Downloading and loading the segmentation model…")
def get_model_session(model_name: str):
    """Load the weights once per running Streamlit process."""
    return new_session(model_name)


def decode_image(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("The image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Each image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(content)) as probe:
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large (maximum 25 megapixels).")
            probe.verify()
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("A supplied file is not a valid image.") from exc


def sample_images() -> list[tuple[str, Image.Image]]:
    return [
        ("Astronaut", Image.fromarray(sample_data.astronaut()).convert("RGB")),
        ("Cat", Image.fromarray(sample_data.chelsea()).convert("RGB")),
        ("Coffee", Image.fromarray(sample_data.coffee()).convert("RGB")),
    ]


def validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password:
            raise ValueError
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        addresses = {
            entry[4][0]
            for entry in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        }
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise ValueError
    except (ValueError, OSError, socket.gaierror) as exc:
        raise ValueError("Enter a valid public HTTP or HTTPS image URL.") from exc


def download_image(url: str) -> Image.Image:
    validate_public_url(url)
    try:
        with requests.get(
            url.strip(), timeout=(5, 20), stream=True, allow_redirects=False,
            headers={"User-Agent": "SOD-Image-Processor/1.0"},
        ) as response:
            if 300 <= response.status_code < 400:
                raise ValueError("Use the direct image URL; redirects are not followed.")
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").split(";", 1)[0].lower().startswith("image/"):
                raise ValueError("A URL did not return an image.")
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_IMAGE_BYTES:
                raise ValueError("Each downloaded image must be 10 MB or smaller.")
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise ValueError("Each downloaded image must be 10 MB or smaller.")
    except ValueError:
        raise
    except requests.RequestException as exc:
        raise ValueError("Could not download one of the image URLs.") from exc
    return decode_image(bytes(content))


def segment_image(original: Image.Image, session) -> tuple[Image.Image, Image.Image]:
    result_rgba = remove(original, session=session)
    mask = np.asarray(result_rgba.getchannel("A"))
    binary_mask = (mask > 128).astype(np.uint8) * 255
    mask_3ch = binary_mask[:, :, None] / 255.0
    original_np = np.asarray(original)
    foreground = (original_np * mask_3ch + 255 * (1 - mask_3ch)).astype(np.uint8)
    return Image.fromarray(binary_mask), Image.fromarray(foreground)


def show_results(images: list[tuple[str, Image.Image]]) -> None:
    try:
        with st.spinner(f"Processing {len(images)} image(s) with {MODEL_NAME}…"):
            session = get_model_session(MODEL_NAME)
            results = [(label, image, *segment_image(image, session)) for label, image in images]
    except Exception as exc:
        st.error(f"Could not load the model or process these images: {exc}")
        st.info("For a lower-memory deployment, use SOD_MODEL=u2netp (the default).")
        return

    for label, original, mask, foreground in results:
        st.subheader(label)
        columns = st.columns(3)
        for column, title, image in zip(columns, ("Original", "Binary mask", "Foreground on white"),
                                        (original, mask, foreground)):
            column.image(image, caption=title, use_container_width=True)
        st.divider()


st.title("Salient Object Detection")
st.write(
    "Extract the most visually prominent object from an image using U²-Net. "
    "The app returns the original, a binary mask, and the foreground on white."
)
st.caption(f"Model: `{MODEL_NAME}` · Up to {MAX_IMAGES} images · 10 MB and 25 MP maximum per image")

sample_tab, upload_tab, url_tab = st.tabs(["Built-in samples", "Upload images", "Image URLs"])

with sample_tab:
    st.write("Run the astronaut, cat, and coffee samples bundled with scikit-image.")
    if st.button("Process sample images", type="primary"):
        show_results(sample_images())

with upload_tab:
    uploads = st.file_uploader(
        "Choose up to three image files", type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
    )
    if len(uploads) > MAX_IMAGES:
        st.error("Choose up to three images.")
    elif uploads and st.button("Process uploaded images", type="primary"):
        try:
            total_size = sum(upload.size for upload in uploads)
            if total_size > MAX_IMAGE_BYTES * MAX_IMAGES:
                raise ValueError("The combined upload must be 30 MB or smaller.")
            decoded = [(upload.name, decode_image(upload.getvalue())) for upload in uploads]
            show_results(decoded)
        except ValueError as exc:
            st.error(str(exc))

with url_tab:
    st.write("Provide three direct, publicly accessible image URLs. Redirects are not followed.")
    with st.form("url_form"):
        urls = [st.text_input(f"Image URL {index}", key=f"url_{index}") for index in range(1, 4)]
        submitted = st.form_submit_button("Process URLs", type="primary")
    if submitted:
        if not all(url.strip() for url in urls):
            st.error("Enter all three image URLs.")
        else:
            try:
                downloaded = [(f"Internet image {index}", download_image(url))
                              for index, url in enumerate(urls, start=1)]
                show_results(downloaded)
            except ValueError as exc:
                st.error(str(exc))
