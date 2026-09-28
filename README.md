# Salient Object Detection using U²-Net

## Problem Statement

Salient Object Detection (SOD) identifies the most visually prominent object in an image and produces a pixel-level foreground/background mask. This project uses the pre-trained U²-Net model through `rembg` for inference only; it does not train a model.

## Objective

Run the notebook's three-image segmentation workflow as a web application. Choose built-in sample images, upload images from the computer, or provide three Internet image addresses. For each image, the page displays the original, binary segmentation mask, and foreground on a white background.

## Model and Workflow

The application uses the pre-trained `u2net` model session from `rembg`. It loads the model once when the server starts and reuses the session.

Input image → PIL/Pillow RGB → U²-Net inference → predicted alpha mask → binary mask → foreground composited on white.

The segmentation logic from the notebook is preserved: `remove(original, session=session)`, take the alpha channel, apply the existing strict threshold comparison `mask > 128`, then composite foreground pixels on a white background. The threshold remains part of the existing algorithm; it is not a user setting.

## The Three Input Options

1. **Built-in PIL sample images:** The server loads the same astronaut, cat, and coffee samples used in notebook Method 1 (`skimage.data`) and converts them into PIL images. No upload is needed.
2. **Upload from my system:** Select one to three image files with the file picker. This option requires a manual upload.
3. **Enter three image URLs:** Paste three direct, publicly accessible HTTP or HTTPS image addresses.

Each option has a **Proceed** button. All processed images display an original, a segmentation mask, and an extracted foreground.

## Technologies

Python 3.11, FastAPI, Uvicorn, rembg/U²-Net, ONNX Runtime, Pillow, NumPy, scikit-image, Requests, HTML, CSS, and vanilla JavaScript. The frontend is served by the same FastAPI service; no separate frontend service or Node build is needed.

## Installation on Windows

In PowerShell, open the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script activation is blocked, install and run with the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Running Locally

```powershell
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The API documentation is at <http://127.0.0.1:8000/docs>, and health status is at <http://127.0.0.1:8000/health>.

## API Endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /` | Serves the web application. |
| `GET /health` | Returns application status and model name. |
| `POST /process-samples` | Processes the three built-in scikit-image sample images. |
| `POST /process-upload` | Accepts one to three multipart files under the repeated field name `images`. |
| `POST /process-urls` | Accepts JSON with exactly three URLs: `{"urls":["https://…/1.jpg","https://…/2.jpg","https://…/3.jpg"]}`. |

Processing endpoints return a `results` array. Each result contains a label and base64 PNG fields named `original`, `mask`, and `foreground`. Uploads are validated from their image contents, limited to 10 MB each, and capped at three. URL requests require public HTTP/HTTPS addresses, use timeouts, are size limited, and validate downloaded image contents. Redirects are not followed.

## Deploy to Render

This project deploys as **one Render Web Service**; FastAPI serves both the frontend and the API.

1. Create a GitHub repository for this project.
2. Push `main.py`, `requirements.txt`, `README.md`, `.python-version`, `.gitignore`, and the `static/` folder to GitHub. The original `enhanced (1).ipynb` can remain in the repository as the notebook source.
3. Sign in to Render and choose **New → Web Service**.
4. Connect GitHub and select this repository.
5. Select the branch you pushed.
6. Set **Build Command** to `pip install -r requirements.txt`.
7. Set **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
8. Use Python 3.11; `.python-version` requests 3.11.11.
9. Create the service and wait for deployment to finish.
10. Open the generated Render URL and test all three choices.
11. Check the generated URL with `/health` and `/docs` appended.

U²-Net is a relatively large pre-trained model. Initial startup can take longer while weights are downloaded and loaded. Inference can also be slow on CPU-only hosting. Some image hosts block automated downloads; use direct image URLs. Hosting plans may sleep and have cold starts depending on the plan.

## Limitations

The pipeline targets the salient foreground, not multiple labeled object classes. Image size and CPU resources affect processing time. The server limits each image to 10 MB and 25 megapixels; external hosts may block downloads.

## Future Scope

Possible future improvements include a downloadable results bundle, background jobs for large batches, and deployment monitoring. These are not part of the current application.
