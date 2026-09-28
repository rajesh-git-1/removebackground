# Salient Object Detection with U²-Net

Streamlit app for salient foreground extraction with the pre-trained `rembg` U²-Net models. For each image it displays the original, a binary mask using the notebook's strict `alpha > 128` threshold, and the foreground composited on white.

## Features

- Process the built-in astronaut, cat, and coffee images.
- Upload up to three images (10 MB and 25 megapixels per image; 30 MB total).
- Process three direct, publicly reachable HTTP(S) image URLs.
- Reuse one model session per running app process. The model is downloaded at first inference and cached by `rembg` on the host.

The deployment default is `u2netp`, the smaller and faster model, to reduce cold-start downloads and memory use on free hosting. To use the larger original model on Streamlit Community Cloud, add `SOD_MODEL = "u2net"` in the app's **Advanced settings → Secrets**. Model files are not committed to this repository. Hosts with ephemeral storage may download the weights again after a restart or redeploy.

## Run locally

Use Python 3.11 (the version configured in `.python-version`). On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

Open the local URL printed by Streamlit, normally <http://localhost:8501>.

## Deploy to Streamlit Community Cloud

1. Push the project to a GitHub repository. Include `main.py`, `requirements.txt`, `.python-version`, and this README. The `static/` folder and notebook are not required by the Streamlit app.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize access to the repository.
3. Choose **Create app**, select the repository and branch, and set the app file path to `main.py`.
4. Open **Advanced settings** and keep Python at 3.11 to match `.python-version`.
5. Deploy. The first inference downloads and initializes the model; allow extra time for this cold start.
6. Try the built-in samples first, then an upload and direct image URLs. Confirm that all three output images appear.
7. If the app exceeds available memory or starts too slowly, use the default `u2netp`, keep batches small, and resize very large source images before uploading. If you set `SOD_MODEL` to `u2net` in Secrets, remove that setting and reboot the app to restore `u2netp`.

### Hosting and resource considerations

- `u2netp` materially reduces model size and CPU/memory load but can produce different segmentation quality from full `u2net`. Set `SOD_MODEL=u2net` only if the hosting plan has enough memory and startup time.
- The model is fetched at runtime by `rembg`, not stored in Git. A cold start requires network access to the model host. Ephemeral host storage means a restart can trigger a new download.
- Inference uses CPU on the standard Streamlit Community Cloud setup. Processing time depends on image size and current host load. The app handles one submission at a time and limits each batch to three images.
- Community Cloud quotas, sleep behavior, supported Python versions, and resource availability can change. Check the current [Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud) if deployment settings or limits differ from these steps.
- URL processing accepts only public HTTP/HTTPS hosts, does not follow redirects, and limits each download to 10 MB. Some image providers block server downloads; use a direct image address or upload the image.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit UI, input validation, model loading, and segmentation pipeline. |
| `requirements.txt` | Runtime dependencies, including CPU ONNX Runtime. |
| `.python-version` | Requests Python 3.11.11 for deployment. |
| `enhanced (1).ipynb` | Original notebook workflow. |

## Troubleshooting

- **Build fails while installing dependencies:** check the deployment build logs. If Python 3.11 is unavailable on the selected host, select a supported Python version and confirm the dependency versions in `requirements.txt` support it.
- **Build log reports a timeout fetching `pypi.org/simple/jsonschema/`:** this is a package-index network timeout. `rembg` and `jsonschema` are pinned to avoid pip's long version backtracking; push the updated requirements and restart/reboot the app. If the log still shows a timeout, retry the deploy after PyPI is reachable from Community Cloud.
- **First run appears stuck:** model download and initialization happen at first inference. Wait for the spinner and inspect app logs for network or memory errors.
- **App restarts during inference:** use `u2netp`, upload fewer/smaller images, and avoid the full `u2net` model on a low-memory plan.
- **A URL is rejected:** provide a direct public image URL with an `image/*` response. Private, local-network, redirected, or non-image URLs are intentionally blocked.
