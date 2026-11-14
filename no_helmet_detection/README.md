# Motorcycle No-Helmet Detection + ANPR

This standalone Python project processes CCTV video, detects helmet/no-helmet head regions, associates a no-helmet rider with a tracked motorcycle, detects its number plate, and reads plate text with OCR. A violation is reported only after repeated frame evidence. It writes an annotated MP4, a CSV event list, and a JSON report.

Association uses person/head overlap, rider-to-motorcycle overlap and foot-point proximity, then motorcycle-to-plate position. These image-space heuristics are intended for fixed-camera footage and may need tuning for a different camera angle.

## Setup (PowerShell)

Run from the directory containing `no_helmet_detection`:

```powershell
cd no_helmet_detection
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first run downloads about 100 MB of ONNX detector weights plus OCR assets, so it needs an internet connection. Inference uses ONNX Runtime on the CPU and does not require PyTorch.

## Run

From the directory containing `no_helmet_detection`:

```powershell
python .\no_helmet_detection\run_detection.py .\wrong_way_detection\backend\data\videos\traffic_short.mp4
```

Replace the sample path with any accessible local video file.

Outputs are written to `no_helmet_detection\outputs\`:

- `<video>_annotated.mp4` — red confirmed or orange pending motorcycle boxes; plate and no-helmet overlays.
- `<video>_violations.csv` — one row per confirmed motorcycle track event.
- `<video>_violations.json` — event data, source paths, and model provenance.

To display the annotated video while processing:

```powershell
python .\no_helmet_detection\run_detection.py .\no_helmet_detection\sample_video.mp4 --show
```

Useful options:

```text
--min-violation-frames 3    Evidence frames needed for a report
--persistence-gap 2         Missing frames tolerated between evidence
--ocr-interval 5            OCR sampling interval while a violation is present
--helmet-model PATH         Use a local helmet model instead of the default
--plate-model PATH          Use a local plate model instead of the default
--vehicle-model PATH        Use a local COCO-compatible ONNX model
```

## Models and limitations

- Helmet/no-helmet: [Nqhia/helmet-detection](https://github.com/Nqhia/helmet-detection), pinned YOLO11 ONNX weights and head-level class labels. The model card identifies the weights as AGPL-3.0.
- Number plate localization: [ml-debi/yolov8-license-plate-detection](https://huggingface.co/ml-debi/yolov8-license-plate-detection), published with an MIT license tag.
- Person/motorcycle boxes: a pinned YOLOv8n COCO ONNX export from [Hugging Face](https://huggingface.co/kshitijjjjjjjjjjjjjjjj/yolov8n-coco-onnx). That model repository does not publish a license or model card; verify its terms before use or provide your own compatible model with `--vehicle-model`.
- OCR: [RapidOCR](https://github.com/RapidAI/RapidOCR), using ONNX Runtime. OCR output is best-effort; blur, small plates, motion, and jurisdiction-specific fonts can produce errors.

The default helmet model was trained on traffic images, but is not calibrated for every motorcycle CCTV camera. The default plate detector is not country-specific. Treat results as review candidates, verify plate text manually, and evaluate/tune on representative footage before operational use. Use footage and plate data only where legally permitted.

The CLI accepts local video files. No sample footage or trained weights are bundled. Pass alternate compatible weights with `--helmet-model` and `--plate-model` when needed.
