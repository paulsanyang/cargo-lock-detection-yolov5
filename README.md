# Aircraft Cargo-Lock Detection with YOLOv5s

A computer vision project that fine-tunes **YOLOv5s** to detect aircraft cargo locks in images. The notebook covers Roboflow data preparation, model training, held-out evaluation, annotated prediction galleries, false-positive inspection, and exploratory Grad-CAM.

## Quick start

1. Create a Python environment with PyTorch installed (GPU recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. Provide a `ROBOFLOW_API_KEY` via environment variable or Colab secret.
4. Open `notebooks/cargo_lock_detection.ipynb` and execute the cells in order. Alternatively, load an existing `best.pt` checkpoint and skip training.

## Repository layout

- `cargo_lock_detection.ipynb` — documented, cleaned research notebook.
- `original_execution_logs.txt` — text logs extracted from the original notebook, kept separately to reduce notebook size.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes API keys, downloaded datasets, weights and generated outputs.

The notebook saves generated artifacts under `outputs/`, including metrics, prediction images, a random sample gallery, and false-positive examples. These outputs are excluded from Git by default; publish only selected, license-compliant figures if desired.

## Dataset and model

Dataset: [AC Locks on Roboflow Universe, version 9](https://universe.roboflow.com/ac-locks/ac-locks/dataset/9). Original training configuration: YOLOv5s initialized from `yolov5s.pt`, 300 epochs. Other training hyperparameters were unspecified in the original notebook. The cleaned notebook uses a fixed random seed for the 80/20 train/validation split and retains the test split for evaluation.

## Notes on interpretability

Grad-CAM is experimental: the notebook scores the strongest pre-NMS YOLOv5 proposal using objectness × class probability. It does not explain a specific post-NMS detection, and the visualization uses a simple resize rather than letterbox preprocessing. See the notebook for caveats.

## Reproducibility

No new training run or runtime validation was performed during the notebook cleanup. Report numerical metrics only after executing the evaluation section with a valid checkpoint. Verify dataset license and attribution before distributing images.
