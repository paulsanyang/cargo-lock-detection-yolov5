# Aircraft Cargo-Lock Detection with YOLOv5s

**Fifteenth Montreal Industrial Problem Solving Workshop**  
**Centre de recherches mathématiques (CRM), Montréal, Canada**

## Overview

This project was undertaken during the Fifteenth Montreal Industrial Problem Solving Workshop at the Centre de recherches mathématiques (CRM) in Montréal, Canada.

The project explores the application of deep learning to aircraft cargo-lock detection using **YOLOv5s**. My contributions included fine-tuning the object detection model, evaluating its performance, analysing predictions and false positives, and implementing exploratory Grad-CAM visualizations.

## Methodology

- **Dataset:** [AC Locks (Version 9)](https://universe.roboflow.com/ac-locks/ac-locks/dataset/9), hosted on Roboflow Universe.
- **Model:** YOLOv5s initialized with pretrained `yolov5s.pt` weights and trained for 300 epochs.
- **Evaluation:** Held-out evaluation, annotated prediction galleries, and false-positive inspection.
- **Interpretability:** Exploratory Grad-CAM analysis of image regions influencing model predictions.

## Repository Structure

```text
cargo-lock-detection-yolov5/
├── cargo_lock_detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

1. Clone the repository and install the required dependencies:

   `pip install -r requirements.txt`

2. Configure your `ROBOFLOW_API_KEY` using an environment variable or Colab secret.

3. Open `cargo_lock_detection.ipynb` and run the notebook cells in order. A GPU-enabled environment is recommended.

%Alternatively, an existing `best.pt` checkpoint can be used to skip training.

## Technologies

Python · PyTorch · YOLOv5s · Ultralytics · Roboflow · OpenCV · Grad-CAM · Jupyter Notebook

## Acknowledgements

This project was undertaken as part of the **Fifteenth Montreal Industrial Problem Solving Workshop**, associated with the **Centre de recherches mathématiques (CRM), Montréal, Canada**.

The dataset is available through [Roboflow Universe](https://universe.roboflow.com/ac-locks/ac-locks/dataset/9). Dataset licensing and attribution requirements should be checked before redistributing images.

**Note:** Grad-CAM analysis is exploratory and does not explain individual post-NMS detections. Model weights, downloaded datasets, and generated outputs are excluded from the repository.
