# 🛣️ Automated Road Damage Detection using YOLO11

![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-blue?logo=ultralytics)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![ONNX](https://img.shields.io/badge/ONNX-Exported-lightgrey?logo=onnx)
![Colab](https://img.shields.io/badge/Google_Colab-Tesla_T4-F9AB00?logo=googlecolab)

## 📌 Overview
This project applies deep learning to automate road infrastructure inspection. Using the state-of-the-art **YOLO11s** object detection architecture, the model identifies and classifies five types of road damage in real-time. 

The model was trained for 50 epochs on a focused, domain-specific dataset (United States roads) to eliminate visual noise and establish highly accurate decision boundaries. The final model is exported to **ONNX** format for cross-platform edge deployment.

## 🗂️ Dataset: RDD2022 (US-Filtered)
The model was trained using a targeted subset of the **Road Damage Dataset 2022 (RDD2022)**. 
- **Classes Detected (5):** 
  - `0`: Longitudinal Crack
  - `1`: Transverse Crack
  - `2`: Alligator Crack
  - `3`: Other Corruption
  - `4`: Pothole
- **Domain Strategy:** The training data is filtered exclusively to **United States roads**. This eliminates "domain noise" (different asphalt types, lighting, and camera setups globally) and allows the model to learn much tighter, domain-specific visual features.

## 🚀 Model Performance

The model achieved strong localization and classification metrics on the validation set at Epoch 49.

| Metric | Score |
| :--- | :---: |
| **mAP@50** | **0.5489** |
| **mAP@50-95** | **0.3054** |
| **Precision** | **0.6085** |
| **Recall** | **0.5170** |

## ⚙️ Training Configuration
- **Architecture:** YOLO11s (Pretrained on ImageNet)
- **Epochs:** 50 (Early stopping patience = 10)
- **Image Size:** 640x640
- **Batch Size:** 16
- **Hardware:** Google Colab (Tesla T4 GPU)
- **Augmentations:** Mosaic (disabled last 10 epochs), HSV jitter, random scale, flip, and random erasing.

## 💻 Installation & Usage

### 1. Install Requirements
```bash
pip install ultralytics opencv-python
```

### 2. Inference using PyTorch (`.pt`)
```python
from ultralytics import YOLO

# Load the model
model = YOLO('road_damage_yolo11_saved_model/train/weights/best.pt')

# Run inference on an image
results = model.predict(source='sample_image.jpg', conf=0.30, imgsz=640, save=True)
```

### 3. Inference using ONNX (`.onnx`) - *Recommended for Deployment*
The repository includes an ONNX exported model (`best.onnx`, ~36.5 MB) which is ideal for deployment on Edge devices (Jetson Nano, Raspberry Pi) or via OpenCV DNN.
```python
from ultralytics import YOLO

# Load ONNX model
onnx_model = YOLO('road_damage_yolo11_saved_model/train/weights/best.onnx')

# Run prediction
results = onnx_model.predict(source='sample_image.jpg', conf=0.30, imgsz=640)
```

## 📂 Repository Structure
```text
📦 Road-Damage-YOLO11
 ┣ 📂 road_damage_yolo11_saved_model/  # YOLO11s Model (50 Epochs, US-Only)
 ┃ ┣ 📂 train/weights/                 # best.pt, last.pt, best.onnx
 ┃ ┣ 📂 sample_predictions/            # Visual demo detections
 ┃ ┗ 📜 results.csv                    # Epoch-by-epoch training logs
 ┣ 📜 Road_Damage_YOLO11_Presentation.pptx # Full Pitch Deck
 ┣ 📜 Presentation_Script.md           # English Speaker Script
 ┣ 📜 Presentation_Script_Arabic.md    # Arabic Speaker Script
 ┗ 📜 README.md
```

## 🔮 Future Work
- **Global Fine-tuning:** Train on the full multi-country dataset for 100+ epochs to generalize the domain.
- **Instance Segmentation:** Move from bounding boxes to polygon segmentation for exact crack area calculation.
- **GPS Mapping:** Integrate predictions with GPS metadata for live city infrastructure dashboards.
- **Edge Deployment:** Deploy the `.onnx` model on a moving vehicle via a Jetson Nano.

---
*Developed with PyTorch & Ultralytics YOLO11.*
