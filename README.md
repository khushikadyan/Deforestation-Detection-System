# 🌳 Deforestation Detection using Satellite Imagery

This project aims to **detect deforestation from satellite images** using a **U-Net deep learning model**. It leverages automated satellite data collection, image segmentation techniques, and Earth Engine + TensorFlow pipelines to help environmental organizations monitor illegal deforestation in near real-time.

<p align="center">
  <img src="Final Output.png" width="70%"/>
</p>

---

## 📌 Project Objective

> **Can an automated model built on satellite imagery help environmental organizations monitor illegal deforestation in near real-time?**

We aim to answer this by building a robust deep learning pipeline that:
- Collects satellite imagery for selected regions and years.
- Trains a U-Net segmentation model on forest/non-forest (or deforested) labels.
- Allows users to input coordinates + year and visualize predictions on a Leaflet map.

---

## 🛰️ Data Collection & Preprocessing

We use **Google Earth Engine (GEE)** to fetch Sentinel-2 or Landsat-8 imagery and NDVI indices.

### Data Sources:
- **Imagery**: Landsat-8 (30m resolution), optionally Sentinel-2 (10m resolution).
- **Labels**: Derived using NDVI and/or forest loss layers from Global Forest Change dataset.

### Preprocessing:
- Region-wise time-lapse collection.
- NDVI-based labeling: Thresholding NDVI to identify vegetation loss.
- Tiling large images into patches.
- Saving as `.npy` and `.tfrecord` formats for model training.

---

## 🧠 Model Architecture

The model uses a **U-Net architecture** implemented in **TensorFlow**, designed for image segmentation tasks.

- **Input**: Satellite image patches (RGB or NDVI composite).
- **Output**: Binary mask indicating deforested areas.

### Features:
- Batch normalization and dropout regularization
- Dice coefficient + binary cross-entropy loss
- Data augmentation (rotation, flipping, contrast)

---

## ⚙️ Training

```bash
python train.py --epochs 50 --batch_size 16 --data_dir ./data --save_model ./models/unet_deforestation.h5

```

## 📈 Evaluation

The trained model is evaluated on a held-out test set using the following metrics:

- **IoU (Intersection over Union)**: Measures the overlap between predicted deforested area and ground truth.
- **Dice Coefficient**: Especially useful for imbalanced classes.
- **Precision & Recall**: To understand false positives and false negatives.
- **F1 Score**: Harmonic mean of precision and recall.


---

## 🌍 Deployment

### Hugging Face Space 🚀  
Visit: [**Deforestation Detection on Hugging Face**](https://huggingface.co/spaces/ojasrohatgi/Deforestation-Detection)

Features:
- Interactive Leaflet map to select coordinates
- Year input to analyze temporal deforestation
- On-click image fetching and prediction overlay
- Downloadable prediction masks for offline use

### Backend Logic
The backend takes user inputs (`lat, lon, year`), queries GEE for imagery, preprocesses it into patches, passes it to the trained U-Net model, and returns a combined prediction image with overlays.

---

## 🧪 How to Run Locally

### 1. Clone the repository

```bash
git clone [https://github.com/Ojas-Rohatgi/Deforestation-Detection](https://github.com/Ojas-Rohatgi/Deforestation-Detection)
cd Deforestation-Detection

```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Prepare data

Ensure you have satellite imagery and labeled NDVI masks stored in the `data/` folder.

### 4. Train the model

```bash
python training.py --epochs 50 --batch\_size 16 --data\_dir ./data --save\_model ./model/unet\_deforestation.h5

```
You can configure the number of epochs, batch size, and save path as needed.

### 5. Run the app

```bash
cd app
python app.py

```
This will launch a web interface where you can:

* Select a region using the interactive Leaflet map
* Input a year (e.g., 2020)
* Fetch satellite imagery and get prediction overlays

---

## ☁️ Deployment

This app is live at:
🔗 [Hugging Face Spaces – ojasrohatgi/Deforestation-Detection](https://huggingface.co/spaces/ojasrohatgi/Deforestation-Detection)

### Features:

* 🌍 Interactive map for selecting location
* 📅 Year input for time-based analysis
* 📷 Automatic image fetch via GEE
* 🧠 Real-time inference with trained U-Net
* 🖼️ Visual overlay of predictions
* ⬇️ Option to download the prediction mask

---

## ⚙️ Tech Stack

| Component        | Tools Used                             |
| ---------------- | -------------------------------------- |
| Model            | TensorFlow 2.x, Keras, U-Net           |
| Data Source      | Google Earth Engine (Landsat-8, NDVI)  |
| Frontend         | Flask, Leaflet.js                      |
| Image Processing | NumPy, OpenCV, Matplotlib, PIL         |
| Deployment       | Hugging Face Spaces, GitHub, Docker    |
| Output Formats   | PNG masks, NumPy arrays, TFRecords     |

---

## 🚀 Future Roadmap

* [ ] Multi-year change detection (e.g., forest loss between 2015–2023)
* [ ] Visual heatmaps of deforestation severity
* [ ] Region-specific fine-tuning for Southeast Asia, Amazon, and Africa
* [ ] Support for multispectral and SAR imagery
* [ ] Webhooks/API integration for automated NGO alerts

---


## ⭐️ Support This Project

If you found this useful:

* 🌟 Star this repository on GitHub
* 🚀 Share the Hugging Face app with your network
* 📢 Report issues or suggest improvements in the Issues tab
