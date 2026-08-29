# FloodGuard 🌊

An AI-powered web application for detecting flooded areas in images using a trained U-Net segmentation model.

## Overview

FloodGuard is an academic project developed to explore the use of computer vision and deep learning for flood detection.

The application allows a user to upload an image through a web interface. The image is preprocessed and passed through a trained U-Net model, which generates a segmentation mask for the detected flood regions. The resulting mask is overlaid on the original image and displayed to the user.

## Features

- Upload an image directly through the web interface
- Automatic image preprocessing
- Flood-region segmentation using a trained U-Net model
- Visual overlay of detected regions
- Simple web interface for viewing the original and processed images
- Flask-based backend API for image analysis

## Tech Stack

### Machine Learning
- Python
- TensorFlow
- Keras
- U-Net

### Image Processing
- OpenCV
- Pillow
- NumPy

### Backend
- Flask

### Frontend
- HTML
- JavaScript
- Tailwind CSS

## How It Works

1. The user uploads an image through the FloodGuard web interface.
2. The Flask backend receives the uploaded image.
3. The image is converted to RGB and resized to `256 × 256`.
4. Pixel values are normalized before being passed to the model.
5. The trained U-Net model generates a prediction mask.
6. A threshold is applied to identify the predicted flood regions.
7. The mask is overlaid on the original image.
8. The processed image is returned to the frontend and displayed to the user.

## Project Structure

```text
FloodGuard/
│
├── app.py
├── Flooddetecting.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── templates/
    └── index.html
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/AarushiAkotiya/floodguard.git
cd floodguard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the trained model

Place the trained U-Net model file in the project directory with the filename:

```text
model.h5
```

### 4. Run the application

```bash
python app.py
```

The application will be available at:

```text
http://localhost:5000
```

## Future Improvements

- Improve model performance with larger and more diverse datasets
- Add support for additional satellite imagery sources
- Integrate flood maps and geographical information
- Deploy the application as a cloud-based service
- Add automated flood alerts and monitoring
