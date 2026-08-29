# FloodGuard 🌊

An AI-powered web application for flood-region detection using a U-Net-based image segmentation model.

## Overview

FloodGuard is an academic project that explores computer vision and deep learning for identifying flooded regions in images.

The project combines a custom U-Net-based segmentation model with a Flask web application. Users can upload an image through the web interface, after which the backend preprocesses the image and generates a pixel-level flood segmentation mask. The predicted mask is then overlaid on the original image for visualization.

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
- U-Net-based segmentation
- Attention gates

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

## Model Architecture

The project uses a U-Net-based encoder-decoder architecture for image segmentation. The encoder progressively extracts spatial features while reducing the image dimensions, and the decoder reconstructs the segmentation output using skip connections.

The architecture includes attention gates to help the model focus on relevant spatial features during decoding.

The main stages are:

```text
Input Image (256 × 256 × 3)
        ↓
Encoder
64 → 128 → 256 → 512 filters
        ↓
Bottleneck
1024 filters
        ↓
Decoder
512 → 256 → 128 → 64 filters
        ↓
Sigmoid Output
        ↓
Flood Segmentation Mask
