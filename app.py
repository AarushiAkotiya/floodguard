import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load the trained U-Net model
MODEL_PATH = 'flood_segmentation_unet.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the image size used during training
IMAGE_SIZE = (256, 256)

@app.route('/')
def home():
    """
    This route renders the main HTML page (your frontend).
    The file 'index.html' must be in a 'templates' folder.
    """
    return render_template('index.html')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    This is the API endpoint to receive an image, process it with the U-Net model,
    and return the flood detection mask.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_np = np.array(image)

        # Preprocess the image for the model
        input_image = cv2.resize(image_np, IMAGE_SIZE)
        input_image = input_image / 255.0  # Normalize pixel values
        input_image = np.expand_dims(input_image, axis=0) # Add batch dimension

        if model is None:
            return jsonify({'error': 'AI model is not available'}), 500

        # Make a prediction using the trained model
        prediction = model.predict(input_image)
        mask = (prediction[0] > 0.5).astype(np.uint8) * 255 # Threshold the prediction

        # Overlay the mask on the original image for visualization
        original_resized = cv2.resize(image_np, IMAGE_SIZE)
        overlay = cv2.merge([mask, mask, np.zeros_like(mask)]) # Create a red mask
        blended_image = cv2.addWeighted(original_resized, 0.7, overlay, 0.3, 0)

        # Convert the processed image to a format that can be sent back
        img_pil = Image.fromarray(blended_image)
        byte_arr = io.BytesIO()
        img_pil.save(byte_arr, format='PNG')
        byte_arr.seek(0)

        return send_file(byte_arr, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
