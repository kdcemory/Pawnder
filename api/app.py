
import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model and metadata
MODEL_PATH = "/app/saved_model"
METADATA_PATH = "/app/model_metadata.json"

print("Loading model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

CLASS_NAMES = metadata["model_info"]["class_names"]
IMAGE_SIZE = metadata["preprocessing"]["image_size"][:2]

print(f"Model loaded successfully. Classes: {CLASS_NAMES}")

def preprocess_image(image_base64):
    """Preprocess base64 image for model input"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Preprocess image
        image_array = preprocess_image(data['image'])
        
        # Create behavior input (zeros for now)
        behavior_size = metadata["preprocessing"]["behavior_input_size"]
        behavior_array = np.zeros((1, behavior_size), dtype=np.float32)
        
        # Get input names from the model signature
        input_names = list(infer.structured_input_signature[1].keys())
        input_shapes = infer.structured_input_signature[1]
        
        # Prepare inputs - match by name and verify shapes
        inputs = {}
        for name in input_names:
            if 'image' in name.lower():
                inputs[name] = tf.constant(image_array)
            elif 'behavior' in name.lower():
                # Make sure behavior input matches expected size
                expected_shape = input_shapes[name].shape.as_list()
                if len(expected_shape) > 1:
                    expected_size = expected_shape[1]
                    if expected_size != behavior_size:
                        # Update behavior array to match model expectation
                        behavior_array = np.zeros((1, expected_size), dtype=np.float32)
                        logger.warning(f"Adjusted behavior input size to {expected_size}")
                inputs[name] = tf.constant(behavior_array)
            else:
                # Fallback based on position and shape
                expected_shape = input_shapes[name].shape.as_list()
                if len(expected_shape) == 4:  # Image-like
                    inputs[name] = tf.constant(image_array)
                else:  # Feature vector
                    feature_size = expected_shape[1] if len(expected_shape) > 1 else expected_shape[0]
                    feature_array = np.zeros((1, feature_size), dtype=np.float32)
                    inputs[name] = tf.constant(feature_array)
        
        # Run inference
        outputs = infer(**inputs)
        
        # Process outputs - find emotion and confidence outputs
        emotion_probs = None
        confidence = None
        
        for key, value in outputs.items():
            if 'emotion' in key.lower() and value.shape[-1] == len(CLASS_NAMES):
                emotion_probs = value.numpy()[0]
            elif 'confidence' in key.lower() and value.shape[-1] == 1:
                confidence = value.numpy()[0][0]
        
        # Fallback if we can't identify outputs by name
        if emotion_probs is None:
            output_values = list(outputs.values())
            emotion_probs = output_values[0].numpy()[0]
            if len(output_values) > 1:
                confidence = output_values[1].numpy()[0][0]
            else:
                confidence = 0.8  # Default confidence
        
        # Get predicted emotion
        predicted_idx = np.argmax(emotion_probs)
        predicted_emotion = CLASS_NAMES[predicted_idx]
        predicted_score = float(emotion_probs[predicted_idx])
        
        # Create response
        response = {
            "emotion": predicted_emotion,
            "confidence": float(confidence) if confidence is not None else 0.8,
            "emotion_score": predicted_score,
            "all_emotions": {
                emotion: float(prob) for emotion, prob in zip(CLASS_NAMES, emotion_probs)
            },
            "model_version": "1.0"
        }
        
        logger.info(f"Prediction: {predicted_emotion} (confidence: {confidence:.2f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify(metadata)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
