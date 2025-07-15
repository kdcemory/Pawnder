# export_model_for_cloud.py
# Run this in VS Code to prepare your model for cloud deployment

import os
import sys
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Configure GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"❌ GPU configuration error: {e}")
else:
    print("⚠️ No GPU found, using CPU")

class ModelExporter:
    def __init__(self, local_project_path):
        self.project_path = Path(local_project_path)
        self.models_dir = self.project_path / "Models"
        self.export_dir = self.project_path / "cloud_export"
        
    def find_best_model(self):
        """Find the best trained model"""
        model_files = []
        
        # Search for .keras model files first (newer format)
        for model_file in self.models_dir.rglob("*.keras"):
            if "best_model" in model_file.name or "final" in model_file.name:
                model_files.append(model_file)
        
        # If no .keras files with best/final, search for any .keras files
        if not model_files:
            model_files = list(self.models_dir.rglob("*.keras"))
        
        # Fallback to .h5 files if no .keras files found
        if not model_files:
            for model_file in self.models_dir.rglob("*.h5"):
                if "best_model" in model_file.name or "final" in model_file.name:
                    model_files.append(model_file)
        
        if not model_files:
            # Final fallback to any .h5 file
            model_files = list(self.models_dir.rglob("*.h5"))
        
        if not model_files:
            raise FileNotFoundError("No model files (.keras or .h5) found!")
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Found {len(model_files)} model file(s)")
        for i, model_file in enumerate(model_files[:3]):  # Show top 3
            print(f"  {i+1}. {model_file} (modified: {model_file.stat().st_mtime})")
        
        return model_files[0]
    
    def export_for_cloud_run(self, model_path):
        """Export model for Cloud Run deployment"""
        print(f"Loading model from: {model_path}")
        print(f"Model format: {model_path.suffix}")
        
        try:
            # Load the model (works for both .keras and .h5 formats)
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully")
            print(f"Model inputs: {[input.shape for input in model.inputs]}")
            print(f"Model outputs: {[output.shape for output in model.outputs]}")
            
            # Print model summary for verification
            print(f"\nModel Summary:")
            model.summary()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"Trying alternative loading method...")
            
            try:
                # Alternative loading method
                import tensorflow as tf
                model = tf.saved_model.load(str(model_path))
                print(f"✅ Model loaded using SavedModel format")
            except Exception as e2:
                print(f"❌ Failed to load model with both methods:")
                print(f"   Method 1 error: {e}")
                print(f"   Method 2 error: {e2}")
                raise Exception(f"Could not load model: {model_path}")
        
        # Create export directory
        saved_model_path = self.export_dir / "saved_model"
        saved_model_path.mkdir(parents=True, exist_ok=True)
        
        # Export as SavedModel
        print(f"\nExporting model to SavedModel format...")
        tf.saved_model.save(model, str(saved_model_path))
        print(f"✅ SavedModel exported to: {saved_model_path}")
        
        # Create metadata file with more detailed information
        try:
            metadata = {
                "model_info": {
                    "original_file": str(model_path),
                    "file_format": model_path.suffix,
                    "input_shapes": [input.shape.as_list() for input in model.inputs],
                    "input_names": [input.name for input in model.inputs],
                    "output_shapes": [output.shape.as_list() for output in model.outputs],
                    "output_names": [output.name for output in model.outputs],
                    "class_names": [
                        "Aggressive/Threatening", 
                        "Curiosity/Alertness", 
                        "Fearful/Anxious",
                        "Happy/Playful", 
                        "Relaxed", 
                        "Stressed", 
                        "Submissive/Appeasement"
                    ],
                    "model_type": "enhanced_dog_emotion_recognition"
                },
                "preprocessing": {
                    "image_size": [224, 224, 3],
                    "normalization": "0-1 range",
                    "behavior_input_size": 64,
                    "image_preprocessing_steps": [
                        "Convert BGR to RGB",
                        "Resize to 224x224", 
                        "Normalize to [0,1]",
                        "Add batch dimension"
                    ]
                },
                "model_architecture": {
                    "backbone": "Enhanced architecture with behavior inputs",
                    "total_parameters": model.count_params() if hasattr(model, 'count_params') else "Unknown",
                    "trainable_parameters": "Unknown"  # Will be calculated if possible
                },
                "export_info": {
                    "export_date": str(datetime.now()),
                    "tensorflow_version": tf.__version__,
                    "export_format": "SavedModel"
                }
            }
        except Exception as e:
            print(f"⚠️ Warning: Could not extract full metadata: {e}")
            # Fallback metadata
            metadata = {
                "model_info": {
                    "original_file": str(model_path),
                    "file_format": model_path.suffix,
                    "class_names": [
                        "Aggressive/Threatening", 
                        "Curiosity/Alertness", 
                        "Fearful/Anxious",
                        "Happy/Playful", 
                        "Relaxed", 
                        "Stressed", 
                        "Submissive/Appeasement"
                    ]
                },
                "preprocessing": {
                    "image_size": [224, 224, 3],
                    "normalization": "0-1 range",
                    "behavior_input_size": 64
                }
            }
        
        with open(self.export_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to: {self.export_dir / 'model_metadata.json'}")
        
        return saved_model_path, metadata
    
    def test_exported_model(self, saved_model_path, metadata):
        """Test the exported model with dummy data"""
        print("\n🧪 Testing exported model...")
        
        # Load the exported model
        loaded_model = tf.saved_model.load(str(saved_model_path))
        infer = loaded_model.signatures["serving_default"]
        
        # Create dummy inputs
        image_shape = metadata["model_info"]["input_shapes"][0]
        behavior_shape = metadata["model_info"]["input_shapes"][1]
        
        dummy_image = tf.random.normal([1] + image_shape[1:])
        dummy_behavior = tf.zeros([1] + behavior_shape[1:])
        
        # Get input tensor names (remove :0 suffix if present)
        input_names = [name.split(':')[0] for name in metadata["model_info"]["input_names"]]
        
        # Create input dictionary
        inputs = {
            input_names[0]: dummy_image,
            input_names[1]: dummy_behavior
        }
        
        # Run inference
        try:
            outputs = infer(**inputs)
            print("✅ Model inference successful!")
            print(f"Output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                print(f"  {key}: shape {value.shape}")
            return True
        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            return False
    
    def create_cloud_run_files(self):
        """Create files needed for Cloud Run deployment"""
        
        # Create main application file
        app_py_content = '''
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
IMAGE_SIZE = metadata["preprocessing"]["image_size"][:2]  # [height, width]

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
        
        # Get input names from metadata
        input_names = [name.split(':')[0] for name in metadata["model_info"]["input_names"]]
        
        # Prepare inputs
        inputs = {
            input_names[0]: tf.constant(image_array),
            input_names[1]: tf.constant(behavior_array)
        }
        
        # Run inference
        outputs = infer(**inputs)
        
        # Process outputs
        emotion_probs = outputs['emotion_output'].numpy()[0]
        confidence = outputs['confidence_output'].numpy()[0][0]
        
        # Get predicted emotion
        predicted_idx = np.argmax(emotion_probs)
        predicted_emotion = CLASS_NAMES[predicted_idx]
        predicted_score = float(emotion_probs[predicted_idx])
        
        # Create response
        response = {
            "emotion": predicted_emotion,
            "confidence": float(confidence),
            "emotion_score": predicted_score,
            "all_emotions": {
                emotion: float(prob) for emotion, prob in zip(CLASS_NAMES, emotion_probs)
            },
            "model_version": "1.0",
            "processing_time_ms": 0  # You can add timing if needed
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
'''
        
        # Write app.py
        with open(self.export_dir / "app.py", 'w') as f:
            f.write(app_py_content)
        
        # Create requirements.txt
        requirements = '''
tensorflow==2.12.0
flask==2.3.2
flask-cors==4.0.0
pillow==9.5.0
numpy==1.24.3
gunicorn==20.1.0
'''
        
        with open(self.export_dir / "requirements.txt", 'w') as f:
            f.write(requirements.strip())
        
        # Create Dockerfile
        dockerfile = '''
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY saved_model/ ./saved_model/
COPY model_metadata.json .
COPY app.py .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
'''
        
        with open(self.export_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile.strip())
        
        # Create .dockerignore
        dockerignore = '''
__pycache__
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt
.git
.pytest_cache
.coverage
.tox
Dockerfile
README.md
.DS_Store
.gitignore
'''
        
        with open(self.export_dir / ".dockerignore", 'w') as f:
            f.write(dockerignore.strip())
        
        print(f"✅ Cloud Run deployment files created in: {self.export_dir}")
    
    def export_all(self):
        """Complete export process"""
        print("🚀 Starting model export for cloud deployment...\n")
        
        # Find best model
        model_path = self.find_best_model()
        print(f"Found model: {model_path}")
        
        # Export for cloud
        saved_model_path, metadata = self.export_for_cloud_run(model_path)
        
        # Test exported model
        test_success = self.test_exported_model(saved_model_path, metadata)
        
        if test_success:
            # Create Cloud Run files
            self.create_cloud_run_files()
            
            print(f"\n✅ Export completed successfully!")
            print(f"📁 Export directory: {self.export_dir}")
            print(f"\nNext steps:")
            print(f"1. Review files in {self.export_dir}")
            print(f"2. Test locally: cd {self.export_dir} && python app.py")
            print(f"3. Deploy to Cloud Run using VS Code or gcloud CLI")
            
            return self.export_dir
        else:
            print("❌ Export failed due to model testing issues")
            return None

if __name__ == "__main__":
    # Update this path to your local project
    PROJECT_PATH = r"C:\Users\kelly\Documents\GitHub\Pawnder"
    
    # Option to specify exact model path
    SPECIFIC_MODEL_PATH = r"C:\Users\kelly\Documents\GitHub\Pawnder\Models\enhanced_dog_emotion_20250525-134150\best_model.keras"
    
    exporter = ModelExporter(PROJECT_PATH)
    
    # Check if specific model path exists and use it
    if os.path.exists(SPECIFIC_MODEL_PATH):
        print(f"🎯 Using specified model: {SPECIFIC_MODEL_PATH}")
        model_path = Path(SPECIFIC_MODEL_PATH)
        
        # Export the specific model
        print("🚀 Starting model export for cloud deployment...\n")
        
        try:
            # Export for cloud
            saved_model_path, metadata = exporter.export_for_cloud_run(model_path)
            
            # Test exported model
            test_success = exporter.test_exported_model(saved_model_path, metadata)
            
            if test_success:
                # Create Cloud Run files
                exporter.create_cloud_run_files()
                
                print(f"\n✅ Export completed successfully!")
                print(f"📁 Export directory: {exporter.export_dir}")
                print(f"\nNext steps:")
                print(f"1. Review files in {exporter.export_dir}")
                print(f"2. Test locally: cd {exporter.export_dir} && python app.py")
                print(f"3. Deploy to Cloud Run using VS Code or gcloud CLI")
                
            else:
                print("❌ Export failed due to model testing issues")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print(f"Please check the model path and format.")
    else:
        print(f"❌ Specific model not found: {SPECIFIC_MODEL_PATH}")
        print(f"Searching for models automatically...")
        
        # Fallback to automatic detection
        export_dir = exporter.export_all()
        
        if export_dir:
            print(f"\n🎉 Ready for cloud deployment!")
            print(f"Export location: {export_dir}")
        else:
            print(f"❌ No suitable models found in {PROJECT_PATH}")
            print(f"Please check your Models directory or update the SPECIFIC_MODEL_PATH variable.")
