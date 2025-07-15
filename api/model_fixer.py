# model_fixer.py - Fix your actual model files
import os
import shutil
import zipfile
import tensorflow as tf
from pathlib import Path
import numpy as np

def fix_keras_files():
    """Fix the Keras .keras files that can't be loaded"""
    print("🔧 Attempting to fix Keras model files...")
    
    keras_files = [
        "best_model.keras",
        "enhanced_dog_emotion_20250525-134150/best_model.keras"
    ]
    
    for keras_file in keras_files:
        if os.path.exists(keras_file):
            print(f"\n🎯 Examining: {keras_file}")
            
            # Check if it's a valid zip file
            try:
                with zipfile.ZipFile(keras_file, 'r') as zip_ref:
                    print(f"  ✅ Valid zip file with {len(zip_ref.namelist())} entries")
                    # List some contents
                    for i, name in enumerate(zip_ref.namelist()[:5]):
                        print(f"    - {name}")
                    if len(zip_ref.namelist()) > 5:
                        print(f"    ... and {len(zip_ref.namelist()) - 5} more")
            except zipfile.BadZipFile:
                print(f"  ❌ Not a valid zip file")
                continue
            except Exception as e:
                print(f"  ❌ Zip check failed: {e}")
                continue
            
            # Try different loading approaches
            loading_methods = [
                ("Standard load", lambda: tf.keras.models.load_model(keras_file)),
                ("No compile", lambda: tf.keras.models.load_model(keras_file, compile=False)),
                ("Custom objects", lambda: tf.keras.models.load_model(keras_file, custom_objects={})),
                ("No compile + custom", lambda: tf.keras.models.load_model(keras_file, compile=False, custom_objects={})),
            ]
            
            for method_name, method in loading_methods:
                try:
                    print(f"  🧪 Trying: {method_name}")
                    model = method()
                    print(f"  ✅ SUCCESS with {method_name}!")
                    print(f"    Input shapes: {[inp.shape for inp in model.inputs]}")
                    print(f"    Output shapes: {[out.shape for out in model.outputs]}")
                    
                    # Save a fixed version
                    fixed_path = f"fixed_{Path(keras_file).name}"
                    model.save(fixed_path)
                    print(f"    💾 Saved fixed version as: {fixed_path}")
                    return fixed_path
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    continue
    
    return None

def convert_savedmodel_to_keras():
    """Try to convert the SavedModel to Keras format to avoid compatibility issues"""
    print("\n🔄 Attempting SavedModel to Keras conversion...")
    
    saved_model_path = "saved_model"
    if not os.path.exists(saved_model_path):
        print("  ❌ No saved_model directory found")
        return None
    
    try:
        # Try to load SavedModel using tf.keras
        print("  🧪 Trying to load SavedModel as Keras model...")
        model = tf.keras.models.load_model(saved_model_path)
        print("  ✅ Successfully loaded SavedModel as Keras!")
        
        # Save as new Keras format
        keras_output_path = "converted_from_savedmodel.keras"
        model.save(keras_output_path)
        print(f"  💾 Saved as Keras format: {keras_output_path}")
        
        print(f"    Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"    Output shapes: {[out.shape for out in model.outputs]}")
        
        return keras_output_path
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return None

def downgrade_tensorflow_suggestion():
    """Suggest TensorFlow downgrade for SavedModel compatibility"""
    print("\n💡 TensorFlow Version Compatibility Fix:")
    print("Your SavedModel was likely created with an older TensorFlow version.")
    print("The '_UserObject' error is common with TF 2.19.0 loading older models.")
    print("\nSuggested solutions:")
    print("1. Downgrade TensorFlow:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow==2.12.0")
    print("2. Or try TensorFlow 2.15.0:")
    print("   pip install tensorflow==2.15.0")
    print("3. Re-export your model with current TensorFlow version")

def create_working_api_with_fixed_model(model_path):
    """Create a simple API that uses the fixed model"""
    if not model_path:
        print("❌ No working model found to create API")
        return
    
    api_code = f'''# working_api_with_real_model.py - Uses your actual fixed model
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from datetime import datetime
from contextlib import asynccontextmanager

# Load the fixed model
MODEL_PATH = "{model_path}"
model = None
CLASS_NAMES = [
    "Aggressive/Threatening", 
    "Curiosity/Alertness", 
    "Fearful/Anxious",
    "Happy/Playful", 
    "Relaxed", 
    "Stressed", 
    "Submissive/Appeasement"
]

class ImageRequest(BaseModel):
    image: str
    filename: str = "image.jpg"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"🚀 Loading model: {{MODEL_PATH}}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded successfully!")
        print(f"Input shapes: {{[inp.shape for inp in model.inputs]}}")
        print(f"Output shapes: {{[out.shape for out in model.outputs]}}")
    except Exception as e:
        print(f"❌ Failed to load model: {{e}}")
        raise
    yield

app = FastAPI(title="Pawnder Real Model API", version="1.0.0", lifespan=lifespan)

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_with_real_model(img_array):
    """Try different prediction approaches with the real model"""
    
    # Method 1: Named inputs (if model expects them)
    try:
        behavior_input = np.zeros((1, 64))  # Adjust size as needed
        predictions = model.predict({{
            'image_input': img_array, 
            'behavior_input': behavior_input
        }}, verbose=0)
        if len(predictions) >= 2:
            return predictions[0][0], predictions[1][0][0]
    except:
        pass
    
    # Method 2: Positional inputs
    try:
        for behavior_size in [64, 46, 32]:
            behavior_input = np.zeros((1, behavior_size))
            predictions = model.predict([img_array, behavior_input], verbose=0)
            if len(predictions) >= 2:
                return predictions[0][0], predictions[1][0][0]
    except:
        pass
    
    # Method 3: Image only
    try:
        predictions = model.predict(img_array, verbose=0)
        if isinstance(predictions, list) and len(predictions) >= 2:
            return predictions[0][0], predictions[1][0][0]
        else:
            # Single output - assume emotion probabilities
            emotion_probs = predictions[0] if len(predictions.shape) > 1 else predictions
            return emotion_probs, 0.8  # Default confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"All prediction methods failed: {{e}}")

@app.get("/")
async def root():
    return {{
        "message": "Pawnder Real Model API",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }}

@app.post("/predict-json")
async def predict_emotion(request: ImageRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        image_data = base64.b64decode(request.image)
        img_array = preprocess_image(image_data)
        
        emotion_probs, confidence_score = predict_with_real_model(img_array)
        
        predicted_idx = np.argmax(emotion_probs)
        predicted_emotion = CLASS_NAMES[predicted_idx]
        emotion_score = float(emotion_probs[predicted_idx])
        confidence_score = float(confidence_score)
        
        all_emotions = {{
            CLASS_NAMES[i]: float(emotion_probs[i]) 
            for i in range(len(CLASS_NAMES))
        }}
        
        return {{
            "success": True,
            "emotion": predicted_emotion,
            "confidence": confidence_score,
            "score": emotion_score,
            "filename": request.filename,
            "all_emotions": all_emotions,
            "model_path": MODEL_PATH,
            "timestamp": datetime.now().isoformat()
        }}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {{str(e)}}")

@app.post("/predict")
async def predict_emotion_file(file: UploadFile = File(...)):
    content = await file.read()
    base64_image = base64.b64encode(content).decode('utf-8')
    request = ImageRequest(image=base64_image, filename=file.filename)
    return await predict_emotion(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''
    
    with open("working_api_with_real_model.py", "w") as f:
        f.write(api_code)
    
    print(f"\n✅ Created working_api_with_real_model.py")
    print(f"This API will use your real model: {model_path}")
    print("Run it with: python working_api_with_real_model.py")

def main():
    print("🚀 Pawnder Model Fixer")
    print("=" * 50)
    
    # Try to fix Keras files
    fixed_keras = fix_keras_files()
    
    # Try to convert SavedModel
    converted_keras = convert_savedmodel_to_keras()
    
    # Use whichever worked
    working_model = fixed_keras or converted_keras
    
    if working_model:
        print(f"\n🎉 SUCCESS! Working model: {working_model}")
        create_working_api_with_fixed_model(working_model)
    else:
        print("\n❌ Could not fix any models")
        downgrade_tensorflow_suggestion()
        print("\nAlternatively, you could:")
        print("1. Re-train your model with current TensorFlow 2.19.0")
        print("2. Use the export script to create a new compatible SavedModel")

if __name__ == "__main__":
    main()
