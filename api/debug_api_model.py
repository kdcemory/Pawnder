# debug_model_outputs.py - Debug what your model actually returns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from datetime import datetime
from contextlib import asynccontextmanager

# Global variables
model = None
model_type = None
infer_func = None
input_shapes = None

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

def load_model_debug():
    """Load model and show detailed info"""
    global model, model_type, infer_func, input_shapes
    
    print(f"🚀 Loading model with TensorFlow {tf.__version__}")
    
    if os.path.exists("saved_model"):
        print("🎯 Loading SavedModel...")
        try:
            model = tf.saved_model.load("saved_model")
            infer_func = model.signatures["serving_default"]
            input_shapes = infer_func.structured_input_signature[1]
            model_type = "savedmodel"
            
            print("✅ SavedModel loaded successfully!")
            print(f"📊 Input shapes: {[(name, spec.shape) for name, spec in input_shapes.items()]}")
            
            # Debug the output signature
            output_shapes = infer_func.structured_outputs
            print(f"📤 Output shapes: {[(name, spec.shape) for name, spec in output_shapes.items()]}")
            print(f"📋 Output keys: {list(output_shapes.keys())}")
            
            return
            
        except Exception as e:
            print(f"❌ SavedModel failed: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_debug()
    yield

app = FastAPI(title="Pawnder Debug API", version="1.0.0", lifespan=lifespan)

def preprocess_image(image_data):
    """Preprocess image for model input"""
    try:
        image = Image.open(io.BytesIO(image_data))
        print(f"📸 Original image: {image.size}, mode: {image.mode}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"🔄 Converted to RGB")
        
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"📐 Preprocessed shape: {img_array.shape}")
        print(f"📊 Value range: {img_array.min():.3f} to {img_array.max():.3f}")
        
        return img_array
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def debug_model_prediction(img_array):
    """Debug the model prediction to see exactly what it returns"""
    try:
        print("\n🔍 === DEBUGGING MODEL PREDICTION ===")
        
        # Create inputs
        input_names = list(input_shapes.keys())
        inputs = {}
        
        print(f"📋 Expected inputs: {input_names}")
        
        for name in input_names:
            if 'image' in name.lower():
                inputs[name] = tf.constant(img_array.astype(np.float32))
                print(f"📸 Created {name}: shape {img_array.shape}")
            elif 'behavior' in name.lower():
                behavior_shape = input_shapes[name].shape.as_list()
                behavior_size = behavior_shape[1] if len(behavior_shape) > 1 else 46
                behavior_input = np.zeros((1, behavior_size), dtype=np.float32)
                inputs[name] = tf.constant(behavior_input)
                print(f"🧠 Created {name}: shape (1, {behavior_size})")
        
        print(f"🎯 Running inference...")
        
        # Run inference
        outputs = infer_func(**inputs)
        
        print(f"\n📤 === RAW MODEL OUTPUTS ===")
        print(f"Number of outputs: {len(outputs)}")
        
        for key, value in outputs.items():
            tensor_val = value.numpy()
            print(f"\n🔑 Output '{key}':")
            print(f"   Shape: {tensor_val.shape}")
            print(f"   Dtype: {tensor_val.dtype}")
            print(f"   Values: {tensor_val}")
            print(f"   Min: {tensor_val.min()}, Max: {tensor_val.max()}")
            print(f"   Sum: {tensor_val.sum()}")
            
            if len(tensor_val.shape) == 2 and tensor_val.shape[1] == len(CLASS_NAMES):
                print(f"   ✅ This looks like EMOTION PROBABILITIES (7 classes)")
            elif len(tensor_val.shape) == 2 and tensor_val.shape[1] == 1:
                print(f"   ✅ This looks like CONFIDENCE SCORE (single value)")
            else:
                print(f"   ❓ Unknown output format")
        
        # Try to extract emotion probs and confidence
        print(f"\n🎯 === EXTRACTING RESULTS ===")
        
        emotion_probs = None
        confidence = None
        
        for key, value in outputs.items():
            tensor_val = value.numpy()
            
            if tensor_val.shape[-1] == len(CLASS_NAMES):
                emotion_probs = tensor_val[0]
                print(f"✅ Found emotion probabilities in '{key}': {emotion_probs}")
            elif tensor_val.shape[-1] == 1:
                confidence = tensor_val[0][0]
                print(f"✅ Found confidence in '{key}': {confidence}")
        
        # Fallback approach
        if emotion_probs is None:
            print("⚠️ Could not identify emotion probs by shape, using first output")
            output_values = list(outputs.values())
            emotion_probs = output_values[0].numpy()[0]
            print(f"📊 Using first output as emotions: {emotion_probs}")
            
            if len(output_values) > 1:
                second_output = output_values[1].numpy()
                print(f"📊 Second output shape: {second_output.shape}")
                print(f"📊 Second output values: {second_output}")
                
                if second_output.shape[-1] == 1:
                    confidence = second_output[0][0]
                    print(f"✅ Using second output as confidence: {confidence}")
                else:
                    confidence = 0.8
                    print(f"⚠️ Second output not single value, using default confidence: {confidence}")
            else:
                confidence = 0.8
                print(f"⚠️ Only one output, using default confidence: {confidence}")
        
        if confidence is None:
            confidence = 0.8
            print(f"⚠️ No confidence found, using default: {confidence}")
        
        print(f"\n🏁 === FINAL RESULTS ===")
        print(f"Emotion probabilities: {emotion_probs}")
        print(f"Confidence: {confidence}")
        print(f"Emotion probs type: {type(emotion_probs)}")
        print(f"Confidence type: {type(confidence)}")
        
        return emotion_probs, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Pawnder Debug API", 
        "model_loaded": model is not None,
        "model_type": model_type
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": model_type,
        "tensorflow_version": tf.__version__,
    }
    
    if model_type == "savedmodel":
        info.update({
            "signatures": list(model.signatures.keys()),
            "input_shapes": {name: spec.shape.as_list() for name, spec in input_shapes.items()},
            "output_shapes": {name: spec.shape.as_list() for name, spec in infer_func.structured_outputs.items()},
        })
    
    return info

@app.post("/debug-predict")
async def debug_predict(request: ImageRequest):
    """Debug prediction endpoint that shows all model outputs"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        print(f"\n🔍 === DEBUG PREDICTION FOR {request.filename} ===")
        
        # Decode and preprocess image
        image_data = base64.b64decode(request.image)
        img_array = preprocess_image(image_data)
        
        # Debug prediction
        emotion_probs, confidence_score = debug_model_prediction(img_array)
        
        # Get results
        predicted_idx = np.argmax(emotion_probs)
        predicted_emotion = CLASS_NAMES[predicted_idx]
        emotion_score = float(emotion_probs[predicted_idx])
        
        # Make sure confidence is a valid number
        if confidence_score is None:
            confidence_score = 0.8
            print("⚠️ Confidence was None, using default 0.8")
        
        confidence_score = float(confidence_score)
        
        print(f"\n🎉 Final Results:")
        print(f"Predicted emotion: {predicted_emotion}")
        print(f"Emotion score: {emotion_score}")
        print(f"Confidence: {confidence_score}")
        
        # Create all emotions dictionary
        all_emotions = {
            CLASS_NAMES[i]: float(emotion_probs[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        response = {
            "success": True,
            "emotion": predicted_emotion,
            "confidence": confidence_score,
            "score": emotion_score,
            "filename": request.filename,
            "all_emotions": all_emotions,
            "debug_info": {
                "emotion_probs_shape": list(emotion_probs.shape) if hasattr(emotion_probs, 'shape') else str(type(emotion_probs)),
                "confidence_type": str(type(confidence_score)),
                "model_type": model_type,
                "tensorflow_version": tf.__version__
            }
        }
        
        return response
        
    except Exception as e:
        print(f"❌ Error in debug predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🐛 Starting Pawnder Debug API...")
    print("🔗 Access at: http://localhost:8001")
    print("📚 API docs at: http://localhost:8001/docs")
    print("🔍 Model info at: http://localhost:8001/model-info")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
