# app.py - Cloud Run optimized version
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from datetime import datetime
from contextlib import asynccontextmanager


# Global model variables
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

# [Include all your EMOTION_DATA dictionary here - copy from your local file]
EMOTION_DATA = {
    # Copy the entire EMOTION_DATA from your fixed_single_output_api.py
    "Happy/Playful": {
        "definition": "Your dog is experiencing joy, excitement, and wants to engage in play or interaction.",
        "safety": "Safe",
        "color": "#4CAF50",
        "advice": "Perfect time for play, training, and bonding activities.",
        "what_dog_thinks": "Play with me, please! This is the best day ever!",
        "body_language": ["Loose, wagging tail", "Open mouth with relaxed jaw"],
        "interaction_tips": ["Engage in play activities", "Use positive reinforcement training"]
    },
    # ... include all emotions
}

class ImageRequest(BaseModel):
    image: str
    filename: str = "image.jpg"

def load_model():
    """Load model in cloud environment"""
    global model, model_type, infer_func, input_shapes
    
    try:
        # In Cloud Run, model should be in /app/saved_model
        model_path = "/app/saved_model"
        if os.path.exists(model_path):
            print(f"🎯 Loading SavedModel from: {model_path}")
            model = tf.saved_model.load(model_path)
            infer_func = model.signatures["serving_default"]
            input_shapes = infer_func.structured_input_signature[1]
            model_type = "savedmodel"
            
            print(f"✅ SavedModel loaded successfully!")
            print(f"📊 Input shapes: {[(name, spec.shape) for name, spec in input_shapes.items()]}")
            return
        else:
            print(f"❌ Model not found at: {model_path}")
            raise Exception("Model not found in cloud environment")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Pawnder Enhanced API (Cloud)", 
    version="2.0.0",
    lifespan=lifespan
)
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [Include all your functions from fixed_single_output_api.py]
# - preprocess_image()
# - calculate_confidence_from_probabilities()
# - predict_with_savedmodel()
# - get_confidence_level()

@app.get("/")
async def root():
    return {
        "message": "Pawnder Enhanced Dog Emotion API (Cloud)", 
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_type": model_type,
        "environment": "cloud_run"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "supported_formats": ["jpg", "jpeg", "png"],
        "features": ["emotion_analysis", "report_cards", "safety_assessment", "interaction_guidance"],
        "note": "Enhanced API with full report cards"
    }

# Include your predict endpoints here
@app.post("/predict")
async def predict_emotion_file(file: UploadFile = File(...)):
    # Copy from your local file
    pass

@app.post("/predict-json")
async def predict_emotion(request: ImageRequest):
    # Copy from your local file
    pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)