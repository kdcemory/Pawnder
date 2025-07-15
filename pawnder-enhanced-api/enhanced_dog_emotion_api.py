# enhanced_api.py - Windows Local Development Version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json
import os
from datetime import datetime

app = FastAPI(title="Pawnder Enhanced API", version="2.0.0")

# Enhanced emotion data for rich responses
EMOTION_DATA = {
    "Happy/Playful": {
        "definition": "Your dog is experiencing joy, excitement, and wants to engage in play or interaction.",
        "safety": "Safe",
        "color": "#4CAF50",
        "advice": "Perfect time for play, training, and bonding activities. Dogs in this state are receptive to positive interactions.",
        "what_dog_thinks": "Play with me, please! This is the best day ever!",
        "body_language": [
            "Loose, wagging tail (often in a circular motion)",
            "Open mouth with relaxed jaw", 
            "Soft, bright eyes with relaxed expression",
            "Bouncy, loose body posture"
        ],
        "interaction_tips": [
            "Engage in play activities",
            "Use positive reinforcement training", 
            "Provide interactive toys",
            "Great time for bonding"
        ]
    },
    "Relaxed": {
        "definition": "Your dog is calm, content, and feels safe in their environment.",
        "safety": "Safe",
        "color": "#2196F3", 
        "advice": "Great time for gentle petting, quiet bonding, or allowing your dog to rest. Avoid high-energy activities.",
        "what_dog_thinks": "Life is good... just 5 more minutes of this perfect nap.",
        "body_language": [
            "Tail in neutral position or gently wagging",
            "Soft, half-closed or fully closed eyes",
            "Relaxed facial muscles",
            "Loose, comfortable body posture"
        ],
        "interaction_tips": [
            "Allow peaceful rest",
            "Gentle petting if welcomed",
            "Maintain calm environment", 
            "Good time for quiet bonding"
        ]
    },
    "Submissive/Appeasement": {
        "definition": "Your dog is showing non-threatening, deferential behavior to avoid conflict or show respect.",
        "safety": "Supervised",
        "color": "#FF9800",
        "advice": "Use gentle, reassuring voice. Avoid dominant postures. Build confidence with positive reinforcement.",
        "what_dog_thinks": "I'm a good dog, I promise! Can we be friends?",
        "body_language": [
            "Lowered body posture",
            "Tail tucked or low wagging",
            "Ears back or flattened",
            "Avoiding direct eye contact"
        ],
        "interaction_tips": [
            "Use gentle, encouraging voice",
            "Build confidence with treats",
            "Avoid dominant postures",
            "Give reassurance and space"
        ]
    },
    "Curiosity/Alertness": {
        "definition": "Your dog is actively investigating their environment and paying attention to stimuli.",
        "safety": "Supervised", 
        "color": "#9C27B0",
        "advice": "Good time for training and exploration. Monitor for overstimulation. Can transition quickly to other emotions.",
        "what_dog_thinks": "What's that? Did you hear that too? Must investigate!",
        "body_language": [
            "Erect, forward-facing ears",
            "Bright, focused eyes",
            "Head tilted or forward",
            "Upright, attentive posture"
        ],
        "interaction_tips": [
            "Provide mental stimulation",
            "Great for training sessions",
            "Allow safe exploration",
            "Monitor for overstimulation"
        ]
    },
    "Stressed": {
        "definition": "Your dog is experiencing discomfort, anxiety, or feeling overwhelmed by their situation.",
        "safety": "Caution",
        "color": "#FF5722",
        "advice": "Remove stressors if possible. Provide a calm environment. Consider professional help if stress persists.",
        "what_dog_thinks": "I need space... this is overwhelming. Get me out of here!",
        "body_language": [
            "Tense body posture", 
            "Panting when not hot or exercised",
            "Excessive drooling",
            "Pacing or restlessness"
        ],
        "interaction_tips": [
            "Remove stressors if possible",
            "Provide quiet, safe space",
            "Use calming techniques",
            "Consider professional help"
        ]
    },
    "Fearful/Anxious": {
        "definition": "Your dog is experiencing fear or anxiety and may feel threatened or unsafe.",
        "safety": "Concerning",
        "color": "#F44336",
        "advice": "Remove fear triggers. Move slowly and speak softly. Give space. Consider counter-conditioning training.",
        "what_dog_thinks": "I'm scared... please help me feel safe. What was that?!",
        "body_language": [
            "Cowering or crouching low",
            "Tail tucked tightly", 
            "Ears pinned back",
            "Trembling or shaking"
        ],
        "interaction_tips": [
            "Move slowly and speak softly",
            "Remove fear triggers",
            "Give space and time",
            "Avoid forcing interaction"
        ]
    },
    "Aggressive/Threatening": {
        "definition": "Your dog is displaying warning signals and may bite if the situation escalates.",
        "safety": "High Danger",
        "color": "#8B0000", 
        "advice": "DO NOT approach. Give immediate space. Avoid direct eye contact. Consult professional trainer/behaviorist immediately.",
        "what_dog_thinks": "Back off! This is mine, don't touch. Final warning!",
        "body_language": [
            "Stiff, rigid body posture",
            "Direct, hard stare",
            "Raised hackles (fur on neck/back)",
            "Bared teeth or snarling"
        ],
        "interaction_tips": [
            "GIVE IMMEDIATE SPACE",
            "Do not approach or touch",
            "Consult professional immediately", 
            "Ensure safety of all people"
        ]
    }
}

# Global model variable
model = None
CLASS_NAMES = list(EMOTION_DATA.keys())

class ImageRequest(BaseModel):
    image: str
    filename: str = "image.jpg"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

def load_model():
    """Load the trained model"""
    global model
    try:
        # Windows local paths for your model
        model_paths = [
            r"C:\Users\kelly\Documents\GitHub\Pawnder\Models\enhanced_dog_emotion_20250525-134150\best_model.keras",
            r"best_model.keras",  # If copied to api folder
            r".\best_model.keras",
            r"..\Models\enhanced_dog_emotion_20250525-134150\best_model.keras"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"🎯 Found model at: {model_path}")
                try:
                    model = tf.keras.models.load_model(model_path)
                    print(f"✅ Model loaded successfully!")
                    print(f"Model input shape: {[inp.shape for inp in model.inputs]}")
                    print(f"Model output shape: {[out.shape for out in model.outputs]}")
                    return
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue
        
        print("⚠️ No model found, using dummy model for testing")
        model = create_dummy_model()
        
    except Exception as e:
        print(f"❌ Error in model loading: {e}")
        model = create_dummy_model()

def create_dummy_model():
    """Create a dummy model for testing when real model isn't available"""
    print("🔧 Creating dummy model for testing...")
    
    class DummyModel:
        def predict(self, inputs):
            batch_size = 1
            # Create realistic emotion probabilities 
            emotion_probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)), size=batch_size)
            confidence = np.random.uniform(0.7, 0.95, size=(batch_size, 1))
            return [emotion_probs, confidence]
    
    return DummyModel()

def preprocess_image(image_data):
    """Preprocess image for model input"""
    try:
        # Decode base64 image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def get_confidence_level(confidence):
    """Convert confidence score to descriptive level"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Moderate"
    else:
        return "Low"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pawnder Enhanced Dog Emotion API", 
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "real" if model and not hasattr(model, "predict") else "dummy",
        "supported_formats": ["jpg", "jpeg", "png"],
        "features": ["emotion_analysis", "report_cards", "safety_assessment", "interaction_guidance"]
    }

@app.post("/predict-json")
async def predict_emotion(request: ImageRequest):
    """Enhanced emotion prediction with comprehensive report card"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Decode and preprocess image
        image_data = base64.b64decode(request.image)
        img_array = preprocess_image(image_data)
        
        # Create dummy behavior input (64 features)
        behavior_input = np.zeros((1, 64))
        
        # Prepare inputs for model
        inputs = {
            'image_input': img_array,
            'behavior_input': behavior_input
        }
        
        # Make prediction
        predictions = model.predict(inputs)
        emotion_probs, confidence = predictions
        
        # Get results
        predicted_idx = np.argmax(emotion_probs[0])
        predicted_emotion = CLASS_NAMES[predicted_idx]
        emotion_score = float(emotion_probs[0][predicted_idx])
        confidence_score = float(confidence[0][0])
        
        # Get emotion information
        emotion_info = EMOTION_DATA[predicted_emotion]
        
        # Create all emotions dictionary
        all_emotions = {
            CLASS_NAMES[i]: float(emotion_probs[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Generate comprehensive response
        response = {
            "success": True,
            "emotion": predicted_emotion,
            "confidence": confidence_score,
            "score": emotion_score,
            "filename": request.filename,
            
            # Enhanced report card for rich UI
            "report_card": {
                # Primary emotion info
                "primary_emotion": {
                    "name": predicted_emotion,
                    "definition": emotion_info["definition"],
                    "score": emotion_score,
                    "color": emotion_info["color"]
                },
                
                # Confidence analysis
                "confidence_analysis": {
                    "score": confidence_score,
                    "level": get_confidence_level(confidence_score),
                    "percentage": round(confidence_score * 100, 1)
                },
                
                # Safety assessment
                "safety_assessment": {
                    "level": emotion_info["safety"],
                    "color": emotion_info["color"],
                    "advice": emotion_info["advice"]
                },
                
                # What the dog might be thinking
                "dog_thoughts": emotion_info["what_dog_thinks"],
                
                # Body language indicators
                "body_language": emotion_info["body_language"],
                
                # Interaction guidance
                "interaction_tips": emotion_info["interaction_tips"],
                
                # All emotion scores for charts
                "emotion_spectrum": [
                    {
                        "emotion": emotion_name,
                        "score": score,
                        "percentage": round(score * 100, 1)
                    }
                    for emotion_name, score in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                ]
            },
            
            # Legacy compatibility
            "all_emotions": all_emotions,
            
            # Metadata
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_version": "enhanced_v2.0",
                "processing_location": "local_windows"
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

# For testing with uploaded files
from fastapi import File, UploadFile

@app.post("/predict")
async def predict_emotion_file(file: UploadFile = File(...)):
    """Predict emotion from uploaded file"""
    
    try:
        # Read file content
        content = await file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # Use the JSON endpoint
        request = ImageRequest(image=base64_image, filename=file.filename)
        return await predict_emotion(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Pawnder Enhanced API...")
    print("🔗 Access at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)