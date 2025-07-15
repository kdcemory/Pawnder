# app.py - Complete Cloud Run FastAPI version with CORS fixes
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

# Complete emotion data with all enhanced features
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

class ImageRequest(BaseModel):
    image: str
    filename: str = "image.jpg"

def load_model():
    """Load model in cloud environment"""
    global model, model_type, infer_func, input_shapes
    
    try:
        # Try different model paths (cloud vs local)
        model_paths = [
            "/app/saved_model",  # Cloud Run path
            "./saved_model",     # Local path
            "saved_model"        # Relative path
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"🎯 Loading SavedModel from: {model_path}")
                model = tf.saved_model.load(model_path)
                infer_func = model.signatures["serving_default"]
                input_shapes = infer_func.structured_input_signature[1]
                model_type = "savedmodel"
                
                print(f"✅ SavedModel loaded successfully!")
                print(f"📊 Input shapes: {[(name, spec.shape) for name, spec in input_shapes.items()]}")
                return
        
        raise Exception("Model not found in any expected location")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def preprocess_image(image_data):
    """Preprocess image for model input"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def calculate_confidence_from_probabilities(emotion_probs):
    """Calculate confidence based on emotion probability distribution"""
    max_prob = np.max(emotion_probs)
    
    # Calculate entropy-based confidence
    epsilon = 1e-8
    probs_safe = emotion_probs + epsilon
    entropy = -np.sum(probs_safe * np.log(probs_safe))
    max_entropy = -np.log(1.0/len(emotion_probs))
    entropy_confidence = 1.0 - (entropy / max_entropy)
    
    # Difference between top two probabilities
    sorted_probs = np.sort(emotion_probs)[::-1]
    if len(sorted_probs) >= 2:
        separation_confidence = sorted_probs[0] - sorted_probs[1]
    else:
        separation_confidence = sorted_probs[0]
    
    # Combine methods
    combined_confidence = (
        0.4 * max_prob +
        0.4 * entropy_confidence +
        0.2 * separation_confidence
    )
    
    confidence = np.clip(combined_confidence, 0.5, 0.95)
    return float(confidence)

def predict_with_savedmodel(img_array):
    """Make prediction using SavedModel - handles single output models"""
    try:
        # Create inputs based on model signature
        input_names = list(input_shapes.keys())
        inputs = {}
        
        for name in input_names:
            if 'image' in name.lower():
                inputs[name] = tf.constant(img_array.astype(np.float32))
            elif 'behavior' in name.lower():
                behavior_shape = input_shapes[name].shape.as_list()
                behavior_size = behavior_shape[1] if len(behavior_shape) > 1 else 46
                behavior_input = np.zeros((1, behavior_size), dtype=np.float32)
                inputs[name] = tf.constant(behavior_input)
        
        print(f"🔍 Making prediction with inputs: {list(inputs.keys())}")
        
        # Run inference
        outputs = infer_func(**inputs)
        
        print(f"📤 Model returned {len(outputs)} output(s)")
        
        # Handle single output (emotion probabilities only)
        if len(outputs) == 1:
            output_key = list(outputs.keys())[0]
            emotion_probs = outputs[output_key].numpy()[0]
            
            print(f"📊 Got emotion probabilities from '{output_key}': {emotion_probs}")
            
            # Calculate confidence from probability distribution
            confidence = calculate_confidence_from_probabilities(emotion_probs)
            
            print(f"🎯 Calculated confidence: {confidence:.3f}")
            
            return emotion_probs, confidence
        
        # Handle multiple outputs
        else:
            emotion_probs = None
            confidence = None
            
            for key, value in outputs.items():
                tensor_val = value.numpy()
                if tensor_val.shape[-1] == len(CLASS_NAMES):
                    emotion_probs = tensor_val[0]
                    print(f"📊 Found emotion probabilities in '{key}'")
                elif tensor_val.shape[-1] == 1:
                    confidence = tensor_val[0][0]
                    print(f"🎯 Found confidence in '{key}'")
            
            if confidence is None and emotion_probs is not None:
                confidence = calculate_confidence_from_probabilities(emotion_probs)
                print(f"🎯 Calculated confidence from probabilities: {confidence:.3f}")
            
            if emotion_probs is None:
                first_output = list(outputs.values())[0].numpy()[0]
                emotion_probs = first_output
                print(f"📊 Using first output as emotion probabilities")
            
            return emotion_probs, confidence
        
    except Exception as e:
        print(f"❌ SavedModel prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up - Loading model...")
    load_model()
    yield
    print("🔄 Shutting down...")

# Create FastAPI app first
app = FastAPI(
    title="Pawnder Enhanced API (Cloud)", 
    version="2.0.0",
    description="Dog emotion analysis API with CORS support for FlutterFlow",
    lifespan=lifespan
)

# CORS configuration - FIXED for FlutterFlow
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.flutterflow.io",
        "https://storage.googleapis.com",
        "https://firestore.googleapis.com",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "https://*.web.app",
        "https://*.firebaseapp.com",
        "*"  # Allow all for now - restrict later
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "User-Agent",
        "Cache-Control",
        "Pragma"
    ],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)

# Add preflight handler for all routes
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests"""
    return {
        "message": "OK",
        "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "path": path
    }

@app.get("/")
async def root():
    return {
        "message": "Pawnder Enhanced Dog Emotion API (Cloud)", 
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_type": model_type,
        "environment": "cloud_run",
        "cors_enabled": True,
        "supported_origins": ["FlutterFlow", "Firebase", "Local development"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "supported_formats": ["jpg", "jpeg", "png"],
        "features": ["emotion_analysis", "report_cards", "safety_assessment", "interaction_guidance"],
        "cors_status": "enabled",
        "note": "Enhanced API with CORS fixes for FlutterFlow integration"
    }

@app.get("/cors-test")
async def cors_test():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "CORS is working correctly!",
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "cors_enabled": True
    }

@app.post("/predict-json")
async def predict_emotion(request: ImageRequest):
    """Enhanced emotion prediction with comprehensive report card"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        print(f"\n🔍 Processing image: {request.filename}")
        
        # Decode and preprocess image
        image_data = base64.b64decode(request.image)
        img_array = preprocess_image(image_data)
        
        # Make prediction
        emotion_probs, confidence_score = predict_with_savedmodel(img_array)
        
        # Ensure we have valid values
        if emotion_probs is None:
            raise HTTPException(status_code=500, detail="Failed to get emotion probabilities")
        
        if confidence_score is None:
            confidence_score = 0.8
            print("⚠️ No confidence score, using default 0.8")
        
        # Get results
        predicted_idx = np.argmax(emotion_probs)
        predicted_emotion = CLASS_NAMES[predicted_idx]
        emotion_score = float(emotion_probs[predicted_idx])
        confidence_score = float(confidence_score)
        
        print(f"📊 Result: {predicted_emotion} (score: {emotion_score:.3f}, confidence: {confidence_score:.3f})")
        
        # Get emotion information
        emotion_info = EMOTION_DATA[predicted_emotion]
        
        # Create all emotions dictionary
        all_emotions = {
            CLASS_NAMES[i]: float(emotion_probs[i]) 
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
                    "percentage": round(confidence_score * 100, 1),
                    "note": "Confidence calculated from emotion probability distribution"
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
                "model_type": model_type,
                "tensorflow_version": tf.__version__,
                "confidence_method": "calculated_from_probabilities",
                "environment": "cloud_run",
                "cors_enabled": True
            }
        }
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

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
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting Pawnder Enhanced API on port {port}")
    print(f"🌐 CORS enabled for FlutterFlow integration")
    uvicorn.run(app, host="0.0.0.0", port=port)