# api_with_inference_script.py
# Enhanced API using your existing inference script

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import sys
from pathlib import Path
import json
import base64
import io
from PIL import Image

# Import your inference script
# Adjust the path as needed
sys.path.append('./ml/notebooks')
from dog_emotion_recognition_inference_script import DogEmotionPredictor

app = FastAPI(title="Pawnder Emotion API with Advanced Inference", version="2.0.0")

# Global predictor instance
predictor = None

# Enhanced emotion information (from your script)
EMOTION_ENHANCEMENTS = {
    "Happy/Playful": {
        "secondary_phrases": [
            "Play with me, please.",
            "Give me some attention",
            "Please, I'm a good dog",
            "I love you, I love you, I love you",
            "I'm friendly",
            "Ball! Ball! Ball! BALL!",
            "Best day ever!",
            "Let's go to the park!"
        ],
        "behavioral_context": "Dogs in this state are great for training, bonding, and exercise activities."
    },
    "Relaxed": {
        "secondary_phrases": [
            "ZZZzzzz",
            "Just relaxing",
            "Nap time",
            "Just 5 more minutes",
            "Belly rubs please",
            "I love nap o'clock",
            "This is my happy spot"
        ],
        "behavioral_context": "Perfect time for quiet bonding or letting your dog rest peacefully."
    },
    "Submissive/Appeasement": {
        "secondary_phrases": [
            "I'm friendly",
            "Please like me",
            "Can we be friends",
            "You're the boss",
            "Im a good dog, I promise",
            "We're cool, right?"
        ],
        "behavioral_context": "Your dog is showing respect and avoiding conflict. Build confidence gently."
    },
    "Curiosity/Alertness": {
        "secondary_phrases": [
            "Whatcha Doing?",
            "Who's there?",
            "What was that?",
            "Did you hear that too?",
            "New dog? Must check",
            "I smell something…."
        ],
        "behavioral_context": "Great time for mental stimulation and training exercises."
    },
    "Stressed": {
        "secondary_phrases": [
            "I need space",
            "I don't feel good",
            "I don't like this",
            "I've had enough",
            "This is overwhelming",
            "Get me out of here!"
        ],
        "behavioral_context": "Remove stressors and provide a calm, safe environment."
    },
    "Fearful/Anxious": {
        "secondary_phrases": [
            "Go away",
            "This scares me",
            "Please don't leave",
            "What was that?!",
            "I'm scared",
            "Too many strangers",
            "I'm intimidated"
        ],
        "behavioral_context": "Your dog needs gentle reassurance and removal of fear triggers."
    },
    "Aggressive/Threatening": {
        "secondary_phrases": [
            "Go away",
            "This is mine, don't touch",
            "I feel threatened",
            "Final warning, buddy",
            "Don't touch my human",
            "Don't test me"
        ],
        "behavioral_context": "DANGER: Professional intervention required immediately."
    }
}

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    global predictor
    try:
        # Initialize your predictor with the actual model path
        model_path = "path/to/your/trained/model.h5"  # Update this path
        config_path = "config.yaml"  # Update this path
        
        predictor = DogEmotionPredictor(
            model_path=model_path,
            config_path=config_path,
            use_gpu=True
        )
        print("✅ Advanced predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        # Fallback to basic predictor
        predictor = None

@app.get("/")
async def root():
    return {
        "message": "Pawnder Advanced Emotion Recognition API",
        "status": "running",
        "version": "2.0.0",
        "features": ["advanced_inference", "behavioral_analysis", "confidence_scoring", "explanations"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "model_loaded": predictor.model is not None if predictor else False,
        "supported_formats": ["jpg", "jpeg", "png", "mp4", "mov", "avi"]
    }

@app.post("/predict")
async def predict_emotion_file(file: UploadFile = File(...)):
    """Predict emotion from uploaded file using advanced inference"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Determine file type
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Use your inference script for image analysis
                result = predictor.predict_image(
                    image_path=tmp_file_path,
                    visualize=False,  # Don't show plots in API
                    save_output=False  # Don't save files in API
                )
                
                # Enhance the result with additional information
                enhanced_result = enhance_image_result(result, file.filename)
                return enhanced_result
                
            elif file_extension in ['.mp4', '.mov', '.avi', '.webm']:
                # Use your inference script for video analysis
                result = predictor.predict_video(
                    video_path=tmp_file_path,
                    output_path=None,  # Don't save output video
                    frame_interval=5,
                    downsample=True  # For faster API processing
                )
                
                # Enhance the result with additional information
                enhanced_result = enhance_video_result(result, file.filename)
                return enhanced_result
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_extension}"
                )
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-json")
async def predict_emotion_json(data: dict):
    """Predict emotion from base64 encoded image"""
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Use your inference script
            result = predictor.predict_image(
                image_path=tmp_file_path,
                visualize=False,
                save_output=False
            )
            
            # Enhance the result
            filename = data.get("filename", "uploaded_image.jpg")
            enhanced_result = enhance_image_result(result, filename)
            return enhanced_result
            
        finally:
            # Clean up
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def enhance_image_result(inference_result, filename):
    """Enhance the inference result with additional API-specific information"""
    
    emotion = inference_result.get('emotion')
    emotion_score = inference_result.get('emotion_score', 0)
    confidence = inference_result.get('confidence', 0)
    
    # Get enhancement data
    enhancement = EMOTION_ENHANCEMENTS.get(emotion, {})
    
    # Create comprehensive response
    enhanced_result = {
        "file_type": "image",
        "filename": filename,
        "analysis_type": "advanced_inference",
        
        # Core prediction data (from your script)
        "emotion": emotion,
        "emotion_score": emotion_score,
        "confidence": confidence,
        "all_emotions": inference_result.get('all_emotions', {}),
        
        # Advanced features from your script
        "explanation": inference_result.get('explanation', ''),
        "behaviors_detected": inference_result.get('behaviors_detected', []),
        "bounding_box": inference_result.get('bounding_box', []),
        "inference_time": inference_result.get('inference_time', 0),
        
        # API enhancements
        "secondary_phrases": enhancement.get('secondary_phrases', []),
        "behavioral_context": enhancement.get('behavioral_context', ''),
        "warning": inference_result.get('warning'),
        
        # Report card style information
        "report_card": {
            "primary_emotion": {
                "name": emotion,
                "score": emotion_score,
                "confidence_level": get_confidence_level(confidence)
            },
            "safety_assessment": get_safety_assessment(emotion),
            "interaction_tips": get_interaction_tips(emotion),
            "what_dog_might_be_thinking": get_random_phrase(enhancement.get('secondary_phrases', [])),
            "technical_details": {
                "model_confidence": confidence,
                "processing_time": inference_result.get('inference_time', 0),
                "behaviors_analyzed": len(inference_result.get('behaviors_detected', []))
            }
        }
    }
    
    return enhanced_result

def enhance_video_result(inference_result, filename):
    """Enhance the video inference result with additional information"""
    
    dominant_emotion = inference_result.get('dominant_emotion')
    emotion_distribution = inference_result.get('emotion_distribution', {})
    
    # Get enhancement data for dominant emotion
    enhancement = EMOTION_ENHANCEMENTS.get(dominant_emotion, {})
    
    # Calculate video statistics
    timeline = inference_result.get('emotion_timeline', [])
    stability_score = calculate_emotion_stability(timeline)
    transitions = calculate_transitions(timeline)
    
    enhanced_result = {
        "file_type": "video",
        "filename": filename,
        "analysis_type": "advanced_video_inference",
        
        # Core video data
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "emotion_timeline": timeline,
        "video_info": {
            "fps": inference_result.get('fps', 0),
            "total_frames": inference_result.get('total_frames', 0),
            "frames_analyzed": inference_result.get('frames_analyzed', 0),
            "processing_time": inference_result.get('processing_time', 0)
        },
        
        # Advanced video analysis
        "explanation": inference_result.get('explanation', ''),
        "emotion_transitions": inference_result.get('emotion_transitions', []),
        
        # Enhanced video features
        "stability_analysis": {
            "score": stability_score,
            "description": get_stability_description(stability_score)
        },
        "transition_analysis": {
            "count": len(transitions),
            "description": get_transition_description(len(transitions), len(timeline))
        },
        
        # Report card for video
        "report_card": {
            "overall_emotion": {
                "name": dominant_emotion,
                "dominance_percentage": emotion_distribution.get(dominant_emotion, 0) * 100
            },
            "video_summary": get_video_summary(dominant_emotion, stability_score, len(transitions)),
            "safety_assessment": get_safety_assessment(dominant_emotion),
            "interaction_tips": get_interaction_tips(dominant_emotion),
            "what_dog_might_be_thinking": get_random_phrase(enhancement.get('secondary_phrases', [])),
            "emotional_journey": describe_emotional_journey(timeline)
        }
    }
    
    return enhanced_result

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

def get_safety_assessment(emotion):
    """Get safety assessment for emotion"""
    safety_levels = {
        "Happy/Playful": {"level": "Safe", "color": "#4CAF50"},
        "Relaxed": {"level": "Safe", "color": "#4CAF50"},
        "Submissive/Appeasement": {"level": "Supervised", "color": "#FF9800"},
        "Curiosity/Alertness": {"level": "Supervised", "color": "#FF9800"},
        "Stressed": {"level": "Caution", "color": "#FF5722"},
        "Fearful/Anxious": {"level": "Concerning", "color": "#F44336"},
        "Aggressive/Threatening": {"level": "High Danger", "color": "#8B0000"}
    }
    return safety_levels.get(emotion, {"level": "Unknown", "color": "#808080"})

def get_interaction_tips(emotion):
    """Get specific interaction tips for the emotion"""
    tips = {
        "Happy/Playful": [
            "Perfect time for play and training",
            "Engage with toys and games",
            "Use positive reinforcement",
            "Great for bonding activities"
        ],
        "Relaxed": [
            "Allow peaceful rest",
            "Gentle petting if welcomed",
            "Maintain calm environment",
            "Good time for quiet bonding"
        ],
        "Submissive/Appeasement": [
            "Use gentle, encouraging voice",
            "Build confidence with treats",
            "Avoid dominant postures",
            "Give reassurance and space"
        ],
        "Curiosity/Alertness": [
            "Provide mental stimulation",
            "Great for training sessions",
            "Allow safe exploration",
            "Monitor for overstimulation"
        ],
        "Stressed": [
            "Remove stressors if possible",
            "Provide quiet, safe space",
            "Use calming techniques",
            "Consider professional help"
        ],
        "Fearful/Anxious": [
            "Move slowly and speak softly",
            "Remove fear triggers",
            "Give space and time",
            "Avoid forcing interaction"
        ],
        "Aggressive/Threatening": [
            "GIVE IMMEDIATE SPACE",
            "Do not approach or touch",
            "Consult professional immediately",
            "Ensure safety of all people"
        ]
    }
    return tips.get(emotion, ["Consult with a professional"])

def get_random_phrase(phrases):
    """Get a random phrase from the list"""
    if not phrases:
        return "Your dog has something to say!"
    
    import random
    return random.choice(phrases)

def calculate_emotion_stability(timeline):
    """Calculate how stable emotions are over time"""
    if len(timeline) < 2:
        return 1.0
    
    changes = 0
    prev_emotion = timeline[0].get('emotion')
    
    for frame in timeline[1:]:
        if frame.get('emotion') != prev_emotion:
            changes += 1
        prev_emotion = frame.get('emotion')
    
    stability = 1 - (changes / len(timeline))
    return round(stability, 3)

def get_stability_description(stability_score):
    """Get description for stability score"""
    if stability_score >= 0.8:
        return "Very stable - your dog maintained consistent emotions"
    elif stability_score >= 0.6:
        return "Moderately stable - some emotional changes occurred"
    elif stability_score >= 0.4:
        return "Somewhat unstable - multiple emotional shifts"
    else:
        return "Highly variable - frequent emotional changes"

def calculate_transitions(timeline):
    """Calculate emotion transitions"""
    transitions = []
    prev_emotion = None
    
    for frame in timeline:
        current_emotion = frame.get('emotion')
        if prev_emotion and current_emotion != prev_emotion:
            transitions.append({
                'from': prev_emotion,
                'to': current_emotion,
                'time': frame.get('time', 0)
            })
        prev_emotion = current_emotion
    
    return transitions

def get_transition_description(transition_count, total_frames):
    """Get description for transition analysis"""
    if transition_count == 0:
        return "No emotional changes - consistent throughout"
    elif transition_count == 1:
        return "Single emotional change during the video"
    elif transition_count < total_frames * 0.1:
        return "Few emotional changes - relatively stable"
    else:
        return "Multiple emotional changes - dynamic emotional state"

def get_video_summary(dominant_emotion, stability_score, transition_count):
    """Generate a video summary"""
    summary = f"Your dog primarily showed {dominant_emotion} throughout the video"
    
    if stability_score >= 0.8:
        summary += " and maintained this emotion consistently."
    elif transition_count > 3:
        summary += " but showed several emotional changes, indicating a dynamic state."
    else:
        summary += " with some natural emotional variation."
    
    return summary

def describe_emotional_journey(timeline):
    """Describe the emotional journey through the video"""
    if len(timeline) < 3:
        return "Brief analysis - not enough frames for detailed journey"
    
    start_emotion = timeline[0].get('emotion')
    end_emotion = timeline[-1].get('emotion')
    
    if start_emotion == end_emotion:
        return f"Your dog started and ended feeling {start_emotion}"
    else:
        return f"Your dog's emotions evolved from {start_emotion} to {end_emotion}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)