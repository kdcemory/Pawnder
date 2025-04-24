# Dog Emotion Recognition Inference Script
# This script handles inference on new images/videos to predict dog emotions

import os
import numpy as np
import tensorflow as tf
import cv2
import json
import yaml
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import torch  # For future integration with YOLO dog detector
import json
from PIL import Image, ExifTags
import time
from sklearn.neighbors import KNeighborsClassifier

class DogEmotionPredictor:
    """Class for making predictions with trained dog emotion models"""
    
    def __init__(self, model_path, config_path="config.yaml", use_gpu=True):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to the trained model
            config_path (str): Path to configuration YAML file
            use_gpu (bool): Whether to use GPU for inference
        """
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path, use_gpu)
        self.emotions = self._load_emotion_list()
        self.behavioral_map = self._load_behavioral_map()
        self.img_size = tuple(self.config['model']['image_size'])
        
        # Initialize dog detector (placeholder for future integration)
        self.dog_detector = self._initialize_dog_detector()
        
        # Set confidence threshold from config
        self.confidence_threshold = self.config.get('inference', {}).get('confidence_threshold', 0.6)
        
        # Initialize temporal smoothing if needed for videos
        self.smoothing_window = 5  # Number of frames to consider for smoothing
        self.emotion_history = []  # Store recent emotion predictions for smoothing
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            # Use default configuration
            print(f"Config file not found: {config_path}, using defaults")
            default_config = {
                "data": {
                    "base_dir": ".",
                },
                "model": {
                    "image_size": [224, 224, 3],
                },
                "inference": {
                    "confidence_threshold": 0.6,
                    "behavior_threshold": 0.5,
                    "output_dir": "predictions"
                }
            }
            return default_config
    
    def _load_model(self, model_path, use_gpu):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set up GPU/CPU configuration
        if use_gpu and tf.config.list_physical_devices('GPU'):
            print("Using GPU for inference")
            # Allow memory growth to prevent TF from allocating all GPU memory
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"Error setting memory growth: {e}")
        else:
            print("Using CPU for inference")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Load the model
        try:
            # Custom objects dict if you have custom layers or losses
            custom_objects = {}
            
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Model loaded successfully from {model_path}")
            
            # Print model summary (optional)
            # model.summary()
            
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
    def _initialize_dog_detector(self):
        """Initialize dog detection model (placeholder for now)"""
        # This would be replaced with actual dog detector initialization
        # For example, loading a YOLO model for dog detection
        
        print("Dog detector not implemented yet. Using full image for analysis.")
        return None
    
    def _load_emotion_list(self):
        """Load the list of emotional categories"""
        # Try to load from a file
        emotion_list_path = os.path.join(
            self.config['data']['base_dir'],
            "emotion_list.json"
        )
        
        if os.path.exists(emotion_list_path):
            with open(emotion_list_path, 'r') as f:
                emotions = json.load(f)
            print(f"Loaded {len(emotions)} emotions from {emotion_list_path}")
            return emotions
        
        # Default emotions if file not found
        default_emotions = [
            "Happy/Playful", "Relaxed", "Submissive/Appeasement", "Curiosity/Alertness", "Stressed",
            "Fearful/Anxious", "Aggressive/Threatening"
        ]
        print(f"Using default list of {len(default_emotions)} emotions")
        return default_emotions
    
    def _load_behavioral_map(self):
        """Load the mapping between behaviors and emotions"""
        # Try to load from a file
        map_path = os.path.join(
            self.config['data']['base_dir'],
            "primary_behavior_matrix.json"
        )
        
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                behavior_map = json.load(f)
            print(f"Loaded behavior map with {len(behavior_map)} behaviors")
            return behavior_map
        
        # Return empty map if file not found
        print("Behavior map not found, explanations will be limited")
        return {}
    
    def _fix_image_orientation(self, img):
        """Fix image orientation based on EXIF data"""
        try:
            # Convert to PIL Image to access EXIF data
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Check for EXIF data
            if hasattr(pil_img, '_getexif') and pil_img._getexif():
                exif = dict(pil_img._getexif().items())
                
                # Get orientation tag (if it exists)
                orientation_tag = None
                for tag, tag_value in ExifTags.TAGS.items():
                    if tag_value == 'Orientation':
                        orientation_tag = tag
                        break
                
                if orientation_tag and orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    
                    # Apply appropriate rotation
                    if orientation == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif orientation == 8:
                        pil_img = pil_img.rotate(90, expand=True)
            
            # Convert back to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Warning: Could not process EXIF data: {e}")
        
        return img
    
    def _preprocess_image(self, img):
        """Preprocess an image for inference"""
        # Check if image is valid
        if img is None or img.size == 0:
            raise ValueError("Invalid image: empty or None")
        
        # Fix orientation based on EXIF data
        img = self._fix_image_orientation(img)
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # Convert BGR to RGB if needed
        if img.shape[2] == 3 and np.mean(img[:, :, 0]) < np.mean(img[:, :, 2]):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def _detect_dog(self, img):
        """Detect dog in the image and return bounding box"""
        # If we have a dog detector model
        if self.dog_detector is not None:
            # This would be replaced with actual detection code
            # For example: boxes = self.dog_detector(img)
            pass
        
        # For now, assume the entire image contains a dog
        height, width = img.shape[:2]
        return [0, 0, width, height]  # [x, y, w, h]
    
    def _predict_behaviors(self, img):
        """Predict behavioral indicators from the image"""
        # In a full implementation, this would analyze specific body parts
        # and predict behavioral indicators
        
        # For now, we use an empty behavior vector
        # The length depends on how many behavioral indicators we have
        behavior_count = len(self.behavioral_map) if self.behavioral_map else 30
        return np.zeros((1, behavior_count), dtype=np.float32)
    
    def _extract_behaviors_from_prediction(self, emotion, emotion_probs):
        """Extract likely behavioral indicators based on predicted emotion"""
        behaviors_present = []
        
        # Map back from emotion to likely behaviors if we have the mapping
        if self.behavioral_map:
            # Find behaviors most strongly associated with this emotion
            for behavior, emotion_map in self.behavioral_map.items():
                if emotion in emotion_map and emotion_map[emotion] > 3:  # Threshold for association strength
                    behaviors_present.append(behavior)
        
        # Limit to top 5 behaviors
        return behaviors_present[:5]
    
    def _get_explanation(self, emotion, behaviors_present, confidence):
        """Generate an explanation for the prediction"""
        explanation = f"I detected your dog is feeling {emotion} (confidence: {confidence:.1%}).\n\n"
        
        # Add behavioral indicators if available
        if behaviors_present and self.behavioral_map:
            explanation += "This is based on detected behavioral indicators:\n"
            for behavior in behaviors_present:
                if behavior in self.behavioral_map and emotion in self.behavioral_map[behavior]:
                    explanation += f"- {behavior}: This is commonly associated with {emotion} emotions.\n"
        
        # Add general description of the emotion
        emotion_descriptions = {
            "Happy/Playful": "Your dog may be happy, excited, wants to play, or are getting something they want. Dogs showing happiness are relaxed and at ease. They may have a loose, wagging tail, open mouth, soft eyes and relaxed ears.",
            "Relaxed": "Relaxed dogs are calm, content, safe, drowsy or sleeping. Content dogs appear calm and satisfied. They typically have a relaxed posture and may have soft eyes.",
            "Submissive/Appeasement": "Non-threatening, Appeasement or submission, can be related to another,  animal or to person. Can be mistaken for aggression but lacks tension in muzzle.",
            "Curiosity/Alertness": "Curious dogs show interest in their surroundings. They may have a forward-focused gaze, perked ears, and an inquisitive head tilt.",
            "Stressed": "Stressed dogs may be uneasy or unsure. They also may be stressed due to pain and or discomfort.",
            "Fearful/Anxious": "stress, discomfot, or fear. Can include pain. Anxious dogs may appear tense, with a lowered posture, ears back, and possible panting or lip licking.",
            "Fearful": "Fearful dogs typically show a lowered body posture, ears back, tail tucked, and may be trembling or trying to hide.",
            "Aggressive/Threatening": "Aggressive dogs may have a stiff posture, direct stare, raised hackles, and possibly bared teeth or growling."
        }
        
        if emotion in emotion_descriptions:
            explanation += f"\n{emotion_descriptions[emotion]}"
        
        # Add safety considerations
        safety_levels = {
            "Happy/Playful": "Safe",
            "Relaxed": "Safe",
            "Submissive/Appeasement": "Supervised",
            "Curiosity/Alertness": "Supervised",
            "Stressed": "Caution",
            "Fearful/Anxious": "Concerning",
            "Aggressive/Threatening": "High Danger"
        }
        
        if emotion in safety_levels:
            safety = safety_levels[emotion]
            explanation += f"\n\nSafety Category: {safety}"
            
            if safety == "Safe":
                explanation += "\nGenerally safe with minimal risk, suitable for children and inexperienced handlers."
            elif safety == "Supervised":
                explanation += "\nSafe for people with basic dog reading skills, requires supervision with children."
            elif safety == "Caution":
                explanation += "\nRequires experienced handling and understanding of de-escalation techniques."
            elif safety == "Concerning":
                explanation += "\nRequires professional/experienced handling with safety protocols."
            elif safety == "High Danger":
                explanation += "\nRequires immediate professional intervention, extremely high bite risk."
        
        return explanation
    
    def _apply_temporal_smoothing(self, emotion_probs):
        """Apply temporal smoothing to video predictions"""
        # Add current probabilities to history
        self.emotion_history.append(emotion_probs)
        
        # Keep only the most recent frames
        if len(self.emotion_history) > self.smoothing_window:
            self.emotion_history.pop(0)
        
        # Average the probabilities over the window
        smoothed_probs = np.mean(self.emotion_history, axis=0)
        
        return smoothed_probs
    
    def predict_image(self, image_path, visualize=True, save_output=True):
        """
        Predict emotion from an image
        
        Args:
            image_path (str): Path to the image
            visualize (bool): Whether to visualize the result
            save_output (bool): Whether to save the visualization
            
        Returns:
            dict: Prediction results
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Start timing
        start_time = time.time()
        
        # Detect dog in the image
        bbox = self._detect_dog(img_rgb)
        
        # Crop to bounding box
        x, y, w, h = bbox
        dog_img = img_rgb[y:y+h, x:x+w]
        
        # Preprocess image
        processed_img = self._preprocess_image(dog_img)
        
        # Predict behavioral indicators (placeholder for now)
        behavior_input = self._predict_behaviors(dog_img)
        
        # Make prediction
        inputs = {
            'image_input': np.expand_dims(processed_img, axis=0),
            'behavior_input': behavior_input
        }
        
        try:
            predictions = self.model.predict(inputs, verbose=0)
            
            # Handle different model output formats
            if isinstance(predictions, list) and len(predictions) >= 2:
                emotion_probs, confidence = predictions[:2]
            elif isinstance(predictions, dict):
                emotion_probs = predictions.get('emotion_output', predictions.get('output', None))
                confidence = predictions.get('confidence_output', np.ones((1, 1)))
            else:
                emotion_probs = predictions
                confidence = np.ones((1, 1))  # Default confidence
            
            # End timing
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")
            
        except Exception as e:
            raise ValueError(f"Error making prediction: {e}")
        
        # Get the predicted emotion
        emotion_idx = np.argmax(emotion_probs[0])
        emotion = self.emotions[emotion_idx]
        emotion_score = float(emotion_probs[0][emotion_idx])
        confidence_score = float(confidence[0][0]) if hasattr(confidence, 'shape') else 1.0
        
        # Extract behaviors based on predicted emotion
        behaviors_present = self._extract_behaviors_from_prediction(
            emotion, emotion_probs[0]
        )
        
        # Generate explanation
        explanation = self._get_explanation(emotion, behaviors_present, confidence_score)
        
        # Create result dictionary
        result = {
            'emotion': emotion,
            'emotion_score': emotion_score,
            'confidence': confidence_score,
            'explanation': explanation,
            'behaviors_detected': behaviors_present,
            'all_emotions': {self.emotions[i]: float(emotion_probs[0][i]) for i in range(len(self.emotions))},
            'bounding_box': [int(c) for c in bbox],
            'inference_time': inference_time
        }
        
        # Check if confidence is below threshold
        if confidence_score < self.confidence_threshold:
            print(f"Warning: Low confidence prediction ({confidence_score:.2f})")
            result['warning'] = f"This prediction has low confidence ({confidence_score:.2f}). The dog's body language may be unclear or the image quality could be affecting results."
        
        # Visualize result if requested
        if visualize or save_output:
            self._visualize_result(img_rgb, result, image_path, visualize, save_output)
        
        return result
    
    def predict_video(self, video_path, output_path=None, frame_interval=5, downsample=False):
        """
        Predict emotions from a video
        
        Args:
            video_path (str): Path to the video
            output_path (str): Path to save the output video
            frame_interval (int): Process every Nth frame
            downsample (bool): Whether to downsample the video for faster processing
            
        Returns:
            dict: Prediction results over time
        """
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Downsample if requested (for faster processing)
        if downsample and min(frame_width, frame_height) > 640:
            scale_factor = 640 / min(frame_width, frame_height)
            frame_width = int(frame_width * scale_factor)
            frame_height = int(frame_height * scale_factor)
            print(f"Downsampling video to {frame_width}x{frame_height} for faster processing")
        
        # Setup output video if requested
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Use platform-appropriate codec
            if os.name == 'nt':  # Windows
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:  # Linux/Mac
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps / frame_interval,  # Adjust output FPS based on frame interval
                (frame_width, frame_height)
            )
        
        # Reset temporal smoothing
        self.emotion_history = []
        
        # Process video frames
        results = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'frames_analyzed': 0,
            'emotion_timeline': [],
            'dominant_emotion': None,
            'processing_time': 0
        }
        
        frame_count = 0
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        start_time = time.time()
        
        print(f"Processing video: {video_path}")
        
        try:
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_count % frame_interval == 0:
                        # Downsample if requested
                        if downsample and min(frame.shape[1], frame.shape[0]) > 640:
                            scale_factor = 640 / min(frame.shape[1], frame.shape[0])
                            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                        
                        # Convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect dog
                        bbox = self._detect_dog(frame_rgb)
                        
                        # Crop to bounding box
                        x, y, w, h = bbox
                        dog_img = frame_rgb[y:y+h, x:x+w]
                        
                        # Preprocess image
                        processed_img = self._preprocess_image(dog_img)
                        
                        # Predict behavioral indicators
                        behavior_input = self._predict_behaviors(dog_img)
                        
                        # Make prediction
                        inputs = {
                            'image_input': np.expand_dims(processed_img, axis=0),
                            'behavior_input': behavior_input
                        }
                        
                        predictions = self.model.predict(inputs, verbose=0)
                        
                        # Handle different model output formats
                        if isinstance(predictions, list) and len(predictions) >= 2:
                            emotion_probs, confidence = predictions[:2]
                        elif isinstance(predictions, dict):
                            emotion_probs = predictions.get('emotion_output', predictions.get('output', None))
                            confidence = predictions.get('confidence_output', np.ones((1, 1)))
                        else:
                            emotion_probs = predictions
                            confidence = np.ones((1, 1))  # Default confidence
                        
                        # Apply temporal smoothing
                        smoothed_probs = self._apply_temporal_smoothing(emotion_probs[0])
                        
                        # Get the predicted emotion
                        emotion_idx = np.argmax(smoothed_probs)
                        emotion = self.emotions[emotion_idx]
                        emotion_score = float(smoothed_probs[emotion_idx])
                        confidence_score = float(confidence[0][0]) if hasattr(confidence, 'shape') else 1.0
                        
                        # Count emotions for dominant emotion calculation
                        emotion_counts[emotion] += 1
                        
                        # Add to timeline
                        results['emotion_timeline'].append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'emotion': emotion,
                            'emotion_score': emotion_score,
                            'confidence': confidence_score
                        })
                        
                        # Update count of analyzed frames
                        results['frames_analyzed'] += 1
                        
                        # Draw results on frame
                        if output_path:
                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Add text
                            text = f"{emotion}: {emotion_score:.2f}, Conf: {confidence_score:.2f}"
                            cv2.putText(
                                frame, text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                            )
                            
                            # Draw emotion bar chart
                            chart_width = 200
                            chart_height = 100
                            chart_x = 10
                            chart_y = 10
                            
                            # Background
                            cv2.rectangle(frame, (chart_x, chart_y), 
                                        (chart_x + chart_width, chart_y + chart_height), 
                                        (255, 255, 255), -1)
                            
                            # Get top 3 emotions
                            top_emotions = sorted(
                                [(self.emotions[i], smoothed_probs[i]) for i in range(len(self.emotions))],
                                key=lambda x: x[1],
                                reverse=True
                            )[:3]
                            
                            # Draw bars
                            for i, (em, score) in enumerate(top_emotions):
                                bar_height = 20
                                bar_y = chart_y + 10 + i * 30
                                bar_width = int(score * chart_width * 0.9)
                                
                                # Draw bar
                                cv2.rectangle(frame, (chart_x + 5, bar_y), 
                                            (chart_x + 5 + bar_width, bar_y + bar_height), 
                                            (0, 255, 0), -1)
                                
                                # Draw label
                                cv2.putText(frame, f"{em}: {score:.2f}", (chart_x + 10, bar_y + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            
                            # Write frame to output video
                            out.write(frame)
                    
                    frame_count += 1
                    pbar.update(1)
        
        finally:
            # Release resources
            cap.release()
            if output_path and 'out' in locals():
                out.release()
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            print(f"Video processing completed in {results['processing_time']:.2f} seconds")
        
        # Calculate dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        results['dominant_emotion'] = dominant_emotion
        
        # Generate explanation for dominant emotion
        behaviors_present = self._extract_behaviors_from_prediction(
            dominant_emotion, np.array([emotion_counts[e] / results['frames_analyzed'] 
                                       for e in self.emotions])
        )
        explanation = self._get_explanation(dominant_emotion, behaviors_present, 0.0)
        results['explanation'] = explanation
        
        # Generate emotion distribution
        results['emotion_distribution'] = {
            emotion: count / results['frames_analyzed'] if results['frames_analyzed'] > 0 else 0
            for emotion, count in emotion_counts.items()
        }
        
        # Calculate emotion transitions (changes over time)
        transitions = []
        prev_emotion = None
        for frame_data in results['emotion_timeline']:
            curr_emotion = frame_data['emotion']
            if prev_emotion and curr_emotion != prev_emotion:
                transitions.append({
                    'from': prev_emotion,
                    'to': curr_emotion,
                    'time': frame_data['time']
                })
            prev_emotion = curr_emotion
        
        results['emotion_transitions'] = transitions
        
        # Save results to JSON if output path is provided
        if output_path:
            json_path = output_path.rsplit('.', 1)[0] + '_analysis.json'
            with open(json_path, 'w') as f:
                # Remove numpy types for JSON serialization
                serializable_results = results.copy()
                if 'emotion_timeline' in serializable_results:
                    serializable_results['emotion_timeline'] = [
                        {k: float(v) if isinstance(v, np.float32) else v 
                         for k, v in frame.items()}
                        for frame in serializable_results['emotion_timeline']
                    ]
                
                json.dump(serializable_results, f, indent=2)
            print(f"Analysis results saved to {json_path}")
        
        return results
    
    def _visualize_result(self, img, result, image_path, display=True, save=True):
        """Visualize prediction result"""
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Get prediction details
        emotion = result['emotion']
        emotion_score = result['emotion_score']
        confidence = result['confidence']
        bbox = result.get('bounding_box', [0, 0, vis_img.shape[1], vis_img.shape[0]])
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Display image with bounding box
        plt.subplot(1, 2, 1)
        plt.imshow(vis_img)
        
        # Draw bounding box
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)
        
        plt.title(f"Predicted: {emotion} ({emotion_score:.2f})")
        plt.axis('off')
        
        # Plot emotion probabilities
        plt.subplot(1, 2, 2)
        emotions = list(result['all_emotions'].keys())
        scores = list(result['all_emotions'].values())
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        emotions = [emotions[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Bar chart for top emotions (limit to 7 for readability)
        bars = plt.barh(emotions[:7], scores[:7], color='skyblue')
        
        # Add confidence score
        plt.axvline(x=confidence, color='red', linestyle='--', label=f'Confidence: {confidence:.2f}')
        
        # Highlight predicted emotion
        for i, e in enumerate(emotions[:7]):
            if e == emotion:
                bars[i].set_color('green')
        
        plt.xlabel('Score')
        plt.ylabel('Emotion')
        plt.title('Top Emotion Predictions')
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()
        
        # Add explanation text
        plt.figtext(0.5, 0.01, result['explanation'], wrap=True, horizontalalignment='center', fontsize=9)
        
        # Save if requested
        if save:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(
                self.config['data']['base_dir'],
                self.config.get('inference', {}).get('output_dir', 'predictions')
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_prediction.png")
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            
            # Save result as JSON
            json_path = os.path.join(output_dir, f"{base_name}_prediction.json")
            with open(json_path, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                serializable_result = {
                    k: (float(v) if isinstance(v, (np.float32, np.float64)) else
                        v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()
                }
                json.dump(serializable_result, f, indent=2)
            print(f"Prediction results saved to {json_path}")
        
        # Display if requested
        if display:
            plt.show()
        else:
            plt.close()

    def batch_process_directory(self, input_dir, output_dir=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all images in a directory
        
        Args:
            input_dir (str): Directory containing images
            output_dir (str): Directory to save results (defaults to predictions in config)
            extensions (tuple): File extensions to process
            
        Returns:
            list: Results for all processed images
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(
                self.config['data']['base_dir'],
                self.config.get('inference', {}).get('output_dir', 'predictions')
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
        
        if not image_files:
            print(f"No images with extensions {extensions} found in {input_dir}")
            return []
        
        # Process each image
        all_results = []
        
        print(f"Processing {len(image_files)} images from {input_dir}")
        for img_path in tqdm(image_files):
            try:
                # Predict with no visualization, but save output
                result = self.predict_image(
                    str(img_path),
                    visualize=False,
                    save_output=True
                )
                
                # Add file path to result
                result['file_path'] = str(img_path)
                all_results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save summary results
        summary_path = os.path.join(output_dir, "batch_summary.json")
        
        # Create summary statistics
        emotion_counts = {}
        confidence_sum = 0
        
        for result in all_results:
            emotion = result['emotion']
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
            
            confidence_sum += result['confidence']
        
        summary = {
            'total_images': len(all_results),
            'average_confidence': confidence_sum / len(all_results) if all_results else 0,
            'emotion_distribution': emotion_counts,
            'processing_time': sum(r.get('inference_time', 0) for r in all_results)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch processing complete. Summary saved to {summary_path}")
        return all_results


# Command-line interface
def parse_args():
    parser = argparse.ArgumentParser(description='Dog Emotion Recognition')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image, video, or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to output file or directory')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization (for images)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving output (for images)')
    parser.add_argument('--frame-interval', type=int, default=5, help='Process every Nth frame (for videos)')
    parser.add_argument('--downsample', action='store_true', help='Downsample video for faster processing')
    return parser.parse_args()


# Main function
def main():
    args = parse_args()
    
    # Initialize predictor
    predictor = DogEmotionPredictor(
        model_path=args.model,
        config_path=args.config,
        use_gpu=not args.no_gpu
    )
    
    # Process based on input type
    if args.batch:
        # Batch process directory
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        results = predictor.batch_process_directory(args.input, args.output)
        
    else:
        # Check if input is image or video
        if os.path.isdir(args.input):
            print(f"Error: {args.input} is a directory. Use --batch flag for directory processing.")
            return
        
        input_ext = os.path.splitext(args.input)[1].lower()
        is_video = input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if is_video:
            # Process video
            results = predictor.predict_video(
                video_path=args.input,
                output_path=args.output,
                frame_interval=args.frame_interval,
                downsample=args.downsample
            )
            
            # Print results
            print(f"\nVideo Analysis Results for {args.input}")
            print(f"Dominant Emotion: {results['dominant_emotion']}")
            print("\nEmotion Distribution:")
            for emotion, percent in sorted(
                results['emotion_distribution'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]:  # Show top 5 emotions
                print(f"  {emotion}: {percent:.1%}")
            
            # Print transitions if available
            if results.get('emotion_transitions'):
                print("\nSignificant Emotion Transitions:")
                for i, transition in enumerate(results['emotion_transitions'][:5]):  # Show first 5 transitions
                    print(f"  {i+1}. {transition['from']} â†’ {transition['to']} at {transition['time']:.1f}s")
            
            print(f"\nExplanation: {results['explanation']}")
            
        else:
            # Process image
            results = predictor.predict_image(
                image_path=args.input,
                visualize=not args.no_vis,
                save_output=not args.no_save
            )
            
            # Print results
            print(f"\nImage Analysis Results for {args.input}")
            print(f"Detected Emotion: {results['emotion']} (Score: {results['emotion_score']:.2f})")
            print(f"Confidence: {results['confidence']:.2f}")
            
            # Print warning if confidence is low
            if 'warning' in results:
                print(f"\nWarning: {results['warning']}")
                
            print(f"\nExplanation: {results['explanation']}")


if __name__ == "__main__":
    main()
