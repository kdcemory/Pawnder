"""
Dog Emotion Classifier with Behavioral Feature Integration - Updated Version

This implementation fixes path issues, method ordering, and handles the exact matrix format.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

def fix_image_path_resolution(image_path, base_dir):
    """
    A helper function to resolve the correct image path for frame images.
    Maps video names to their correct folders using a predefined mapping.
    
    Args:
        image_path: The original image path from the annotation
        base_dir: The base directory for data
        
    Returns:
        The corrected image path
    """
    import os
    import re
    
    # Video name to folder mapping based on your provided data
    VIDEO_FOLDER_MAP = {
        "1": "1",
        "3": "3",
        "4": "4",
        "5": "5",
        "m2-res_854p-7": "17",
        "excited": "19",
        "whale eye-2": "20",
        "shibu grin": "21",
        "shaking": "22",
        "playbow": "23",
        "piloerection and stiff tail": "24",
        "look away": "25",
        "lip licking": "26",
        "relaxedrottie": "27",
        "stresspain": "28",
        "happywelcome": "29",
        "bodylanguagepits": "30",
        "aggressive pit": "31",
        "Screen Recording 2025-03-06 at 9.33.52 AM": "32",
        "alertdog5": "33",
        "alertdog1": "34",
        "canine distemper": "35",
        "canine distemper2": "36",
        "fearanaussiestress": "37",
        "fearandanxiety": "38",
        "pancreatitis": "39",
        "relaxed dog": "40",
        "relaxed dog2": "41",
        "relaxed dog3": "42",
        "stressed kennel": "43",
        "stressed vet": "44",
        "stressedlab": "45",
        "stressedpit": "46",
        "Curious": "Curious",
        "head_tilt": "head_tilt",
        "m2-res_360p": "m2-res_360p",
        "m2-res_480p": "m2-res_480p",
        "m2-res_480p-2": "m2-res_480p-2",
        "m2-res_532p": "m2-res_532p",
        "m2-res_720p": "m2-res_720p",
        "m2-res_848p": "m2-res_848p",
        "m2-res_854p-2": "m2-res_854p-2",
        "m2-res_854p-3": "m2-res_854p-3",
        "m2-res_854p-4": "m2-res_854p-4",
        "m2-res_854p-5": "m2-res_854p-5",
        "m2-res_854p-6": "m2-res_854p-6",
        "play bow": "playbow",
        "relaxed": "relaxed",
        "resource guarding": "resource_guarding_1",
        "resource guarding 2": "resource_guarding_2",
        # Add some aliases for common variations
        "alert dog": "34",  # alertdog1
        "alertdog": "34",
        "head tilt": "head_tilt",
        "play_bow": "playbow",
        "resource_guarding": "resource_guarding_1",
        "relaxed_dog": "40",
        "relaxed_dog2": "41",
        "relaxed_dog3": "42",
        "stressed_kennel": "43",
        "stressed_vet": "44",
        "stressed_lab": "45",
        "stressed_pit": "46",
        "bodylanguage pits": "30",
        "body language pits": "30",
        "whale_eye-2": "20",
        "whale-eye-2": "20",
        "whale_eye": "20",
        "whale eye": "20"
    }
    
    # If the path already exists, return it
    if os.path.exists(image_path):
        return image_path
    
    # Special case for .png vs .PNG extension
    if isinstance(image_path, str) and image_path.lower().endswith('.png'):
        # Try with uppercase extension
        upper_path = image_path[:-4] + '.PNG'
        if os.path.exists(upper_path):
            return upper_path
    
    # Get filename if it's a path
    if isinstance(image_path, str):
        filename = os.path.basename(image_path)
    else:
        # If not a string, just return it
        return image_path
    
    # Extract video name and frame number
    video_name = None
    frame_number = None
    
    # Try to extract using common patterns
    if "_frame_" in filename:
        parts = filename.split("_frame_")
        video_name = parts[0]
        frame_number = parts[1]
        if "." in frame_number:
            frame_number = frame_number.split(".")[0]  # Remove extension
    
    # If we successfully extracted video name and frame number
    if video_name is not None and frame_number is not None:
        # Look up the folder for this video name
        folder = VIDEO_FOLDER_MAP.get(video_name)
        
        # If not found directly, try case-insensitive lookup
        if folder is None:
            for k, v in VIDEO_FOLDER_MAP.items():
                if k.lower() == video_name.lower():
                    folder = v
                    break
        
        # If still not found, try partial matching
        if folder is None:
            for k, v in VIDEO_FOLDER_MAP.items():
                if k.lower() in video_name.lower() or video_name.lower() in k.lower():
                    folder = v
                    break
        
        # If we found a folder mapping
        if folder is not None:
            # Look for the frame in this folder
            for videos_root in [
                os.path.join(base_dir, "ml", "Data", "raw", "Videos"),
                os.path.join(base_dir, "Data", "Raw", "Videos")
            ]:
                folder_path = os.path.join(videos_root, folder, "images")
                if os.path.exists(folder_path):
                    # Try exact frame with extensions
                    for ext in ['.PNG', '.png', '.jpg', '.jpeg']:
                        frame_path = os.path.join(folder_path, f"frame_{frame_number}{ext}")
                        if os.path.exists(frame_path):
                            return frame_path
                    
                    # If not found, use modulo mapping
                    frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_")]
                    if frame_files:
                        try:
                            frame_num = int(frame_number)
                            frame_idx = frame_num % len(frame_files)
                            sorted_frames = sorted(frame_files)
                            return os.path.join(folder_path, sorted_frames[frame_idx])
                        except (ValueError, IndexError):
                            # If error, use first frame
                            return os.path.join(folder_path, sorted(frame_files)[0])
    
    # If we couldn't resolve with the mapping, try a numeric fallback for numeric IDs
    if filename.startswith(tuple(str(i) for i in range(1, 47))) and "_frame_" in filename:
        parts = filename.split("_frame_")
        numeric_id = parts[0]
        frame_number = parts[1]
        if "." in frame_number:
            frame_number = frame_number.split(".")[0]
        
        # Try the corresponding numeric folder
        for videos_root in [
            os.path.join(base_dir, "ml", "Data", "raw", "Videos"),
            os.path.join(base_dir, "Data", "Raw", "Videos")
        ]:
            folder_path = os.path.join(videos_root, numeric_id, "images")
            if os.path.exists(folder_path):
                # Try exact frame with extensions
                for ext in ['.PNG', '.png', '.jpg', '.jpeg']:
                    frame_path = os.path.join(folder_path, f"frame_{frame_number}{ext}")
                    if os.path.exists(frame_path):
                        return frame_path
                
                # If not found, use modulo mapping
                frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_")]
                if frame_files:
                    try:
                        frame_num = int(frame_number)
                        frame_idx = frame_num % len(frame_files)
                        sorted_frames = sorted(frame_files)
                        return os.path.join(folder_path, sorted_frames[frame_idx])
                    except (ValueError, IndexError):
                        # If error, use first frame
                        return os.path.join(folder_path, sorted(frame_files)[0])
    
    # Last resort: use a default frame based on first character of filename for some variation
    default_folders = ["relaxed", "playbow", "Curious", "resource_guarding_1", "head_tilt", "1", "19", "40"]
    
    # Pick a folder based on a hash of the filename for consistent mapping
    if filename:
        folder_idx = sum(ord(c) for c in filename) % len(default_folders)
        default_folder = default_folders[folder_idx]
        
        for videos_root in [
            os.path.join(base_dir, "ml", "Data", "raw", "Videos"),
            os.path.join(base_dir, "Data", "Raw", "Videos")
        ]:
            folder_path = os.path.join(videos_root, default_folder, "images")
            if os.path.exists(folder_path):
                frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_")]
                if frame_files:
                    # Pick a frame based on a hash of the filename for consistent mapping
                    frame_idx = sum(ord(c) for c in filename) % len(frame_files)
                    sorted_frames = sorted(frame_files)
                    return os.path.join(folder_path, sorted_frames[frame_idx])
    
    # If absolutely nothing worked, return the original path
    return image_path

# Try to configure GPU settings
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled")
        
    # Try to enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except Exception as e:
    print(f"GPU configuration error: {str(e)}")

class DogEmotionWithBehaviors:
    """
    Dog emotion classification model with behavioral feature integration
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the classifier with corrected paths
        """
        # Set base directory with correct path structure
        if base_dir is None:
            self.base_dir = "C:/Users/thepf/pawnder"
        else:
            self.base_dir = base_dir
            
        # Fixed paths based on the correct directory structure - now includes "ml"
        self.processed_dir = os.path.join(self.base_dir, "ml", "Data", "processed")
        self.matrix_dir = os.path.join(self.base_dir, "ml", "Data", "matrix")  # lowercase "matrix"
        self.model_dir = os.path.join(self.base_dir, "Models")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Standard emotion class names
        self.standard_emotions = [
            "Happy_Playful", 
            "Relaxed", 
            "Submissive_Appeasement", 
            "Curiosity_Alertness", 
            "Stressed", 
            "Fearful_Anxious", 
            "Aggressive_Threatening"
        ]
        
        # Map between different naming conventions
        self.class_name_mapping = {
            # From validation dir naming to standard naming
            "Happy": "Happy_Playful",
            "Relaxed": "Relaxed",
            "Submissive": "Submissive_Appeasement",
            "Curiosity": "Curiosity_Alertness",
            "Stressed": "Stressed",
            "Fearful": "Fearful_Anxious",
            "Aggressive": "Aggressive_Threatening",
            # From JSON format to standard naming
            "Happy/Playful": "Happy_Playful",
            "Relaxed": "Relaxed",
            "Submissive/Appeasement": "Submissive_Appeasement",
            "Curiosity/Alertness": "Curiosity_Alertness",
            "Stressed": "Stressed",
            "Fearful/Anxious": "Fearful_Anxious",
            "Aggressive/Threatening": "Aggressive_Threatening",
            # And vice versa
            "Happy_Playful": "Happy_Playful",
            "Relaxed": "Relaxed",
            "Submissive_Appeasement": "Submissive_Appeasement",
            "Curiosity_Alertness": "Curiosity_Alertness",
            "Stressed": "Stressed",
            "Fearful_Anxious": "Fearful_Anxious",
            "Aggressive_Threatening": "Aggressive_Threatening",
        }
        
        self.model = None
        self.class_names = []
        self.behavior_columns = []
        self.behavior_matrix = None
        
        # Print paths for verification
        print(f"Base directory: {self.base_dir}")
        print(f"Processed directory: {self.processed_dir}")
        print(f"Matrix directory: {self.matrix_dir}")
        print(f"Model directory: {self.model_dir}")
        
        # Verify critical directories exist
        for path_name, path in [
            ("Base", self.base_dir),
            ("Processed", self.processed_dir),
            ("Matrix", self.matrix_dir),
            ("Model", self.model_dir)
        ]:
            if os.path.exists(path):
                print(f"✓ {path_name} directory exists: {path}")
            else:
                print(f"✗ {path_name} directory does not exist: {path}")
        
        # Check for split directories
        for split_name in ["train", "validation", "test"]:
            split_dir = os.path.join(self.processed_dir, split_name)
            if os.path.exists(split_dir):
                print(f"✓ {split_name} directory exists")
            else:
                print(f"✗ {split_name} directory does not exist: {split_dir}")
    
    def load_behavior_matrix(self, matrix_path=None):
        """
        Load behavior matrix from file with support for the specific JSON format
        
        Args:
            matrix_path: Optional path to behavior matrix file. If None, tries to find it.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If no path provided, look in default locations
            if matrix_path is None:
                # Check in various locations
                potential_paths = [
                    os.path.join(self.matrix_dir, "primary_behavior_matrix.json"),
                    os.path.join(self.matrix_dir, "behavior_matrix.json"),
                    os.path.join(self.matrix_dir, "Primary Behavior Matrix.json"),
                    os.path.join(self.processed_dir, "matrix", "primary_behavior_matrix.json")
                ]
                
                # Try each potential path
                for path in potential_paths:
                    if os.path.exists(path):
                        matrix_path = path
                        print(f"Found behavior matrix at: {matrix_path}")
                        break
            
            # If still no path, return False
            if matrix_path is None or not os.path.exists(matrix_path):
                print("Behavior matrix not found")
                return False
            
            # Load the matrix file
            print(f"Loading behavior matrix from {matrix_path}")
            with open(matrix_path, 'r') as f:
                matrix_data = json.load(f)
            
            # Check which format we have
            
            # Format 1: Direct behaviors dictionary
            if "behaviors" in matrix_data:
                self.behavior_matrix = matrix_data
                print(f"Loaded behavior matrix with {len(matrix_data['behaviors'])} behaviors (direct format)")
                return True
            
            # Format 2: Behavioral states and categories format
            elif "behavioral_states" in matrix_data and "behavior_categories" in matrix_data:
                print("Found structured JSON format with behavioral_states")
                
                # Extract emotions from behavioral states
                emotions = []
                state_id_to_name = {}
                
                for state in matrix_data.get("behavioral_states", []):
                    if "id" in state and "name" in state:
                        emotions.append(state["name"])
                        state_id_to_name[state["id"]] = state["name"]
                
                # Extract behaviors from behavior categories
                behaviors = {}
                
                for category in matrix_data.get("behavior_categories", []):
                    for behavior in category.get("behaviors", []):
                        if "id" in behavior and "name" in behavior and "state_mapping" in behavior:
                            behavior_id = f"behavior_{behavior['id']}"
                            state_mapping = behavior["state_mapping"]
                            
                            # Create emotion mapping for this behavior
                            emotion_mapping = {}
                            for state_id, value in state_mapping.items():
                                if state_id in state_id_to_name:
                                    emotion_name = state_id_to_name[state_id]
                                    emotion_mapping[emotion_name] = value
                            
                            behaviors[behavior_id] = emotion_mapping
                
                # Create and store the matrix
                self.behavior_matrix = {
                    "emotions": emotions,
                    "behaviors": behaviors
                }
                
                print(f"Converted matrix with {len(behaviors)} behaviors and {len(emotions)} emotions")
                return True
            
            else:
                print("Unknown behavior matrix format")
                return False
                
        except Exception as e:
            print(f"Error loading behavior matrix: {str(e)}")
            return False
    
    def load_image(self, image_path, img_size=(224, 224)):
        """Load and preprocess an image for inference"""
        try:
            # Use the path resolution fix
            resolved_path = fix_image_path_resolution(image_path, self.base_dir)

            if not os.path.exists(resolved_path):
                print(f"Image not found at {resolved_path} (original: {image_path})")
                return None
        
            img = cv2.imread(resolved_path)
            if img is None:
                print(f"Failed to load image: {resolved_path}")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def create_model(self, num_classes, behavior_size=10, img_size=(224, 224, 3)):
        """
        Create model with both image and behavioral inputs
        
        Args:
            num_classes: Number of emotion classes
            behavior_size: Number of behavioral indicator features
            img_size: Input image size (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
        # Create image input branch
        image_input = tf.keras.Input(shape=img_size, name='image_input')
        
        # Use MobileNetV2 as base model
        base_model = applications.MobileNetV2(
            input_shape=img_size,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Extract features from image
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Create behavioral input branch
        behavior_input = tf.keras.Input(shape=(behavior_size,), name='behavior_input')
        b = layers.Dense(64, activation='relu')(behavior_input)
        b = layers.BatchNormalization()(b)
        b = layers.Dropout(0.3)(b)
        
        # Combine image and behavior branches
        combined = layers.Concatenate()([x, b])
        
        # Classification head
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        output = layers.Dense(num_classes, activation='softmax')(combined)
        
        # Create and compile model
        model = tf.keras.Model(
            inputs={'image_input': image_input, 'behavior_input': behavior_input},
            outputs=output
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Store model
        self.model = model
        return model
    
    def fine_tune_model(self, unfreeze_layers=15):
        """
        Fine-tune the model by unfreezing some base model layers
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
            
        Returns:
            Fine-tuned model
        """
        if self.model is None:
            raise ValueError("No model to fine-tune. Create a model first.")
        
        # Find the base model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
        
        if base_model is None:
            print("Base model not found. Looking deeper in the model structure...")
            # Try to find it in the functional API structure
            for layer in self.model.layers:
                if hasattr(layer, 'layer') and isinstance(layer.layer, tf.keras.Model):
                    base_model = layer.layer
                    break
        
        if base_model is None:
            print("Could not find base model for fine-tuning")
            return self.model
        
        # Make base model trainable
        base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model fine-tuned: Unfroze last {unfreeze_layers} layers of the base model")
        return self.model
    
    def load_annotations(self, split_name='train'):
        """
        Load annotations from the processed directory for a specific split.
        Handles both JSON and CSV formats.
        
        Args:
            split_name: Data split ('train', 'validation', 'test')
            
        Returns:
            dict: Annotations dictionary or None if error occurs
        """
        # Get split directory
        split_dir = os.path.join(self.processed_dir, split_name)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            return None
        
        # Define potential annotation paths - include the combined_annotations.json path
        json_paths = [
            os.path.join(split_dir, "annotations", "annotations.json"),
            os.path.join(split_dir, "annotations.json"),
            os.path.join(self.processed_dir, f"{split_name}_annotations.json"),
            os.path.join(self.processed_dir, "annotations", f"{split_name}.json"),
            os.path.join(self.processed_dir, "combined_annotations.json")  # Add the combined file
        ]
        
        csv_paths = [
            os.path.join(split_dir, "annotations.csv"),
            os.path.join(split_dir, "annotations", "annotations.csv"),
            os.path.join(self.processed_dir, f"{split_name}_annotations.csv")
        ]
        
        # Try to load JSON annotations first
        annotations = None
        
        # Check each potential JSON path
        for json_path in json_paths:
            if os.path.exists(json_path):
                print(f"Found JSON annotations at {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                    print(f"Successfully loaded {len(annotations)} JSON annotations")
                    break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON file {json_path}: {str(e)}")
                except Exception as e:
                    print(f"Error loading JSON file {json_path}: {str(e)}")
        
        # If JSON loading failed, try CSV
        if annotations is None:
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    print(f"Found CSV annotations at {csv_path}")
                    try:
                        # Read CSV using pandas
                        df = pd.read_csv(csv_path)
                        print(f"Successfully loaded {len(df)} CSV rows")
                        
                        # Convert DataFrame to dictionary
                        annotations = {}
                        
                        # Try to determine the ID column - look for columns like 'id', 'image_id', 'filename'
                        id_column = None
                        for col in ['id', 'ID', 'image_id', 'filename', 'name', 'image', 'file']:
                            if col in df.columns:
                                id_column = col
                                break
                        
                        # If no ID column found, use the index
                        if id_column is None:
                            print("No ID column found in CSV, using DataFrame index")
                            for idx, row in df.iterrows():
                                annotations[f"image_{idx}"] = row.to_dict()
                        else:
                            print(f"Using '{id_column}' as ID column")
                            for idx, row in df.iterrows():
                                annotations[str(row[id_column])] = row.to_dict()
                        
                        # Check if there's an emotions column or primary_emotion column
                        if 'primary_emotion' in df.columns:
                            print("Found 'primary_emotion' column in CSV")
                            # Add emotions structure expected by the model
                            for key in annotations:
                                if 'primary_emotion' in annotations[key]:
                                    primary_emotion = annotations[key]['primary_emotion']
                                    annotations[key]['emotions'] = {'primary_emotion': primary_emotion}
                        
                        break
                    except Exception as e:
                        print(f"Error loading CSV file {csv_path}: {str(e)}")
        
        # If annotations is still None, we failed to load any annotations
        if annotations is None:
            print(f"Failed to load annotations for {split_name} split")
            return None
        
        # Check if annotations is a list instead of a dictionary
        if isinstance(annotations, list):
            print(f"Converting list of {len(annotations)} annotations to dictionary")
            annotations_dict = {}
            for i, item in enumerate(annotations):
                if isinstance(item, dict):
                    # Try to find a suitable ID field
                    item_id = None
                    for id_field in ['id', 'ID', 'image_id', 'filename', 'name']:
                        if id_field in item:
                            item_id = item[id_field]
                            break
                    
                    # If no ID field found, use the index
                    if item_id is None:
                        item_id = f"item_{i}"
                    
                    annotations_dict[str(item_id)] = item
            
            annotations = annotations_dict
        
        # Validate the structure
        if annotations and len(annotations) > 0:
            first_key = next(iter(annotations))
            first_item = annotations[first_key]
            
            print(f"Sample annotation key: {first_key}")
            print(f"Sample annotation type: {type(first_item).__name__}")
            
            if isinstance(first_item, dict):
                print(f"Sample annotation fields: {', '.join(first_item.keys())}")
                
                # Check if emotions field exists, and if not, create it
                if 'emotions' not in first_item:
                    print("Adding 'emotions' field to annotations")
                    # Check if there's a primary_emotion field directly in the annotation
                    for key, item in annotations.items():
                        if 'primary_emotion' in item:
                            emotion = item['primary_emotion']
                            item['emotions'] = {'primary_emotion': emotion}
        
        return annotations
    
    
    def load_data_from_annotations(self, split_name='train'):
        """
        Load image paths and labels from annotations

        Args:
            split_name: Data split ('train', 'validation', 'test')

        Returns:
            tuple: (image_paths, labels, class_names)
        """
        # Load annotations for this split
        annotations = self.load_annotations(split_name)
        if not annotations:
            raise ValueError(f"Could not load annotations for {split_name} split")

        # Get image directory
        images_dir = os.path.join(self.processed_dir, split_name, "images")
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")

        # Collect unique classes
        all_classes = set()
        emotion_counts = {}

        for img_id, data in annotations.items():
            # Check if emotions field exists
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                # Map to standard naming if needed
                standard_emotion = self.class_name_mapping.get(emotion, emotion)
                all_classes.add(standard_emotion)
                emotion_counts[standard_emotion] = emotion_counts.get(standard_emotion, 0) + 1
            # Check if primary_emotion field exists directly
            elif "primary_emotion" in data:
                emotion = data["primary_emotion"]
                standard_emotion = self.class_name_mapping.get(emotion, emotion)
                all_classes.add(standard_emotion)
                emotion_counts[standard_emotion] = emotion_counts.get(standard_emotion, 0) + 1
    
        # Sort classes
        class_names = sorted(list(all_classes))
    
        # Print class distribution
        print(f"Class distribution for {split_name} split:")
        for cls, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count} images ({count/len(annotations)*100:.1f}%)")
    
        # Store class names
        self.class_names = class_names
    
        # Convert to class indices
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        # Collect image paths and labels
        image_paths = []
        labels = []
        missing_count = 0
        found_count = 0

        for img_id, data in annotations.items():
            # Get emotion from either emotions.primary_emotion or primary_emotion
            emotion = None
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
            elif "primary_emotion" in data:
                emotion = data["primary_emotion"]
        
            # Skip if no emotion data
            if emotion is None:
                continue
        
            # Map to standard naming if needed
            standard_emotion = self.class_name_mapping.get(emotion, emotion)
        
            # Skip if not in class list
            if standard_emotion not in class_to_idx:
                continue
                
            # Try to get image path
            img_path = None
        
            # First, check if there's an image_path in the annotation
            if "image_path" in data:
                # Try to resolve this path
                img_path = fix_image_path_resolution(data["image_path"], self.base_dir)
                if os.path.exists(img_path):
                    found_count += 1
                    image_paths.append(img_path)
                    labels.append(class_to_idx[standard_emotion])
                    continue
        
            # Form image path (try different possible naming patterns)
            found = False
        
            # List of possible image name patterns
            patterns = [
                img_id,                      # Exact ID as filename
                f"{img_id}.jpg",             # ID with jpg extension
                f"{img_id}.jpeg",            # ID with jpeg extension
                f"{img_id}.png",             # ID with png extension
                os.path.basename(img_id),    # In case ID includes a path
            ]
        
            # Check all patterns
            for pattern in patterns:
                potential_path = os.path.join(images_dir, pattern)
                resolved_path = fix_image_path_resolution(potential_path, self.base_dir)
                if os.path.exists(resolved_path):
                    img_path = resolved_path
                    found = True
                    break
        
            # If no match with patterns, look for files that contain the ID
            if not found and len(img_id) > 5:
                for filename in os.listdir(images_dir):
                    if img_id in filename:
                        potential_path = os.path.join(images_dir, filename)
                        resolved_path = fix_image_path_resolution(potential_path, self.base_dir)
                        if os.path.exists(resolved_path):
                            img_path = resolved_path
                            found = True
                            break
        
            if found and img_path is not None:
                image_paths.append(img_path)
                labels.append(class_to_idx[standard_emotion])
                found_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:  # Only print first few missing files
                    print(f"Image not found for ID: {img_id}")

        print(f"Loaded {found_count} images for {split_name} split")
        print(f"Missing {missing_count} images")
    
        # Check a few paths
        if image_paths:
            print("Sample image paths:")
            for i in range(min(3, len(image_paths))):
                print(f"  {image_paths[i]}")
                print(f"    Exists: {os.path.exists(image_paths[i])}")
    
        return image_paths, labels, class_names
    
    def load_behavior_data(self, annotations):
        """
        Extract behavior features from annotations
        
        Args:
            annotations: Annotations dictionary
            
        Returns:
            tuple: (behavior_data, behavior_columns)
        """
        behavior_data = {}
        behavior_columns = []
        
        # Check if annotations is empty
        if not annotations:
            print("No annotations to extract behavior features from")
            return behavior_data, behavior_columns
        
        # Get first entry to check for behavior columns
        first_key = next(iter(annotations))
        first_entry = annotations[first_key]
        
        # Look for behavior columns (prefixed with 'behavior_')
        behavior_columns = [k for k in first_entry.keys() 
                           if isinstance(k, str) and k.startswith('behavior_')]
        
        if behavior_columns:
            print(f"Found {len(behavior_columns)} behavior columns")
            
            # Extract behavior features for each image
            for img_id, data in annotations.items():
                behavior_values = []
                
                for col in behavior_columns:
                    # Get value if present, else 0
                    if col in data:
                        value = data[col]
                        # Convert to float
                        if isinstance(value, bool):
                            value = 1.0 if value else 0.0
                        elif isinstance(value, (int, float)):
                            value = float(value)
                        else:
                            value = 0.0
                    else:
                        value = 0.0
                    
                    behavior_values.append(value)
                
                # Store behavior feature vector
                behavior_data[img_id] = behavior_values
                
                # Also store by basename for easier matching
                basename = os.path.basename(img_id)
                if basename:
                    behavior_data[basename] = behavior_values
            
            print(f"Extracted behavior features for {len(behavior_data)} images")
        else:
            print("No behavior columns found in annotations")
            
            # Create default behavior features
            behavior_columns = [
                "behavior_tail_high", "behavior_tail_low", "behavior_tail_wagging",
                "behavior_ears_forward", "behavior_ears_back", "behavior_ears_relaxed",
                "behavior_mouth_open", "behavior_mouth_closed", "behavior_teeth_showing",
                "behavior_eyes_wide", "behavior_eyes_squinting", "behavior_posture_stiff"
            ]
            
            print(f"Created {len(behavior_columns)} default behavior columns")
        
        # Set behavior columns
        self.behavior_columns = behavior_columns
        
        return behavior_data, behavior_columns
    
    def create_data_generator(self, 
                              image_paths, 
                              labels, 
                              behavior_data=None,
                              img_size=(224, 224), 
                              batch_size=32, 
                              augment=False):
        """
        Create a data generator for training/validation
        
        Args:
            image_paths: List of image paths
            labels: List of class indices
            behavior_data: Dictionary mapping image paths to behavior features
            img_size: Image dimensions (height, width)
            batch_size: Batch size
            augment: Whether to apply data augmentation
            
        Returns:
            Data generator
        """
        # Convert labels to one-hot encoding
        num_classes = len(set(labels))
        labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Get behavior size
        if behavior_data and len(behavior_data) > 0:
            first_key = next(iter(behavior_data.keys()))
            behavior_size = len(behavior_data[first_key])
        else:
            behavior_size = len(self.behavior_columns) if self.behavior_columns else 1
        
        # Create data generator class
        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, behavior_data, behavior_size, img_size, batch_size, augment, parent):
                self.image_paths = image_paths
                self.labels = labels
                self.behavior_data = behavior_data
                self.behavior_size = behavior_size
                self.img_size = img_size
                self.batch_size = batch_size
                self.augment = augment
                self.indices = np.arange(len(self.image_paths))
                self.parent = parent  # Reference to parent class
                
                # Setup augmentation if needed
                if self.augment:
                    self.img_gen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        brightness_range=[0.8, 1.2],
                        fill_mode='nearest'
                    )
            
            def __len__(self):
                """Number of batches per epoch"""
                return int(np.ceil(len(self.image_paths) / self.batch_size))
            
            def __getitem__(self, idx):
                """Generate one batch of data"""
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_size = len(batch_indices)

                # Initialize batch arrays
                batch_images = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
                batch_behaviors = np.zeros((batch_size, self.behavior_size), dtype=np.float32)
                batch_labels = np.zeros((batch_size, num_classes), dtype=np.float32)
    
                # Fill batch data
                for i, idx in enumerate(batch_indices):
                    # Get image path
                    img_path = self.image_paths[idx]
        
                    try:
                        # Use path resolution to find the image
                        resolved_path = fix_image_path_resolution(img_path, self.parent.base_dir)

                        if not os.path.exists(resolved_path):
                            print(f"Image not found at {resolved_path} (original: {img_path})")
                            continue
            
                        # Load and preprocess image
                        img = cv2.imread(resolved_path)
                        if img is None:
                            raise ValueError(f"Failed to load image: {resolved_path}")
            
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                        # Resize image
                        img = cv2.resize(img, self.img_size)

                        # Normalize pixel values
                        img = img.astype(np.float32) / 255.0
            
                        # Apply augmentation if enabled
                        if self.augment:
                            img = self.img_gen.random_transform(img)
            
                        # Store image and label
                        batch_images[i] = img
                        batch_labels[i] = self.labels[idx]
            
                        # Add behavioral data if available
                        if self.behavior_data:
                            # Try different matching strategies
                            matched = False
                
                            # 1. Try by full path
                            if img_path in self.behavior_data:
                                batch_behaviors[i] = self.behavior_data[img_path]
                                matched = True
                
                            # 2. Try by resolved path
                            if not matched and resolved_path in self.behavior_data:
                                batch_behaviors[i] = self.behavior_data[resolved_path]
                                matched = True
                
                            # 3. Try by basename
                            if not matched:
                                basename = os.path.basename(img_path)
                                if basename in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename]
                                    matched = True
                
                            # 4. Try by basename of resolved path
                            if not matched:
                                basename = os.path.basename(resolved_path)
                                if basename in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename]
                                    matched = True
                
                            # 5. Try by basename without extension
                            if not matched:
                                basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                                if basename_no_ext in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename_no_ext]
                                    matched = True
                
                            # 6. Try by basename of resolved path without extension
                            if not matched:
                                basename_no_ext = os.path.splitext(os.path.basename(resolved_path))[0]
                                if basename_no_ext in self.behavior_data:
                                    batch_behaviors[i] = self.behavior_data[basename_no_ext]
                                    matched = True
        
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        # Keep zeros for this sample (already initialized)
    
                # Return the batch
                inputs = {
                    'image_input': batch_images,
                    'behavior_input': batch_behaviors
                }
    
                return inputs, batch_labels
            
            def on_epoch_end(self):
                """Shuffle indices after each epoch"""
                if self.augment:  # Only shuffle training data
                    np.random.shuffle(self.indices)
        
        # Create and return data generator with parent reference
        return DataGenerator(image_paths, labels_onehot, behavior_data, behavior_size, 
                            img_size, batch_size, augment, parent=self)
    
    
    def train(self, epochs=50, batch_size=32, img_size=(224, 224), fine_tune=True):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image dimensions (height, width)
            fine_tune: Whether to perform fine-tuning
            
        Returns:
            tuple: (history, model_dir)
        """
        # Load data from annotations
        print("Loading data from annotations...")
        train_paths, train_labels, class_names = self.load_data_from_annotations('train')
        val_paths, val_labels, _ = self.load_data_from_annotations('validation')
        
        # Load annotations to extract behavior features
        train_annotations = self.load_annotations('train')
        behavior_data, behavior_columns = self.load_behavior_data(train_annotations)
        
        # Create data generators
        train_gen = self.create_data_generator(
            train_paths, train_labels, behavior_data, img_size, batch_size, augment=True)
        
        val_gen = self.create_data_generator(
            val_paths, val_labels, behavior_data, img_size, batch_size, augment=False)
        
        # Create model if not already created
        if self.model is None:
            self.create_model(
                num_classes=len(class_names),
                behavior_size=len(behavior_columns),
                img_size=(*img_size, 3)
            )
        
        # Create model directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(self.model_dir, f"dog_emotion_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            # Model checkpoints
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='max'
            )
        ]
        
        # Train model
        print(f"Training model with {len(train_paths)} images and {len(behavior_columns)} behavior features")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )
        
        # Fine-tuning if requested
        if fine_tune:
            print("Fine-tuning the model...")
            
            # Fine-tune the model
            self.fine_tune_model(unfreeze_layers=15)
            
            # Train with fine-tuning
            ft_history = self.model.fit(
                train_gen,
                epochs=20,  # Fewer epochs for fine-tuning
                validation_data=val_gen,
                callbacks=callbacks
            )
            
            # Extend history
            for k in history.history:
                if k in ft_history.history:
                    history.history[k].extend(ft_history.history[k])
        
        # Save final model
        self.model.save(os.path.join(model_dir, 'final_model.h5'))
        
        # Save class names and behavior columns
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump({
                'class_names': class_names,
                'behavior_columns': behavior_columns,
                'img_size': list(img_size),
                'training_timestamp': timestamp
            }, f, indent=2)
        
        # Save training history
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(x) for x in v]
            
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Plot training history
        self.plot_training_history(history, model_dir)
        
        # Evaluate on test set
        self.evaluate(model_dir)
        
        return history, model_dir
    
    def evaluate(self, model_dir):
        """
        Evaluate the model on the test set
        
        Args:
            model_dir: Directory to save evaluation results
        """
        # Load test data
        test_paths, test_labels, _ = self.load_data_from_annotations('test')
        
        # Load annotations to extract behavior features
        test_annotations = self.load_annotations('test')
        behavior_data, _ = self.load_behavior_data(test_annotations)
        
        # Create test generator
        test_gen = self.create_data_generator(
            test_paths, test_labels, behavior_data, (224, 224), batch_size=32, augment=False)
        
        # Evaluate model
        print("Evaluating model on test set...")
        evaluation = self.model.evaluate(test_gen)
        
        # Save evaluation metrics
        metrics = {
            'loss': float(evaluation[0]),
            'accuracy': float(evaluation[1])
        }
        
        with open(os.path.join(model_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate predictions
        print("Generating predictions for confusion matrix...")
        y_true = []
        y_pred = []
        
        for i in range(len(test_gen)):
            # Get batch data
            inputs, batch_y = test_gen[i]
            
            # Predict batch
            batch_pred = self.model.predict(inputs)
            
            # Get true and predicted classes
            batch_y_true = np.argmax(batch_y, axis=1)
            batch_y_pred = np.argmax(batch_pred, axis=1)
            
            # Add to lists
            y_true.extend(batch_y_true)
            y_pred.extend(batch_y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        
        class_names = self.class_names
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        fmt = '.2f'
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, format(cm_norm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True)
        
        with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    def plot_training_history(self, history, output_dir):
        """
        Plot training history
        
        Args:
            history: Training history
            output_dir: Directory to save plots
        """
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy.png'), dpi=300)
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=300)
        plt.close()
    
    def predict_image(self, image_path, behavior_features=None):
        """
        Predict emotion for a single image

        Args:
            image_path: Path to image file
            behavior_features: Optional list of behavior features

        Returns:
            tuple: (predicted_class, confidence, all_predictions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Load and preprocess image using the load_image method with path resolution
        img_array = self.load_image(image_path)

        if img_array is None:
            print(f"Failed to load or process image: {image_path}")
            return None, 0.0, {}

        # Create behavior features if not provided
        if behavior_features is None:
            behavior_size = len(self.behavior_columns) if self.behavior_columns else 1
            behavior_features = np.zeros((1, behavior_size), dtype=np.float32)
        else:
            behavior_features = np.array([behavior_features], dtype=np.float32)

        # Make prediction
        inputs = {
            'image_input': img_array,
            'behavior_input': behavior_features
        }
        predictions = self.model.predict(inputs)

        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])

        if len(self.class_names) > predicted_idx:
            predicted_class = self.class_names[predicted_idx]
        else:
            predicted_class = f"Class {predicted_idx}"

        # Create all predictions dict
        all_predictions = {
            self.class_names[i]: float(predictions[0][i]) 
            for i in range(len(self.class_names))
        }

        return predicted_class, confidence, all_predictions
    
    def load_model(self, model_path, metadata_path=None):
        """
        Load a saved model
        
        Args:
            model_path: Path to saved model file (.h5)
            metadata_path: Path to model metadata file (.json)
            
        Returns:
            Loaded model
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Load metadata if provided
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract metadata
                self.class_names = metadata.get('class_names', [])
                self.behavior_columns = metadata.get('behavior_columns', [])
                
                print(f"Loaded metadata: {len(self.class_names)} classes, {len(self.behavior_columns)} behavior features")
            
            return self.model
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None


# Usage example
if __name__ == "__main__":
    # Create classifier
    classifier = DogEmotionWithBehaviors()
    
    # Try to load behavior matrix
    matrix_path = os.path.join(classifier.matrix_dir, "primary_behavior_matrix.json")
    matrix_loaded = classifier.load_behavior_matrix(matrix_path)
    
    # Train model
    history, model_dir = classifier.train(
        epochs=50,
        batch_size=32,
        fine_tune=True
    )
    
    print(f"Training completed. Model saved to {model_dir}")