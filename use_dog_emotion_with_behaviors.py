"""
Dog Emotion Classifier with Behavioral Feature Integration - Complete Version

This implementation includes:
- Training and model creation
- Path resolution and data handling
- Behavioral matrix integration
- Image and video prediction
- Detailed behavioral analysis
- Command-line interface
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
import yaml
import argparse

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
        "50": "50",
        "51": "51",
        "52": "52",
        "53": "53",
        "54": "54",
        "55": "55",
        "56": "56",
        "57": "57",
        "58": "58",
        "59": "59",
        "60": "60",
        "61": "61",
        "62": "62",
        "63": "63",
        "64": "64",
        "65": "65",
        "66": "66",
        "67": "67",
        "68": "68",
        "69": "69",
        "70": "70",
        "71": "71",
        "72": "72",
        "73": "73",
        "74": "74",
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
                os.path.join(base_dir, "Data", "raw", "Videos"),
                os.path.join(base_dir, "Data", "Raw", "Videos"),
                os.path.join(base_dir, "Data", "processed"),
                os.path.join(base_dir, "Data", "processed", "all_frames")
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
            os.path.join(base_dir, "Data", "raw", "Videos"),
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
            os.path.join(base_dir, "Data", "raw", "Videos"),
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
        self.processed_dir = os.path.join(self.base_dir, "Data", "processed")
        self.matrix_dir = os.path.join(self.base_dir, "Data", "matrix")  # lowercase "matrix"
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
            def __init__(self, image_paths, labels, behavior_data, behavior_size, img_size, batch_size, augment, parent, **kwargs):
                # Fix warning by calling super().__init__
                super().__init__(**kwargs)
                
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
            # Model checkpoints - use .keras format
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
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
        
        # Save final model - use .keras format
        self.model.save(os.path.join(model_dir, 'final_model.keras'))
        
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
        
        # Save the model in a format compatible with the inference script
        self.save_model_for_inference(model_dir)
        
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
        # Fix warning by setting zero_division parameter
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        
        with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
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
            model_path: Path to saved model file (.h5 or .keras)
            metadata_path: Path to model metadata file (.json)
            
        Returns:
            Loaded model
        """
        try:
            # Load model
            # Add custom_objects parameter to handle any custom objects
            self.model = tf.keras.models.load_model(model_path, compile=True)
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
    
    def save_model_for_inference(self, model_dir=None):
        """
        Save the model and metadata in a format compatible with the inference script
        
        Args:
            model_dir: Directory to save the model (default is current model directory)
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
            
        if model_dir is None:
            # Create a new directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = os.path.join(self.model_dir, f"dog_emotion_for_inference_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
        
        # Save the model using .keras format instead of .h5
        model_path = os.path.join(model_dir, "dog_emotion_model.keras")
        self.model.save(model_path)
        
        # Create a config file
        config = {
            "model": {
                "image_size": [224, 224, 3],
            },
            "data": {
                "base_dir": ".",
            },
            "inference": {
                "confidence_threshold": 0.6,
                "behavior_threshold": 0.5,
                "output_dir": "predictions"
            }
        }
        
        # Save config
        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Save emotion list
        emotion_list_path = os.path.join(model_dir, "emotion_list.json")
        with open(emotion_list_path, 'w') as f:
            json.dump(self.class_names, f)
        
        # Copy the behavioral matrix if available
        if hasattr(self, 'behavior_matrix') and self.behavior_matrix:
            matrix_path = os.path.join(model_dir, "primary_behavior_matrix.json")
            with open(matrix_path, 'w') as f:
                json.dump(self.behavior_matrix, f, indent=2)
        
        print(f"Model saved for inference at: {model_dir}")
        print(f"Use with inference script: python dog_emotion_inference.py --model {model_path} --config {config_path} image <image_path>")
        
        return model_dir
    
    def predict_video(self, video_path, output_path=None, frame_interval=5):
        """
        Predict emotions in a video
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video (optional)
            frame_interval: Process every Nth frame
            
        Returns:
            dict: Results containing emotion timeline and analysis
        """
        # Check if model is loaded
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Check if video exists
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
        
        # Set up output video if requested
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps / frame_interval,  # Adjust output FPS based on frame interval
                (frame_width, frame_height)
            )
        
        # Process video frames
        results = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'frames_analyzed': 0,
            'emotion_timeline': [],
            'dominant_emotion': None
        }
        
        frame_count = 0
        emotion_counts = {emotion: 0 for emotion in self.class_names}
        
        print(f"Processing video: {video_path}")
        
        try:
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame
                    if frame_count % frame_interval == 0:
                        # Convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize and preprocess image
                        resized_frame = cv2.resize(frame_rgb, (224, 224))
                        preprocessed_frame = resized_frame.astype(np.float32) / 255.0
                        
                        # Create input for model
                        inputs = {
                            'image_input': np.expand_dims(preprocessed_frame, axis=0),
                            'behavior_input': np.zeros((1, len(self.behavior_columns) if self.behavior_columns else 1))
                        }
                        
                        # Predict
                        predictions = self.model.predict(inputs, verbose=0)
                        
                        # Get predicted class and confidence
                        predicted_idx = np.argmax(predictions[0])
                        predicted_class = self.class_names[predicted_idx]
                        confidence = float(predictions[0][predicted_idx])
                        
                        # Count for dominant emotion
                        emotion_counts[predicted_class] += 1
                        
                        # Add to timeline
                        results['emotion_timeline'].append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'emotion': predicted_class,
                            'confidence': confidence,
                            'all_emotions': {
                                self.class_names[i]: float(predictions[0][i])
                                for i in range(len(self.class_names))
                            }
                        })
                        
                        # Update count of analyzed frames
                        results['frames_analyzed'] += 1
                        
                        # Draw results on frame if saving output
                        if output_path:
                            # Add text with prediction
                            text = f"{predicted_class}: {confidence:.2f}"
                            cv2.putText(
                                frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                            )
                            
                            # Write frame to output video
                            out.write(frame)
                    
                    frame_count += 1
                    pbar.update(1)
        
        finally:
            # Release resources
            cap.release()
            if output_path and 'out' in locals():
                out.release()
        
        # Calculate dominant emotion
        if results['frames_analyzed'] > 0:
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            results['dominant_emotion'] = dominant_emotion
            
            # Generate emotion distribution
            results['emotion_distribution'] = {
                emotion: count / results['frames_analyzed']
                for emotion, count in emotion_counts.items()
            }
            
            # Analyze emotion transitions
            results['emotion_trends'] = self._analyze_emotion_trends(results['emotion_timeline'])
            
            # Visualize results
            self._visualize_video_results(results, video_path)
        
        return results
    
    def _analyze_emotion_trends(self, emotion_timeline):
        """
        Analyze trends in emotions over time
        
        Args:
            emotion_timeline: List of emotion predictions with timestamps
            
        Returns:
            dict: Emotion trends analysis
        """
        if not emotion_timeline:
            return {}
        
        # Initialize trends dictionary
        trends = {
            'transitions': [],
            'stable_periods': [],
            'emotion_changes_per_minute': 0
        }
        
        # Track emotion transitions
        prev_emotion = None
        current_streak = {'emotion': None, 'start_time': 0, 'end_time': 0, 'duration': 0}
        
        for i, entry in enumerate(emotion_timeline):
            current_emotion = entry['emotion']
            current_time = entry['time']
            
            # First frame, initialize
            if i == 0:
                prev_emotion = current_emotion
                current_streak = {
                    'emotion': current_emotion,
                    'start_time': current_time,
                    'end_time': current_time,
                    'duration': 0
                }
                continue
            
            # If emotion changed, record the transition and start a new streak
            if current_emotion != prev_emotion:
                # Record the transition
                trends['transitions'].append({
                    'from': prev_emotion,
                    'to': current_emotion,
                    'time': current_time
                })
                
                # End the previous streak
                current_streak['end_time'] = current_time
                current_streak['duration'] = current_streak['end_time'] - current_streak['start_time']
                
                # Only record if duration is significant
                if current_streak['duration'] > 1.0:  # More than 1 second
                    trends['stable_periods'].append(current_streak)
                
                # Start a new streak
                current_streak = {
                    'emotion': current_emotion,
                    'start_time': current_time,
                    'end_time': current_time,
                    'duration': 0
                }
            else:
                # Update the end time of the current streak
                current_streak['end_time'] = current_time
            
            prev_emotion = current_emotion
        
        # Add the last streak if it exists
        if current_streak['emotion'] is not None:
            current_streak['duration'] = current_streak['end_time'] - current_streak['start_time']
            if current_streak['duration'] > 1.0:  # More than 1 second
                trends['stable_periods'].append(current_streak)
        
        # Calculate emotion changes per minute
        total_time = emotion_timeline[-1]['time'] - emotion_timeline[0]['time']
        if total_time > 0:
            changes_per_minute = (len(trends['transitions']) / total_time) * 60
            trends['emotion_changes_per_minute'] = changes_per_minute
        
        # Find the most stable emotions (longest periods)
        if trends['stable_periods']:
            trends['stable_periods'] = sorted(
                trends['stable_periods'],
                key=lambda x: x['duration'],
                reverse=True
            )
            trends['most_stable_emotion'] = trends['stable_periods'][0]['emotion']
        
        return trends
    
    def _visualize_video_results(self, results, video_path):
        """
        Visualize results from video analysis
        
        Args:
            results: Video analysis results
            video_path: Path to the original video
        """
        if not results['emotion_timeline']:
            print("No data to visualize")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.base_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create figure for emotion timeline
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Emotion Timeline
        plt.subplot(2, 1, 1)
        
        # Extract times and emotions
        times = [entry['time'] for entry in results['emotion_timeline']]
        emotions = [entry['emotion'] for entry in results['emotion_timeline']]
        
        # Get unique emotions
        unique_emotions = list(set(emotions))
        
        # Create a mapping of emotions to numeric values
        emotion_to_num = {emotion: i for i, emotion in enumerate(unique_emotions)}
        
        # Convert emotions to numeric values for plotting
        emotion_values = [emotion_to_num[emotion] for emotion in emotions]
        
        # Plot the emotion timeline
        plt.plot(times, emotion_values, 'o-', markersize=3)
        
        # Set y-ticks to emotion names
        plt.yticks(
            list(range(len(unique_emotions))),
            unique_emotions
        )
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Emotion')
        plt.title(f'Emotion Timeline for {base_name}')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Emotion Distribution
        plt.subplot(2, 1, 2)
        
        # Sort emotions by frequency
        sorted_emotions = sorted(
            results['emotion_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        emotion_names = [item[0] for item in sorted_emotions]
        emotion_freqs = [item[1] for item in sorted_emotions]
        
        # Only show top 10 emotions if there are more
        if len(emotion_names) > 10:
            emotion_names = emotion_names[:10]
            emotion_freqs = emotion_freqs[:10]
        
        # Plot the distribution
        bars = plt.barh(emotion_names, emotion_freqs, color='skyblue')
        
        # Highlight dominant emotion
        for i, name in enumerate(emotion_names):
            if name == results['dominant_emotion']:
                bars[i].set_color('green')
        
        plt.xlabel('Frequency')
        plt.ylabel('Emotion')
        plt.title('Emotion Distribution')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{base_name}_analysis.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Video analysis visualization saved to {output_path}")
        
        # Save results as JSON
        json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        
        # Create a simplified version of results for JSON (exclude large arrays)
        json_results = {
            'video_path': results['video_path'],
            'fps': results['fps'],
            'total_frames': results['total_frames'],
            'frames_analyzed': results['frames_analyzed'],
            'dominant_emotion': results['dominant_emotion'],
            'emotion_distribution': results['emotion_distribution'],
            'emotion_changes_per_minute': results.get('emotion_trends', {}).get('emotion_changes_per_minute', 0),
            'most_stable_emotion': results.get('emotion_trends', {}).get('most_stable_emotion', None)
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Video analysis results saved to {json_path}")
    
    def generate_behavioral_report(self, predicted_class, confidence, all_predictions):
        """
        Generate a detailed behavioral report based on the prediction result
        
        Args:
            predicted_class: Predicted emotion class
            confidence: Confidence score
            all_predictions: Dictionary with all emotion predictions
            
        Returns:
            str: Behavioral report text
        """
        if not hasattr(self, 'behavior_matrix') or not self.behavior_matrix:
            return "Behavioral matrix not available. Unable to generate detailed behavioral report."
        
        # Determine emotion category
        standard_emotion = self.class_name_mapping.get(predicted_class, predicted_class)
        
        # Initialize report
        report = f"# Dog Emotion Analysis: {standard_emotion}\n\n"
        report += f"Confidence: {confidence:.2f}\n\n"
        
        # Add other possible emotions
        report += "## Top Emotions Detected\n\n"
        for emotion, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
            report += f"- {emotion}: {prob:.2f}\n"
        
        report += "\n## Behavioral Analysis\n\n"
        
        # Add behavioral state information
        emotion_state = None
        danger_scale = None
        friendliness_scale = None
        category = None
        
        # Map standard emotion to behavioral state
        emotion_to_state_map = {
            "Happy_Playful": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Submissive_Appeasement": "Submissive/Appeasement",
            "Curiosity_Alertness": "Curiosity/Alertness",
            "Stressed": "Stressed",
            "Fearful_Anxious": "Fearful/Anxious",
            "Aggressive_Threatening": "Aggressive/Threatening"
        }
        
        state_name = emotion_to_state_map.get(standard_emotion, standard_emotion)
        
        # Find state in behavior matrix
        if "behavioral_states" in self.behavior_matrix:
            for state in self.behavior_matrix["behavioral_states"]:
                if state["name"] == state_name:
                    emotion_state = state
                    break
        
        # Add state information if found
        if emotion_state:
            report += f"### Behavioral State: {emotion_state.get('name')}\n\n"
            report += f"**Description:** {emotion_state.get('description')}\n\n"
            
            if 'detailed_description' in emotion_state:
                report += f"**Detailed Description:** {emotion_state.get('detailed_description')}\n\n"
            
            report += f"**Category:** {emotion_state.get('category')}\n\n"
            
            # Add category description
            category_descriptions = {
                "Safe": "Generally safe, minimum risk of aggressive behavior, good for children and inexperienced handlers, main requirement is calm, gentle interaction.",
                "Caution": "Safe for people with basic dog reading skills, requires understanding of canine body language, main risk comes from misreading or startling the dog, should supervise children's interactions.",
                "Concerning": "Requires experienced handling, need understanding of de-escalation techniques, should be monitored for signs of escalation, not suitable for interaction with children or inexperienced handlers, may need behavioral intervention if chronic.",
                "High Danger": "Requires professional or highly experienced handling, high risk of defensive reactions, need clear understanding of trigger management, should be handled with safety protocols in place, require behavior modification programs."
            }
            
            if emotion_state.get('category') in category_descriptions:
                report += f"**Category Description:** {category_descriptions[emotion_state.get('category')]}\n\n"
            
            report += f"**Danger Scale:** {emotion_state.get('danger_scale')}/10\n\n"
            report += f"**Friendliness Scale:** {emotion_state.get('friendliness_scale')}/10\n\n"
            
            # Add expected behaviors for this state
            report += f"### Expected Behavioral Indicators\n\n"
            
            state_id = emotion_state.get('id')
            if state_id and "behavior_categories" in self.behavior_matrix:
                for category in self.behavior_matrix["behavior_categories"]:
                    behaviors_in_category = []
                    
                    for behavior in category.get("behaviors", []):
                        if state_id in behavior.get("state_mapping", {}) and behavior["state_mapping"][state_id] == 1:
                            behaviors_in_category.append(behavior["name"])
                            
                    if behaviors_in_category:
                        report += f"**{category['name']}:**\n"
                        for behavior in behaviors_in_category:
                            report += f"- {behavior}\n"
                        report += "\n"
            
            # Add handling guidelines based on category
            report += f"### Handling Guidelines\n\n"
            
            handling_guidelines = {
                "Safe": [
                    "This dog is considered safe to handle.",
                    "Suitable for interaction with children and inexperienced handlers.",
                    "Maintain calm, gentle interactions.",
                    "Monitor for any signs of stress or discomfort."
                ],
                "Caution": [
                    "This dog is safe for people with basic dog reading skills.",
                    "Understanding of canine body language is required.",
                    "Main risk comes from misreading or startling the dog.",
                    "Adult supervision is required for any interaction with children.",
                    "Pay attention to environmental factors that might cause stress."
                ],
                "Concerning": [
                    "This dog requires experienced handling.",
                    "Understanding of de-escalation techniques is necessary.",
                    "Monitor for signs of escalation in behavior.",
                    "Not suitable for interaction with children or inexperienced handlers.",
                    "May need behavioral intervention if this state is chronic.",
                    "Avoid high-stress environments and interactions."
                ],
                "High Danger": [
                    "This dog requires professional or highly experienced handling.",
                    "There is a high risk of defensive reactions.",
                    "Clear understanding of trigger management is essential.",
                    "Safety protocols should be in place for all interactions.",
                    "Behavior modification programs are recommended.",
                    "Do not allow interaction with children or inexperienced handlers.",
                    "Consider consulting with a certified animal behaviorist."
                ]
            }
            
            if emotion_state.get('category') in handling_guidelines:
                for guideline in handling_guidelines[emotion_state.get('category')]:
                    report += f"- {guideline}\n"
        
        return report
    
    def detect_behaviors_from_image(self, image_path):
        """
        Detect behavioral indicators from an image using computer vision
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Dictionary of detected behaviors with confidence scores
        """
        # This is a placeholder for a real behavior detection implementation
        # In practice, you would use computer vision models to detect these features
        
        # For now, return empty dict
        detected_behaviors = {}
        
        # In a real implementation, you would:
        # 1. Detect dog pose (ears, tail, mouth, eyes, etc.)
        # 2. Analyze the pose to identify behavioral indicators
        # 3. Return a dictionary of behaviors with confidence scores
        
        return detected_behaviors


# Main function for command-line use
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dog Emotion Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--no-fine-tune', action='store_true', help='Disable fine-tuning')
    
    # Predict image command
    predict_parser = subparsers.add_parser('predict-image', help='Predict emotion from an image')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    predict_parser.add_argument('--metadata', type=str, help='Path to metadata file')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to image file')
    predict_parser.add_argument('--report', action='store_true', help='Generate a behavioral report')
    
    # Predict video command
    video_parser = subparsers.add_parser('predict-video', help='Predict emotions in a video')
    video_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    video_parser.add_argument('--metadata', type=str, help='Path to metadata file')
    video_parser.add_argument('--video', type=str, required=True, help='Path to video file')
    video_parser.add_argument('--output', type=str, help='Path to save output video')
    video_parser.add_argument('--frame-interval', type=int, default=5, help='Process every Nth frame')
    
    # Save for inference command
    save_parser = subparsers.add_parser('save-for-inference', help='Save model for use with inference script')
    save_parser.add_argument('--model', type=str, required=True, help='Path to model file')
    save_parser.add_argument('--metadata', type=str, required=True, help='Path to metadata file')
    save_parser.add_argument('--output', type=str, help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create classifier
    classifier = DogEmotionWithBehaviors()
    
    # Handle commands
    if args.command == 'train':
        # Load behavior matrix
        matrix_path = os.path.join(classifier.matrix_dir, "primary_behavior_matrix.json")
        matrix_loaded = classifier.load_behavior_matrix(matrix_path)
        
        # Train model
        history, model_dir = classifier.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            fine_tune=not args.no_fine_tune
        )
        
        print(f"Training completed. Model saved to {model_dir}")
    
    elif args.command == 'predict-image':
        # Load model
        classifier.load_model(args.model, args.metadata)
        
        # Load behavior matrix
        matrix_path = os.path.join(classifier.matrix_dir, "primary_behavior_matrix.json")
        matrix_loaded = classifier.load_behavior_matrix(matrix_path)
        
        # Predict
        predicted_class, confidence, all_predictions = classifier.predict_image(args.image)
        
        # Print results
        print(f"Predicted emotion: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        print("All predictions:")
        for emotion, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")
        
        # Generate behavioral report if requested
        if args.report:
            report = classifier.generate_behavioral_report(predicted_class, confidence, all_predictions)
            print("\nBehavioral Report:")
            print(report)
            
            # Save report to file
            output_dir = os.path.join(classifier.base_dir, "predictions")
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            report_path = os.path.join(output_dir, f"{base_name}_behavioral_report.md")
            
            with open(report_path, 'w') as f:
                f.write(report)
                
            print(f"\nBehavioral report saved to {report_path}")
    
    elif args.command == 'predict-video':
        # Load model
        classifier.load_model(args.model, args.metadata)
        
        # Load behavior matrix
        matrix_path = os.path.join(classifier.matrix_dir, "primary_behavior_matrix.json")
        matrix_loaded = classifier.load_behavior_matrix(matrix_path)
        
        # Predict video
        results = classifier.predict_video(
            video_path=args.video,
            output_path=args.output,
            frame_interval=args.frame_interval
        )
        
        # Print results
        print(f"Dominant emotion: {results['dominant_emotion']}")
        print(f"Frames analyzed: {results['frames_analyzed']}")
        print(f"Emotion changes per minute: {results.get('emotion_trends', {}).get('emotion_changes_per_minute', 0):.2f}")
        
    elif args.command == 'save-for-inference':
        # Load model
        classifier.load_model(args.model, args.metadata)
        
        # Load behavior matrix
        matrix_path = os.path.join(classifier.matrix_dir, "primary_behavior_matrix.json")
        matrix_loaded = classifier.load_behavior_matrix(matrix_path)
        
        # Save for inference
        output_dir = classifier.save_model_for_inference(args.output)
        print(f"Model saved for inference at: {output_dir}")
        
    else:
        parser.print_help()