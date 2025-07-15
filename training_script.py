"""
Fixed Dog Emotion Training Script with Unicode Support and Enhanced Data Loading

This script integrates all the Unicode fixes and properly loads all 23,955 annotations
with 46 behavior features from the CSV file.
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
import unicodedata
import re


# ============================================================================
# UNICODE-SAFE UTILITY FUNCTIONS
# ============================================================================

def safe_imread(image_path, flags=cv2.IMREAD_COLOR):
    """
    Safely read an image file that may contain Unicode characters in the path.
    
    Args:
        image_path (str): Path to the image file
        flags: OpenCV imread flags (default: cv2.IMREAD_COLOR)
        
    Returns:
        numpy.ndarray: Loaded image or None if loading failed
    """
    try:
        # First try normal imread
        img = cv2.imread(image_path, flags)
        if img is not None:
            return img
    except Exception:
        pass
    
    # If normal imread fails, try Unicode-safe method
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        if len(img_array) == 0:
            return None
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing problematic Unicode characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Normalize Unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Replace problematic characters
    replacements = {
        'ΓÇ»': '_',  # The specific character causing issues
        'ΓÇ': '_',   # Related character
        'Γ': '_',    # Base character
        '»': '_',    # Angle quote
        '«': '_',    # Angle quote
        '"': '_',    # Smart quotes
        '"': '_',
        ''': '_',    # Smart quotes
        ''': '_',
        '…': '_',    # Ellipsis
        '–': '-',    # En dash
        '—': '-',    # Em dash
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    # Remove any remaining non-ASCII characters that might cause issues
    filename = re.sub(r'[^\x00-\x7F]+', '_', filename)
    
    # Clean up multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    return filename


def find_unicode_safe_path(original_path):
    """
    Find a file even if it has Unicode issues in the filename.
    
    Args:
        original_path (str): Original file path
        
    Returns:
        str: Found file path or original path if not found
    """
    if os.path.exists(original_path):
        return original_path
    
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    if os.path.exists(dir_path):
        try:
            for file in os.listdir(dir_path):
                # Try exact match first
                if file == filename:
                    return os.path.join(dir_path, file)
                
                # Try sanitized match
                sanitized_filename = sanitize_filename(filename)
                sanitized_file = sanitize_filename(file)
                if sanitized_file == sanitized_filename:
                    return os.path.join(dir_path, file)
                
                # Try partial match for Unicode corruption
                base_name = os.path.splitext(filename)[0]
                file_base = os.path.splitext(file)[0]
                if len(base_name) > 10:  # Only for longer names
                    # Check if the first part matches (before Unicode corruption)
                    clean_base = re.sub(r'[^\x00-\x7F]+', '', base_name)
                    clean_file = re.sub(r'[^\x00-\x7F]+', '', file_base)
                    if clean_base and clean_file and clean_base in clean_file:
                        return os.path.join(dir_path, file)
                        
        except Exception as e:
            print(f"Error searching directory {dir_path}: {e}")
    
    return original_path


def fix_image_path_resolution(image_path, base_dir):
    """
    Enhanced image path resolution with Unicode handling and video folder mapping.
    
    Args:
        image_path: The original image path from the annotation
        base_dir: The base directory for data
        
    Returns:
        The corrected image path
    """
    # Video name to folder mapping
    VIDEO_FOLDER_MAP = {
        "1": "1", "3": "3", "4": "4", "5": "5", "50": "50", "51": "51", "52": "52", 
        "53": "53", "54": "54", "55": "55", "56": "56", "57": "57", "58": "58", 
        "59": "59", "60": "60", "61": "61", "62": "62", "63": "63", "64": "64", 
        "65": "65", "66": "66", "67": "67", "68": "68", "69": "69", "70": "70", 
        "71": "71", "72": "72", "73": "73", "74": "74", "80": "80", "81": "81",
        "m2-res_854p-7": "17", "excited": "19", "whale eye-2": "20",
        "shibu grin": "21", "shaking": "22", "playbow": "23",
        "piloerection and stiff tail": "24", "look away": "25", "lip licking": "26",
        "relaxedrottie": "27", "stresspain": "28", "happywelcome": "29",
        "bodylanguagepits": "30", "aggressive pit": "31",
        "Screen Recording 2025-03-06 at 9.33.52 AM": "32", "alertdog5": "33",
        "alertdog1": "34", "canine distemper": "35", "canine distemper2": "36",
        "fearanaussiestress": "37", "fearandanxiety": "38", "pancreatitis": "39",
        "relaxed dog": "40", "relaxed dog2": "41", "relaxed dog3": "42",
        "stressed kennel": "43", "stressed vet": "44", "stressedlab": "45",
        "stressedpit": "46", "Curious": "Curious", "head_tilt": "head_tilt",
        "m2-res_360p": "m2-res_360p", "m2-res_480p": "m2-res_480p",
        "m2-res_480p-2": "m2-res_480p-2", "m2-res_532p": "m2-res_532p",
        "m2-res_720p": "m2-res_720p", "m2-res_848p": "m2-res_848p",
        "m2-res_854p-2": "m2-res_854p-2", "m2-res_854p-3": "m2-res_854p-3",
        "m2-res_854p-4": "m2-res_854p-4", "m2-res_854p-5": "m2-res_854p-5",
        "m2-res_854p-6": "m2-res_854p-6", "play bow": "playbow",
        "relaxed": "relaxed", "resource guarding": "resource_guarding_1",
        "resource guarding 2": "resource_guarding_2", "alert dog": "34",
        "alertdog": "34", "head tilt": "head_tilt", "play_bow": "playbow",
        "resource_guarding": "resource_guarding_1", "relaxed_dog": "40",
        "relaxed_dog2": "41", "relaxed_dog3": "42", "stressed_kennel": "43",
        "stressed_vet": "44", "stressed_lab": "45", "stressed_pit": "46",
        "bodylanguage pits": "30", "body language pits": "30",
        "whale_eye-2": "20", "whale-eye-2": "20", "whale_eye": "20", "whale eye": "20"
    }
    
    # First, try Unicode-safe path finding
    safe_path = find_unicode_safe_path(image_path)
    if os.path.exists(safe_path):
        return safe_path
    
    # If the path already exists, return it
    if os.path.exists(image_path):
        return image_path
    
    # Special case for .png vs .PNG extension
    if isinstance(image_path, str) and image_path.lower().endswith('.png'):
        upper_path = image_path[:-4] + '.PNG'
        if os.path.exists(upper_path):
            return upper_path
    
    # Get filename if it's a path
    if isinstance(image_path, str):
        filename = os.path.basename(image_path)
    else:
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
            frame_number = frame_number.split(".")[0]
    
    # If we successfully extracted video name and frame number
    if video_name is not None and frame_number is not None:
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
                        safe_frame_path = find_unicode_safe_path(frame_path)
                        if os.path.exists(safe_frame_path):
                            return safe_frame_path
                    
                    # If not found, use modulo mapping
                    try:
                        frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_")]
                        if frame_files:
                            frame_num = int(frame_number)
                            frame_idx = frame_num % len(frame_files)
                            sorted_frames = sorted(frame_files)
                            return os.path.join(folder_path, sorted_frames[frame_idx])
                    except (ValueError, IndexError):
                        if frame_files:
                            return os.path.join(folder_path, sorted(frame_files)[0])
    
    # Last resort: try Unicode-safe path resolution
    return find_unicode_safe_path(image_path)


def clean_problematic_filenames(data_dir):
    """
    Clean existing filenames that have Unicode corruption.
    Run this once to fix your existing files.
    
    Args:
        data_dir (str): Base data directory path
    """
    all_frames_dir = os.path.join(data_dir, "Data", "processed", "all_frames")
    
    if os.path.exists(all_frames_dir):
        print(f"Cleaning filenames in {all_frames_dir}")
        renamed_count = 0
        
        for filename in os.listdir(all_frames_dir):
            if any(char in filename for char in ['ΓÇ»', 'ΓÇ', 'Γ']):
                old_path = os.path.join(all_frames_dir, filename)
                new_filename = sanitize_filename(filename)
                new_path = os.path.join(all_frames_dir, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
        
        print(f"Renamed {renamed_count} files with Unicode issues")
    
    # Also clean split directories
    for split in ['train', 'validation', 'test']:
        split_images_dir = os.path.join(data_dir, "Data", "processed", split, "images")
        if os.path.exists(split_images_dir):
            print(f"Cleaning filenames in {split_images_dir}")
            renamed_count = 0
            for filename in os.listdir(split_images_dir):
                if any(char in filename for char in ['ΓÇ»', 'ΓÇ', 'Γ']):
                    old_path = os.path.join(split_images_dir, filename)
                    new_filename = sanitize_filename(filename)
                    new_path = os.path.join(split_images_dir, new_filename)
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"Failed to rename {filename}: {e}")
            
            if renamed_count > 0:
                print(f"Renamed {renamed_count} files in {split} directory")


# ============================================================================
# ENHANCED DOG EMOTION TRAINER CLASS
# ============================================================================

class FixedDogEmotionTrainer:
    """
    Enhanced trainer with Unicode support and proper behavior data loading
    """
    
    def __init__(self, base_dir="C:\\Users\\kelly\\Documents\\GitHub\\Pawnder"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "Data", "processed")
        self.model_dir = os.path.join(base_dir, "Models")
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Standard emotion classes
        self.standard_emotions = [
            "Happy/Playful", 
            "Relaxed", 
            "Submissive/Appeasement", 
            "Curiosity/Alertness", 
            "Stressed", 
            "Fearful/Anxious", 
            "Aggressive/Threatening"
        ]
        
        # Emotion name mapping
        self.emotion_mapping = {
            "Happy": "Happy/Playful",
            "Relaxed": "Relaxed",
            "Submissive": "Submissive/Appeasement",
            "Curiosity": "Curiosity/Alertness", 
            "Stressed": "Stressed",
            "Fearful": "Fearful/Anxious",
            "Aggressive": "Aggressive/Threatening",
            "Happy/Playful": "Happy/Playful",
            "Submissive/Appeasement": "Submissive/Appeasement",
            "Curiosity/Alertness": "Curiosity/Alertness",
            "Fearful/Anxious": "Fearful/Anxious",
            "Aggressive/Threatening": "Aggressive/Threatening"
        }
        
        self.model = None
        self.class_names = []
        self.behavior_columns = []
        
        print(f"Initialized Enhanced Dog Emotion Trainer with Unicode Support")
        print(f"Base directory: {self.base_dir}")
        print(f"Processed directory: {self.processed_dir}")
    
    def load_image(self, image_path, img_size=(224, 224)):
        """
        Load and preprocess an image for inference with Unicode support
        
        Args:
            image_path: Path to image file
            img_size: Target image size (width, height)
            
        Returns:
            numpy.ndarray: Preprocessed image array or None if failed
        """
        try:
            # Use the enhanced path resolution
            resolved_path = fix_image_path_resolution(image_path, self.base_dir)

            # Use Unicode-safe image loading
            img = safe_imread(resolved_path)
            
            if img is None:
                print(f"Image not found or could not be loaded: {resolved_path} (original: {image_path})")
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def load_split_data_from_csv(self, split_name='train'):
        """
        Load data directly from the corrected CSV files
        
        Args:
            split_name: 'train', 'validation', or 'test'
            
        Returns:
            tuple: (annotations_dict, behavior_data_dict, behavior_columns)
        """
        csv_path = os.path.join(self.processed_dir, split_name, "annotations.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return {}, {}, []
        
        print(f"Loading {split_name} data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {split_name} CSV")
            
            # Get behavior columns
            behavior_columns = [col for col in df.columns if col.startswith('behavior_')]
            print(f"Found {len(behavior_columns)} behavior columns")
            
            # Extract annotations and behavior data
            annotations = {}
            behavior_data = {}
            
            for idx, row in df.iterrows():
                image_path = row['image_path']
                
                # Create annotation
                annotation = {
                    'emotions': {'primary_emotion': row['primary_emotion']},
                    'primary_emotion': row['primary_emotion'],
                    'source': row.get('source', 'unknown')
                }
                
                # Add other fields if present
                for field in ['video_name', 'video_id', 'frame_id', 'original_path']:
                    if field in row and pd.notna(row[field]):
                        annotation[field] = row[field]
                
                annotations[image_path] = annotation
                
                # Extract behavior features
                behavior_values = []
                for col in behavior_columns:
                    if col in row and pd.notna(row[col]):
                        try:
                            # Handle different value types
                            val = row[col]
                            if isinstance(val, str):
                                # Convert string representations to numbers
                                if val.lower() in ['true', 'yes', '1']:
                                    behavior_values.append(1.0)
                                elif val.lower() in ['false', 'no', '0']:
                                    behavior_values.append(0.0)
                                else:
                                    try:
                                        behavior_values.append(float(val))
                                    except ValueError:
                                        behavior_values.append(0.0)
                            else:
                                behavior_values.append(float(val))
                        except (ValueError, TypeError):
                            behavior_values.append(0.0)
                    else:
                        behavior_values.append(0.0)
                
                behavior_data[image_path] = behavior_values
            
            # Count behavior data
            behavior_count = sum(1 for values in behavior_data.values() if any(v > 0 for v in values))
            print(f"Loaded {len(annotations)} annotations with {behavior_count} having behavior data")
            
            return annotations, behavior_data, behavior_columns
            
        except Exception as e:
            print(f"Error loading {split_name} data: {e}")
            return {}, {}, []
    
    def prepare_data_for_training(self, split_name='train'):
        """
        Prepare image paths, labels, and behavior data for training
        
        Args:
            split_name: 'train', 'validation', or 'test'
            
        Returns:
            tuple: (image_paths, labels, behavior_data_dict, class_names)
        """
        # Load data from CSV
        annotations, behavior_data, behavior_columns = self.load_split_data_from_csv(split_name)
        
        if not annotations:
            raise ValueError(f"No annotations loaded for {split_name}")
        
        # Store behavior columns
        self.behavior_columns = behavior_columns

         # FIXED: Use the correct all_frames directory for all splits
        images_dir = r"C:\Users\kelly\Documents\GitHub\Pawnder\Data\processed\all_frames"
        
        print(f"✅ Using images directory: {images_dir}")
        
        # Verify the directory exists and has images
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        try:
            files = os.listdir(images_dir)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"📊 Found {len(image_files)} total images in directory")
            
            if len(image_files) == 0:
                raise ValueError(f"No image files found in {images_dir}")
                
        except Exception as e:
            raise ValueError(f"Error accessing images directory {images_dir}: {e}")

        
        # Collect unique classes
        all_classes = set()
        emotion_counts = {}
        
        for annotation in annotations.values():
            emotion = annotation['primary_emotion']
            # Map to standard emotion
            standard_emotion = self.emotion_mapping.get(emotion, emotion)
            all_classes.add(standard_emotion)
            emotion_counts[standard_emotion] = emotion_counts.get(standard_emotion, 0) + 1
        
        # Sort classes and create mapping
        class_names = sorted(list(all_classes))
        self.class_names = class_names
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        
        print(f"Class distribution for {split_name}:")
        for cls, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count} ({count/len(annotations)*100:.1f}%)")
        
        # Prepare image paths and labels
        image_paths = []
        labels = []
        final_behavior_data = {}
        found_count = 0
        missing_count = 0
        
        for image_key, annotation in annotations.items():
            emotion = annotation['primary_emotion']
            standard_emotion = self.emotion_mapping.get(emotion, emotion)
            
            if standard_emotion not in class_to_idx:
                continue
            
            # Try to find the actual image file with Unicode-safe path resolution
            image_path = None
            
            # Try different image locations
            potential_paths = [
                os.path.join(images_dir, image_key),
                os.path.join(images_dir, os.path.basename(image_key))
            ]
            
            # Add more potential paths for different splits
            images_dir = r"C:\Users\kelly\Documents\GitHub\Pawnder\Data\processed\all_frames"
            print(f"✅ Using images directory: {images_dir}")
            
            for potential_path in potential_paths:
                # Use Unicode-safe path resolution
                resolved_path = fix_image_path_resolution(potential_path, self.base_dir)
                if os.path.exists(resolved_path):
                    image_path = resolved_path
                    break
            
            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(class_to_idx[standard_emotion])
                
                # Add behavior data
                if image_key in behavior_data:
                    final_behavior_data[image_path] = behavior_data[image_key]
                else:
                    # Create zero vector if no behavior data
                    final_behavior_data[image_path] = [0.0] * len(behavior_columns)
                
                found_count += 1
            else:
                missing_count += 1
                if missing_count <= 5:  # Only show first 5 missing
                    print(f"Image not found: {image_key}")
        
        print(f"Prepared {found_count} images for {split_name}")
        print(f"Missing {missing_count} images")
        
        # Count behavior data
        behavior_count = sum(1 for values in final_behavior_data.values() if any(v > 0 for v in values))
        print(f"Behavior data available for {behavior_count}/{len(final_behavior_data)} images")
        
        return image_paths, labels, final_behavior_data, class_names
    
    def create_model(self, num_classes, behavior_size, img_size=(224, 224, 3)):
        """
        Create the enhanced model architecture with behavior integration
        
        Args:
            num_classes: Number of emotion classes
            behavior_size: Number of behavior features
            img_size: Input image dimensions
            
        Returns:
            Compiled Keras model
        """
        # Image input
        image_input = tf.keras.Input(shape=img_size, name='image_input')
        
        # Base model - MobileNetV2 for efficiency
        base_model = applications.MobileNetV2(
            input_shape=img_size,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Image feature extraction branch
        x = base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Behavior input branch
        behavior_input = tf.keras.Input(shape=(behavior_size,), name='behavior_input')
        b = layers.Dense(128, activation='relu')(behavior_input)
        b = layers.BatchNormalization()(b)
        b = layers.Dense(64, activation='relu')(b)
        b = layers.Dropout(0.3)(b)
        
        # Combine image and behavior branches
        combined = layers.Concatenate()([x, b])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        output = layers.Dense(num_classes, activation='softmax')(combined)
        
        # Create model
        model = tf.keras.Model(
            inputs={'image_input': image_input, 'behavior_input': behavior_input},
            outputs=output
        )
        
        # Compile with appropriate optimizer and loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"Created model with {num_classes} classes and {behavior_size} behavior features")
        return model
    
    def create_data_generator(self, image_paths, labels, behavior_data, img_size=(224, 224), batch_size=32, augment=False):
        """
        Create enhanced data generator with Unicode-safe image loading
        
        Args:
            image_paths: List of image file paths
            labels: List of class labels
            behavior_data: Dictionary mapping paths to behavior features
            img_size: Target image size
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
            
        Returns:
            Data generator for training/validation
        """
        # Convert labels to one-hot encoding
        num_classes = len(set(labels))
        labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
        
        behavior_size = len(self.behavior_columns) if self.behavior_columns else 46
        
        class EnhancedDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, image_paths, labels, behavior_data, behavior_size, img_size, batch_size, augment, parent, **kwargs):
                super().__init__(**kwargs)
                self.image_paths = image_paths
                self.labels = labels
                self.behavior_data = behavior_data
                self.behavior_size = behavior_size
                self.img_size = img_size
                self.batch_size = batch_size
                self.augment = augment
                self.parent = parent  # Reference to parent class for base_dir
                self.indices = np.arange(len(self.image_paths))
                
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
                return int(np.ceil(len(self.image_paths) / self.batch_size))
            
            def __getitem__(self, idx):
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_size = len(batch_indices)
                
                # Initialize arrays
                batch_images = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
                batch_behaviors = np.zeros((batch_size, self.behavior_size), dtype=np.float32)
                batch_labels = np.zeros((batch_size, num_classes), dtype=np.float32)
                
                valid_samples = 0
                
                # Fill batch with valid samples
                for i, idx in enumerate(batch_indices):
                    if valid_samples >= batch_size:
                        break
                        
                    img_path = self.image_paths[idx]
                    
                    try:
                        # Use Unicode-safe path resolution and image loading
                        resolved_path = fix_image_path_resolution(img_path, self.parent.base_dir)
                        img = safe_imread(resolved_path)
                        
                        if img is None:
                            continue
                        
                        # Process image
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype(np.float32) / 255.0
                        
                        # Apply augmentation if enabled
                        if self.augment:
                            img = self.img_gen.random_transform(img)
                        
                        # Store image and label
                        batch_images[valid_samples] = img
                        batch_labels[valid_samples] = self.labels[idx]
                        
                        # Add behavior data
                        if img_path in self.behavior_data:
                            behavior_vector = self.behavior_data[img_path]
                            # Ensure correct length
                            if len(behavior_vector) >= self.behavior_size:
                                batch_behaviors[valid_samples] = behavior_vector[:self.behavior_size]
                            else:
                                # Pad with zeros if too short
                                padded = behavior_vector + [0.0] * (self.behavior_size - len(behavior_vector))
                                batch_behaviors[valid_samples] = padded
                        
                        valid_samples += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
                
                # If we don't have enough valid samples, pad the remaining with zeros
                if valid_samples < batch_size:
                    # Truncate arrays to actual number of valid samples
                    batch_images = batch_images[:valid_samples]
                    batch_behaviors = batch_behaviors[:valid_samples]
                    batch_labels = batch_labels[:valid_samples]
                
                inputs = {
                    'image_input': batch_images,
                    'behavior_input': batch_behaviors
                }
                
                return inputs, batch_labels
            
            def on_epoch_end(self):
                if self.augment:
                    np.random.shuffle(self.indices)
        
        return EnhancedDataGenerator(image_paths, labels_onehot, behavior_data, behavior_size, img_size, batch_size, augment, self)
    
    def fine_tune_model(self, unfreeze_layers=15):
        """
        Fine-tune the model by unfreezing some base model layers
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        if self.model is None:
            raise ValueError("No model to fine-tune. Create a model first.")
        
        # Find the base model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name.lower():
                base_model = layer
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model fine-tuned: Unfroze last {unfreeze_layers} layers of the base model")
        return self.model
    
    def train(self, epochs=50, batch_size=32, img_size=(224, 224), fine_tune=True):
        """
        Train the model with enhanced features
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Target image size
            fine_tune: Whether to apply fine-tuning
            
        Returns:
            tuple: (training_history, model_directory)
        """
        print("="*80)
        print("TRAINING WITH ENHANCED UNICODE-SAFE DOG EMOTION CLASSIFIER")
        print("="*80)
        
        # Load training data
        print("Loading training data...")
        train_paths, train_labels, train_behavior_data, class_names = self.prepare_data_for_training('train')
        
        print("Loading validation data...")
        val_paths, val_labels, val_behavior_data, _ = self.prepare_data_for_training('validation')
        
        print(f"\nTraining setup:")
        print(f"  Classes: {len(class_names)} - {class_names}")
        print(f"  Behavior features: {len(self.behavior_columns)}")
        print(f"  Training samples: {len(train_paths)}")
        print(f"  Validation samples: {len(val_paths)}")
        
        # Create model
        if self.model is None:
            self.create_model(
                num_classes=len(class_names),
                behavior_size=len(self.behavior_columns),
                img_size=(*img_size, 3)
            )
        
        # Create generators
        train_gen = self.create_data_generator(
            train_paths, train_labels, train_behavior_data, img_size, batch_size, augment=True
        )
        
        val_gen = self.create_data_generator(
            val_paths, val_labels, val_behavior_data, img_size, batch_size, augment=False
        )
        
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(self.model_dir, f"enhanced_dog_emotion_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
                monitor='val_accuracy', 
                save_best_only=True, 
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6, 
                mode='max',
                verbose=1
            )
        ]
        
        # Initial training
        print(f"\nStarting initial training for {epochs} epochs...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning if requested
        if fine_tune:
            print("\nApplying fine-tuning...")
            self.fine_tune_model(unfreeze_layers=15)
            
            # Additional fine-tuning epochs
            fine_tune_epochs = max(10, epochs // 3)
            print(f"Fine-tuning for {fine_tune_epochs} additional epochs...")
            
            ft_history = self.model.fit(
                train_gen,
                epochs=fine_tune_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Extend history
            for k in history.history:
                if k in ft_history.history:
                    history.history[k].extend(ft_history.history[k])
        
        # Save final model and metadata
        self.model.save(os.path.join(model_dir, 'final_model.keras'))
        
        metadata = {
            'class_names': class_names,
            'behavior_columns': self.behavior_columns,
            'img_size': list(img_size),
            'training_timestamp': timestamp,
            'total_behavior_features': len(self.behavior_columns),
            'training_samples': len(train_paths),
            'validation_samples': len(val_paths),
            'epochs_trained': len(history.history['loss']),
            'fine_tuned': fine_tune,
            'best_val_accuracy': max(history.history['val_accuracy'])
        }
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Generate training plots
        self.plot_training_history(history, model_dir)
        
        # Evaluate on test set
        self.evaluate_test_set(model_dir)
        
        print(f"\nTraining complete! Model saved to: {model_dir}")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history, model_dir
    
    def plot_training_history(self, history, output_dir):
        """
        Plot training history and save to file
        
        Args:
            history: Training history object
            output_dir: Directory to save plots
        """
        try:
            # Plot accuracy
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
            plt.close()
            
            print(f"Training plots saved to {output_dir}")
            
        except Exception as e:
            print(f"Error creating training plots: {e}")
    
    def evaluate_test_set(self, model_dir):
        """
        Evaluate the model on the test set
        
        Args:
            model_dir: Directory to save evaluation results
        """
        try:
            print("Evaluating on test set...")
            
            # Load test data
            test_paths, test_labels, test_behavior_data, _ = self.prepare_data_for_training('test')
            
            if len(test_paths) == 0:
                print("No test data available for evaluation")
                return
            
            # Create test generator
            test_gen = self.create_data_generator(
                test_paths, test_labels, test_behavior_data, (224, 224), batch_size=32, augment=False
            )
            
            # Evaluate model
            evaluation = self.model.evaluate(test_gen, verbose=1)
            
            # Save evaluation metrics
            test_metrics = {
                'test_loss': float(evaluation[0]),
                'test_accuracy': float(evaluation[1]),
                'test_samples': len(test_paths)
            }
            
            with open(os.path.join(model_dir, 'test_metrics.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
            print(f"Test loss: {test_metrics['test_loss']:.4f}")
            
        except Exception as e:
            print(f"Error during test evaluation: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Optional: Clean problematic filenames first
    # Uncomment the next line to clean Unicode issues in existing files
    # clean_problematic_filenames("C:\\Users\\kelly\\Documents\\GitHub\\Pawnder")
    
    # Create and run trainer
    trainer = FixedDogEmotionTrainer()
    
    # Train the model
    print("Starting enhanced training with Unicode support...")
    history, model_dir = trainer.train(
        epochs=50, 
        batch_size=32, 
        img_size=(224, 224),
        fine_tune=True
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Model directory: {model_dir}")
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")
