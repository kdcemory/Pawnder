# keypoint_enhanced_data_generator.py
# Enhanced data generator that incorporates keypoint data with emotion recognition

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import json
import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("KeypointEnhancedDataGenerator")

def load_keypoint_annotations(keypoint_dir):
    """
    Load keypoint annotations from files
    
    Args:
        keypoint_dir (str): Directory containing keypoint annotation files
        
    Returns:
        dict: Dictionary of keypoints indexed by image ID
    """
    logger.info(f"Loading keypoint annotations from {keypoint_dir}...")
    
    if not os.path.exists(keypoint_dir):
        logger.warning(f"Keypoint directory not found: {keypoint_dir}")
        return {}
    
    # Load all keypoint files
    keypoints = {}
    keypoint_files = glob.glob(os.path.join(keypoint_dir, "*.json"))
    
    for kp_file in tqdm(keypoint_files, desc="Loading keypoint files"):
        try:
            with open(kp_file, 'r') as f:
                keypoint_data = json.load(f)
            
            # Extract the image ID from filename
            filename = os.path.basename(kp_file)
            image_id = os.path.splitext(filename)[0]
            
            # Store keypoints indexed by image ID
            keypoints[image_id] = keypoint_data
            
        except Exception as e:
            logger.warning(f"Error loading keypoint file {kp_file}: {str(e)}")
    
    logger.info(f"Loaded keypoints for {len(keypoints)} images")
    return keypoints

class KeypointEnhancedDataGenerator(tf.keras.utils.Sequence):
    """
    Enhanced data generator for dog emotion recognition with keypoints
    
    This generator extends the functionality of DogEmotionDataGenerator by
    incorporating keypoint data when available.
    """
    
    def __init__(self, annotations_df, img_dir, behavior_cols, emotion_col, 
                 keypoint_dir=None, num_keypoints=17,
                 img_size=(224, 224), batch_size=32, shuffle=True, augment=False,
                 use_bbox=True, unknown_as_none=False, emotion_to_idx=None):
        """
        Initialize the data generator
        
        Args:
            annotations_df (pd.DataFrame): DataFrame with annotations
            img_dir (str): Directory containing the images
            behavior_cols (list): List of behavioral indicator columns
            emotion_col (str): Column name for emotion labels
            keypoint_dir (str): Directory containing keypoint annotations
            num_keypoints (int): Number of keypoints to extract
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data between epochs
            augment (bool): Whether to apply data augmentation
            use_bbox (bool): Whether to use bounding boxes for cropping
            unknown_as_none (bool): Treat "Unknown" emotion as None and reduce confidence
            emotion_to_idx (dict): Optional predefined emotion to index mapping
        """
        self.annotations = annotations_df
        self.img_dir = img_dir
        self.behavior_cols = behavior_cols
        self.emotion_col = emotion_col
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.use_bbox = use_bbox
        self.unknown_as_none = unknown_as_none
        self.num_keypoints = num_keypoints
        
        # Set up keypoint loading if directory is provided
        self.use_keypoints = keypoint_dir is not None
        self.keypoints = {}
        if self.use_keypoints:
            self.keypoints = load_keypoint_annotations(keypoint_dir)
            logger.info(f"Using keypoint data with {len(self.keypoints)} annotations")
        
        # Get unique emotions for one-hot encoding
        if emotion_to_idx is not None:
            # Use provided emotion mapping
            self.emotion_to_idx = emotion_to_idx
            self.emotions = list(emotion_to_idx.keys())
        else:
            # Determine emotions from data
            all_emotions = self.annotations[self.emotion_col].dropna().unique()
            # Filter out "Unknown" if needed
            if self.unknown_as_none:
                self.emotions = [e for e in all_emotions if e.lower() != "unknown"]
            else:
                self.emotions = all_emotions
                
            self.emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotions)}
        
        # Print emotion mapping for debugging
        logger.info(f"Using {len(self.emotions)} emotions in data generator: {self.emotions}")
        
        # Create indices and shuffle if needed
        self.indices = np.arange(len(self.annotations))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Set up augmentation if needed
        if self.augment:
            self.img_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.annotations) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.annotations.iloc[batch_indices]
        
        # Initialize batch arrays
        batch_images = np.zeros((len(batch_data), *self.img_size, 3), dtype=np.float32)
        batch_behaviors = np.zeros((len(batch_data), len(self.behavior_cols)), dtype=np.float32)
        batch_emotions = np.zeros((len(batch_data), len(self.emotions)), dtype=np.float32)
        batch_confidence = np.ones((len(batch_data), 1), dtype=np.float32)  # Default confidence of 1.0
        
        # Initialize keypoints array if using them
        if self.use_keypoints:
            batch_keypoints = np.zeros((len(batch_data), self.num_keypoints, 2), dtype=np.float32)
            batch_keypoint_visibility = np.zeros((len(batch_data), self.num_keypoints), dtype=np.float32)
        
        # Fill batch data
        for i, (_, row) in enumerate(batch_data.iterrows()):
            # Load and preprocess image
            img_path = os.path.join(self.img_dir, row['image_path'])
            try:
                img = self._load_and_preprocess_image(img_path, row)
                batch_images[i] = img
                
                # Extract behavioral indicators
                for j, col in enumerate(self.behavior_cols):
                    batch_behaviors[i, j] = row[col] if col in row and not pd.isna(row[col]) else 0
                
                # One-hot encode emotion
                emotion = row[self.emotion_col]
                if pd.isna(emotion) or (self.unknown_as_none and emotion.lower() == "unknown"):
                    # No clear emotion, use lower confidence
                    batch_confidence[i] = 0.5
                    # Use a balanced distribution of emotions
                    batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
                else:
                    emotion_idx = self.emotion_to_idx.get(emotion, -1)
                    if emotion_idx >= 0:
                        batch_emotions[i, emotion_idx] = 1.0
                    else:
                        # Unknown emotion, use lower confidence
                        batch_confidence[i] = 0.7
                        # Use a balanced distribution
                        batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
                
                # Load keypoints if available
                if self.use_keypoints:
                    # Try different strategies to find keypoints
                    keypoints_found = False
                    
                    # Strategy 1: Direct lookup by image path
                    image_id = os.path.splitext(os.path.basename(row['image_path']))[0]
                    if image_id in self.keypoints:
                        keypoints_found = self._process_keypoints(
                            self.keypoints[image_id], 
                            batch_keypoints, 
                            batch_keypoint_visibility, 
                            i, img.shape[:2]
                        )
                    
                    # Strategy 2: Try with 'stanford_' prefix removed (if present)
                    if not keypoints_found and image_id.startswith('stanford_'):
                        stanford_id = image_id[9:]  # Remove 'stanford_' prefix
                        if stanford_id in self.keypoints:
                            keypoints_found = self._process_keypoints(
                                self.keypoints[stanford_id], 
                                batch_keypoints, 
                                batch_keypoint_visibility, 
                                i, img.shape[:2]
                            )
                    
                    # Strategy 3: Try to find keypoints in original ID if available
                    if not keypoints_found and 'original_id' in row:
                        original_id = os.path.splitext(os.path.basename(row['original_id']))[0]
                        if original_id in self.keypoints:
                            keypoints_found = self._process_keypoints(
                                self.keypoints[original_id], 
                                batch_keypoints, 
                                batch_keypoint_visibility, 
                                i, img.shape[:2]
                            )
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                # If image loading fails, use zeros and lower confidence
                batch_confidence[i] = 0.3
                # Use a balanced distribution for emotions
                batch_emotions[i] = np.ones(len(self.emotions)) / len(self.emotions)
        
        # Return inputs and outputs
        inputs = {'image_input': batch_images, 'behavior_input': batch_behaviors}
        
        # Add keypoints to inputs if using them
        if self.use_keypoints:
            inputs['keypoint_input'] = batch_keypoints
            inputs['keypoint_visibility'] = batch_keypoint_visibility
        
        outputs = {'emotion_output': batch_emotions, 'confidence_output': batch_confidence}
        
        return inputs, outputs
    
    def _process_keypoints(self, keypoint_data, batch_keypoints, batch_visibility, index, img_shape):
        """
        Process and normalize keypoints for a single image
        
        Args:
            keypoint_data (dict): Keypoint data from annotation file
            batch_keypoints (np.array): Batch array for keypoints
            batch_visibility (np.array): Batch array for keypoint visibility
            index (int): Index in the batch
            img_shape (tuple): Image shape (height, width)
            
        Returns:
            bool: Whether keypoints were successfully processed
        """
        try:
            # Extract keypoints depending on format
            if isinstance(keypoint_data, dict) and 'keypoints' in keypoint_data:
                # Format: {"keypoints": [[x1, y1, v1], [x2, y2, v2], ...]}
                kp_array = np.array(keypoint_data['keypoints'])
                
                # Check if we have the right number of keypoints
                if len(kp_array) != self.num_keypoints:
                    logger.warning(f"Expected {self.num_keypoints} keypoints, but got {len(kp_array)}")
                    return False
                
                # Check if we have visibility values
                if kp_array.shape[1] >= 3:
                    # Format with visibility: [x, y, v]
                    coords = kp_array[:, :2]
                    visibility = kp_array[:, 2]
                else:
                    # Format without visibility: [x, y]
                    coords = kp_array
                    visibility = np.ones(len(kp_array))
                
            elif isinstance(keypoint_data, list) and len(keypoint_data) == self.num_keypoints:
                # Format: [[x1, y1, v1], [x2, y2, v2], ...] or [[x1, y1], [x2, y2], ...]
                kp_array = np.array(keypoint_data)
                
                if kp_array.shape[1] >= 3:
                    coords = kp_array[:, :2]
                    visibility = kp_array[:, 2]
                else:
                    coords = kp_array
                    visibility = np.ones(len(kp_array))
                
            else:
                logger.warning(f"Unrecognized keypoint format: {type(keypoint_data)}")
                return False
            
            # Normalize coordinates to [0, 1] range
            h, w = img_shape
            normalized_coords = coords.copy()
            normalized_coords[:, 0] /= w
            normalized_coords[:, 1] /= h
            
            # Clip to ensure values are in [0, 1] range
            normalized_coords = np.clip(normalized_coords, 0, 1)
            
            # Store normalized coordinates and visibility
            batch_keypoints[index] = normalized_coords
            batch_visibility[index] = visibility
            
            return True
            
        except Exception as e:
            logger.warning(f"Error processing keypoints: {e}")
            return False
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch if needed"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_and_preprocess_image(self, img_path, row):
        """Load and preprocess an image"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop to bounding box if available and enabled
        if self.use_bbox and all(col in row.index and not pd.isna(row[col]) 
                               for col in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
            x, y = int(row['bbox_x']), int(row['bbox_y'])
            w, h = int(row['bbox_width']), int(row['bbox_height'])
            
            # Ensure bbox is within image bounds
            img_h, img_w = img.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # Crop if bbox is valid
            if w > 10 and h > 10:  # Ensure minimal size to avoid tiny crops
                img = img[y:y+h, x:x+w]
            else:
                logger.warning(f"Invalid bounding box dimensions: {w}x{h} for {img_path}")
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply augmentation if needed
        if self.augment:
            img = self.img_gen.random_transform(img)
        
        return img