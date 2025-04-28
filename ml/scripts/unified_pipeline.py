# Unified Dog Emotion Recognition Pipeline
# Combines data processing from pawnderdataprocessingpipeline.ipynb with
# advanced model architectures from pawnder-model-from-dog-emotion.ipynb

import os
import sys
import glob
import json
import shutil
import re
import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Flatten, Concatenate, BatchNormalization,
    TimeDistributed, LSTM, Bidirectional
)
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import yaml
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Install required packages if not already installed
try:
    import cv2
except ImportError:
    print("Installing OpenCV...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# =====================================================================
# PART 1: Configuration and Setup
# =====================================================================

# Project paths setup
DEFAULT_CONFIG_PATH = "/content/drive/MyDrive/Colab Notebooks/Pawnder/config.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Load configuration from YAML file or use defaults"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
            return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        
        # Create a minimal default config
        default_config = {
            "data": {
                "base_dir": "/content/drive/MyDrive/Colab Notebooks/Pawnder",
                "raw_data_dir": "Data/Raw",
                "processed_data_dir": "Data/processed",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "model": {
                "image_size": [224, 224, 3],
                "batch_size": 32,
                "learning_rate": 0.001,
                "dropout_rate": 0.5,
                "backbone": "mobilenetv2"
            },
            "training": {
                "checkpoint_dir": "Models/checkpoints",
                "logs_dir": "Models/logs"
            },
            "inference": {
                "confidence_threshold": 0.6,
                "behavior_threshold": 0.5,
                "output_dir": "Data/processed/predictions"
            }
        }
        
        print("Using default configuration")
        return default_config

# Load configuration first to get base_dir
config = load_config()
base_dir = config["data"]["base_dir"]

# Create a helper function to build full paths
def get_path(relative_path):
    """Build a full path from a relative path"""
    return os.path.join(base_dir, relative_path)

# Define directory paths based on config
DATA_DIRS = {
    'base': base_dir,
    'raw': get_path(config["data"]["raw_data_dir"]),
    'processed': get_path(config["data"]["processed_data_dir"]),
    'stanford_original': get_path(os.path.join(config["data"]["raw_data_dir"], "stanford_dog_pose")),
    'personal_dataset': get_path(os.path.join(config["data"]["raw_data_dir"], "personal_dataset")),
    'videos': get_path(config["data"].get("videos_dir", os.path.join(config["data"]["raw_data_dir"], "personal_dataset", "videos"))),
    'personal_images': get_path(config["data"].get("images_dir", os.path.join(config["data"]["raw_data_dir"], "personal_dataset", "images"))),
    'interim': get_path(config["data"].get("interim_dir", "Data/interim")),
    'matrix': get_path(config["data"].get("matrix_dir", "Data/Matrix")),
    'models': get_path(config["data"].get("models_dir", "Models")),
    'checkpoints': get_path(config["training"]["checkpoint_dir"]),
    'logs': get_path(config["training"]["logs_dir"]),
    'predictions': get_path(config["inference"]["output_dir"])
}

# Extract emotion mapping from config
EMOTION_MAPPING = config.get("emotions", {}).get("mapping", {})
if not EMOTION_MAPPING:
    # Default emotion mapping if not in config
    EMOTION_MAPPING = {
        "Happy or Playful": "Happy/Playful",
        "Relaxed": "Relaxed",
        "Submissive": "Submissive/Appeasement",
        "Excited": "Happy/Playful",
        "Drowsy or Bored": "Relaxed",
        "Curious or Confused": "Curiosity/Alertness",
        "Confident or Alert": "Curiosity/Alertness",
        "Jealous": "Stressed",
        "Stressed": "Stressed",
        "Frustrated": "Stressed",
        "Unsure or Uneasy": "Fearful/Anxious",
        "Possessive, Territorial, Dominant": "Fearful/Anxious",
        "Fear or Aggression": "Aggressive/Threatening",
        "Pain": "Stressed"
    }

# Define emotion classes from the EMOTION_MAPPING
CLASS_NAMES = sorted(list(set(EMOTION_MAPPING.values())))

# Safe class names for directories
SAFE_CLASS_NAMES = [emotion.replace("/", "_").replace("\\", "_") for emotion in CLASS_NAMES]
CLASS_MAP = dict(zip(CLASS_NAMES, SAFE_CLASS_NAMES))

def ensure_directories():
    """Ensure all required directories exist"""
    for path in DATA_DIRS.values():
        if isinstance(path, str) and os.path.isabs(path):  # Only create if it's an absolute path
            Path(path).mkdir(parents=True, exist_ok=True)
            
    # Create subdirectories for processed data
    for split in ["all_by_class", "train_by_class", "val_by_class", "test_by_class"]:
        split_dir = os.path.join(DATA_DIRS['processed'], split)
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        
        for safe_name in SAFE_CLASS_NAMES:
            class_dir = os.path.join(split_dir, safe_name)
            Path(class_dir).mkdir(parents=True, exist_ok=True)
            
        # Create unknown class directory
        unknown_dir = os.path.join(split_dir, "unknown")
        Path(unknown_dir).mkdir(parents=True, exist_ok=True)
    
    # Create combined frames directory
    combined_frames_dir = os.path.join(DATA_DIRS['processed'], "all_frames")
    Path(combined_frames_dir).mkdir(parents=True, exist_ok=True)
    
    # Create directories for model outputs
    Path(DATA_DIRS['checkpoints']).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIRS['logs']).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIRS['predictions']).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")

# =====================================================================
# PART 2: Data Processing Pipeline
# =====================================================================

def parse_xml_annotation(xml_file):
    """Parse CVAT XML annotations"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get video name from task source
        video_name = None
        for meta in root.findall('.//meta'):
            for task in meta.findall('.//task'):
                for source in task.findall('.//source'):
                    video_name = source.text
                    if video_name:
                        # Remove extension if present
                        video_name = os.path.splitext(video_name)[0]

        if not video_name:
            video_name = os.path.basename(os.path.dirname(xml_file))

        # Extract annotations
        annotations = {}

        for track in root.findall('.//track'):
            for box in track.findall('.//box'):
                frame_num = int(box.get('frame', 0))
                frame_id = f"frame_{frame_num:06d}"

                # Look for Primary Emotion attribute
                emotion = None
                for attr in box.findall('.//attribute'):
                    name = attr.get('name')
                    if name == "Primary Emotion":
                        emotion = attr.text
                        break

                if emotion:
                    # Map to standardized emotion if needed
                    if emotion in EMOTION_MAPPING:
                        emotion = EMOTION_MAPPING[emotion]

                    annotations[frame_id] = {
                        "emotions": {"primary_emotion": emotion},
                        "video_name": video_name,
                        "frame": frame_num,
                        "original_format": "xml"
                    }

        return annotations, video_name

    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {str(e)}")
        return {}, None

def process_video_folder(video_folder):
    """Process a video folder with images and annotations"""
    folder_name = os.path.basename(video_folder)
    print(f"\nProcessing video folder: {folder_name}")

    # Define output directories
    combined_frames_dir = os.path.join(DATA_DIRS['processed'], "all_frames")
    output_dir = DATA_DIRS['processed']

    # Find any XML file in this folder
    xml_files = glob.glob(os.path.join(video_folder, "*.xml"))
    if not xml_files:
        print(f"  No XML files found in {video_folder}")
        return 0, {}

    # Use the first XML file
    xml_file = xml_files[0]
    print(f"  Using XML file: {os.path.basename(xml_file)}")

    # Parse annotations
    annotations, video_name = parse_xml_annotation(xml_file)
    if not video_name:
        video_name = folder_name

    # Generate video ID
    video_id = ''.join(e for e in video_name if e.isalnum())[:8]

    # Look for images directory
    images_dir = os.path.join(video_folder, "images")
    if not os.path.exists(images_dir):
        print(f"  Images directory not found: {images_dir}")
        return 0, {}

    # Count annotations by emotion
    emotion_counts = defaultdict(int)
    for frame_id, data in annotations.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1

    print(f"  Found {len(annotations)} annotated frames")
    print("  Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {emotion}: {count}")

    # Get all files in the images directory
    all_files = os.listdir(images_dir)
    print(f"  Found {len(all_files)} files in images directory")

    # Create filename mapping for quick lookup
    filename_map = {}
    for filename in all_files:
        match = re.search(r'frame_0*(\d+)', filename.lower())
        if match:
            frame_num = int(match.group(1))
            if frame_num not in filename_map:
                filename_map[frame_num] = []
            filename_map[frame_num].append(filename)

    # Process each file
    processed_frames = {}
    processed_count = 0
    missing_count = 0

    for frame_id, annotation in tqdm(annotations.items(), desc=f"Processing {folder_name}", leave=False):
        # Extract frame number
        match = re.search(r'frame_0*(\d+)', frame_id)
        if not match:
            continue

        frame_num = int(match.group(1))

        # Find matching file
        if frame_num in filename_map and filename_map[frame_num]:
            src_filename = filename_map[frame_num][0]
            src_path = os.path.join(images_dir, src_filename)

            # Get emotion and create safe version
            emotion = annotation["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            # Create new filename with consistent format
            new_filename = f"video_{video_id}_frame_{frame_num:06d}{os.path.splitext(src_filename)[1]}"
            dst_path = os.path.join(combined_frames_dir, new_filename)

            # Copy to combined directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

            # Copy to class directory
            class_dir = os.path.join(output_dir, "all_by_class", safe_emotion)
            class_path = os.path.join(class_dir, new_filename)
            os.makedirs(os.path.dirname(class_path), exist_ok=True)
            shutil.copy2(src_path, class_path)

            # Add to processed frames
            processed_frames[new_filename] = {
                "emotions": {"primary_emotion": emotion},
                "video_name": video_name,
                "video_id": video_id,
                "frame_id": frame_id,
                "original_path": src_path,
                "source": "video_frames"
            }

            processed_count += 1
        else:
            missing_count += 1

    print(f"  Processed {processed_count} frames, {missing_count} frames missing")
    return processed_count, processed_frames

def process_personal_images():
    """Process personal dataset images"""
    print("\nProcessing Personal Dataset Images")
    
    # Define paths
    personal_images_dir = DATA_DIRS['personal_images']
    emotions_file = os.path.join(DATA_DIRS['interim'], "emotions_only.json")
    combined_frames_dir = os.path.join(DATA_DIRS['processed'], "all_frames")
    output_dir = DATA_DIRS['processed']

    # Check if emotions_only.json exists
    if not os.path.exists(emotions_file):
        print(f"  Error: emotions_only.json not found at {emotions_file}")
        return 0, {}

    # Load the emotions file
    try:
        with open(emotions_file, 'r') as f:
            all_annotations = json.load(f)
        print(f"  Loaded {len(all_annotations)} annotations from emotions_only.json")
    except Exception as e:
        print(f"  Error loading emotions file: {str(e)}")
        return 0, {}

    # Check if personal_images_dir exists
    if not os.path.exists(personal_images_dir):
        print(f"  Error: Personal images directory not found at {personal_images_dir}")
        return 0, {}

    # Build a map of all personal images
    personal_images = {}
    image_files = []

    # Look for images in the root and subdirectories
    for root, dirs, files in os.walk(personal_images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
                # Add with basename as key
                personal_images[file] = file_path
                # Also add without extension
                name_without_ext = os.path.splitext(file)[0]
                personal_images[name_without_ext] = file_path

    print(f"  Found {len(image_files)} total personal images")

    # Find personal images in annotations
    processed_frames = {}
    processed_count = 0

    for img_id, annotation in tqdm(all_annotations.items(), desc="Processing personal annotations"):
        # Skip if this doesn't have an emotion
        if "emotions" not in annotation or "primary_emotion" not in annotation["emotions"]:
            continue

        # Get the filename
        filename = os.path.basename(img_id)
        basename = os.path.splitext(filename)[0]

        # Check if we have this image
        image_path = None
        if filename in personal_images:
            image_path = personal_images[filename]
        elif basename in personal_images:
            image_path = personal_images[basename]

        # Skip if it's a Stanford image
        is_stanford = re.match(r'n\d+_\d+', basename) is not None
        if is_stanford:
            continue

        # Skip if it's a video frame (already processed)
        is_video_frame = "frame_" in basename
        if is_video_frame:
            continue

        if image_path:
            # Get emotion and create safe version
            emotion = annotation["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            # Create new filename
            new_filename = f"personal_{basename}{os.path.splitext(image_path)[1]}"
            dst_path = os.path.join(combined_frames_dir, new_filename)

            # Copy to combined directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(image_path, dst_path)

                # Copy to class directory
                class_dir = os.path.join(output_dir, "all_by_class", safe_emotion)
                class_path = os.path.join(class_dir, new_filename)
                os.makedirs(os.path.dirname(class_path), exist_ok=True)
                shutil.copy2(image_path, class_path)

                # Add to processed frames
                processed_frames[new_filename] = {
                    "emotions": {"primary_emotion": emotion},
                    "original_id": img_id,
                    "original_path": image_path,
                    "source": "personal"
                }

                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} personal images")

            except Exception as e:
                print(f"  Error copying {image_path}: {str(e)}")

    print(f"  Processed {processed_count} personal images with emotion annotations")

    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in processed_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1

    print("  Personal dataset emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {emotion}: {count}")

    return processed_count, processed_frames

def process_stanford_dataset():
    """Process Stanford dog dataset images"""
    print("\nProcessing Stanford Dog Dataset")
    
    # Define paths
    stanford_images_dir = os.path.join(DATA_DIRS['stanford_original'], "Images")
    emotions_file = os.path.join(DATA_DIRS['interim'], "emotions_only.json")
    combined_frames_dir = os.path.join(DATA_DIRS['processed'], "all_frames")
    output_dir = DATA_DIRS['processed']

    # Check if emotions_only.json exists
    if not os.path.exists(emotions_file):
        print(f"  Error: emotions_only.json not found at {emotions_file}")
        return 0, {}

    # Load the emotions file
    try:
        with open(emotions_file, 'r') as f:
            all_annotations = json.load(f)
        print(f"  Loaded {len(all_annotations)} annotations from emotions_only.json")
    except Exception as e:
        print(f"  Error loading emotions file: {str(e)}")
        return 0, {}

    # Check if stanford_images_dir exists
    if not os.path.exists(stanford_images_dir):
        print(f"  Error: Stanford images directory not found at {stanford_images_dir}")
        return 0, {}

    # Find all stanford breed directories
    breed_dirs = []
    for item in os.listdir(stanford_images_dir):
        breed_path = os.path.join(stanford_images_dir, item)
        if os.path.isdir(breed_path) and item.startswith('n'):  # Stanford uses n* format for breed directories
            breed_dirs.append(breed_path)

    print(f"  Found {len(breed_dirs)} breed directories")

    # Build a map of all Stanford images
    stanford_images = {}
    for breed_dir in breed_dirs:
        breed_name = os.path.basename(breed_dir)
        image_files = glob.glob(os.path.join(breed_dir, "*.jpg")) + \
                     glob.glob(os.path.join(breed_dir, "*.png"))

        for image_path in image_files:
            image_name = os.path.basename(image_path)
            stanford_images[image_name] = image_path
            # Also add without extension
            name_without_ext = os.path.splitext(image_name)[0]
            stanford_images[name_without_ext] = image_path

    print(f"  Found {len(stanford_images)} total Stanford dog images")

    # Find Stanford images in annotations
    processed_frames = {}
    processed_count = 0

    for img_id, annotation in tqdm(all_annotations.items(), desc="Processing Stanford annotations"):
        # Skip if this doesn't have an emotion
        if "emotions" not in annotation or "primary_emotion" not in annotation["emotions"]:
            continue

        # Check if this is likely a Stanford image
        is_stanford = False
        filename = os.path.basename(img_id)
        basename = os.path.splitext(filename)[0]

        # Stanford images often have format like n02085620_10074.jpg
        if re.match(r'n\d+_\d+', basename) or filename in stanford_images or basename in stanford_images:
            is_stanford = True

        if not is_stanford:
            continue

        # Find the image path
        image_path = None
        if filename in stanford_images:
            image_path = stanford_images[filename]
        elif basename in stanford_images:
            image_path = stanford_images[basename]

        if image_path:
            # Get emotion and create safe version
            emotion = annotation["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            # Create new filename
            new_filename = f"stanford_{basename}{os.path.splitext(image_path)[1]}"
            dst_path = os.path.join(combined_frames_dir, new_filename)

            # Copy to combined directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(image_path, dst_path)

                # Copy to class directory
                class_dir = os.path.join(output_dir, "all_by_class", safe_emotion)
                class_path = os.path.join(class_dir, new_filename)
                os.makedirs(os.path.dirname(class_path), exist_ok=True)
                shutil.copy2(image_path, class_path)

                # Add to processed frames
                processed_frames[new_filename] = {
                    "emotions": {"primary_emotion": emotion},
                    "original_id": img_id,
                    "original_path": image_path,
                    "source": "stanford"
                }

                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} Stanford images")

            except Exception as e:
                print(f"  Error copying {image_path}: {str(e)}")

    print(f"  Processed {processed_count} Stanford images with emotion annotations")

    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in processed_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1

    print("  Stanford emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {emotion}: {count}")

    return processed_count, processed_frames

def create_train_val_test_splits(all_frames):
    """Create train/val/test splits from processed frames"""
    # Get split ratios from config
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]
    
    output_dir = DATA_DIRS['processed']
    
    # Group by emotion
    frames_by_emotion = defaultdict(list)
    for filename, data in all_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        safe_emotion = emotion.replace("/", "_").replace("\\", "_")
        frames_by_emotion[safe_emotion].append((filename, data))

    print("\nEmotion distribution:")
    for emotion, frames in sorted(frames_by_emotion.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {emotion}: {len(frames)} frames")

    # Perform stratified split
    train_files = []
    val_files = []
    test_files = []

    for emotion, files in frames_by_emotion.items():
        random.shuffle(files)
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])

    # Copy files to split directories
    split_counts = defaultdict(lambda: defaultdict(int))
    combined_frames_dir = os.path.join(output_dir, "all_frames")

    for file_list, split_dir, split_name in [
        (train_files, os.path.join(output_dir, "train_by_class"), "train"),
        (val_files, os.path.join(output_dir, "val_by_class"), "validation"),
        (test_files, os.path.join(output_dir, "test_by_class"), "test")
    ]:
        print(f"\nCreating {split_name} split with {len(file_list)} files...")

        for filename, data in tqdm(file_list, desc=f"Processing {split_name} split"):
            emotion = data["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            src_path = os.path.join(combined_frames_dir, filename)
            dst_path = os.path.join(split_dir, safe_emotion, filename)

            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                split_counts[split_name][safe_emotion] += 1

    # Print split statistics
    print("\nFinal Dataset Statistics:")
    for split_name in ["train", "validation", "test"]:
        emotion_counts = split_counts[split_name]
        total = sum(emotion_counts.values())

        print(f"\n{split_name.capitalize()} split:")
        print(f"  Total: {total} images")
        print("  Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percent = count / total * 100 if total > 0 else 0
            print(f"    {emotion}: {count} ({percent:.1f}%)")
            
    # Save split counts to file for future reference
    split_stats = {
        "train": dict(split_counts["train"]),
        "validation": dict(split_counts["validation"]),
        "test": dict(split_counts["test"])
    }
    
    split_stats_path = os.path.join(output_dir, "split_statistics.json")
    with open(split_stats_path, 'w') as f:
        json.dump(split_stats, f, indent=2)
        
    print(f"\nSplit statistics saved to: {split_stats_path}")
    
    return split_counts

def process_data():
    """Run the complete data processing pipeline"""
    print("Starting Pawnder data processing pipeline...")
    
    # Set random seed for reproducibility
    random.seed(42)

    # Make sure directories exist
    ensure_directories()

    # Step 1: Process video folders
    print("\nStep 1: Processing video folders...")
    video_folders = []
    videos_dir = DATA_DIRS['videos']
    
    if os.path.exists(videos_dir):
        for item in os.listdir(videos_dir):
            folder_path = os.path.join(videos_dir, item)
            if os.path.isdir(folder_path):
                video_folders.append(folder_path)

        print(f"Found {len(video_folders)} video folders")

        video_frames = {}
        total_video_frames = 0

        for video_folder in tqdm(video_folders, desc="Processing video folders"):
            count, frames = process_video_folder(video_folder)
            total_video_frames += count
            video_frames.update(frames)

        print(f"\nProcessed {total_video_frames} video frames from {len(video_folders)} folders")
    else:
        print(f"Video directory not found: {videos_dir}")
        video_frames = {}
        total_video_frames = 0

    # Step 2: Process Stanford dataset
    print("\nStep 2: Processing Stanford dataset...")
    stanford_count, stanford_frames = process_stanford_dataset()

    # Step 3: Process personal images
    print("\nStep 3: Processing personal dataset images...")
    personal_count, personal_frames = process_personal_images()

    # Step 4: Combine all annotations
    all_frames = {}
    all_frames.update(video_frames)
    all_frames.update(stanford_frames)
    all_frames.update(personal_frames)

    total_frames = len(all_frames)
    print(f"\nTotal processed frames: {total_frames}")
    print(f"  - {len(video_frames)} video frames")
    print(f"  - {len(stanford_frames)} Stanford images")
    print(f"  - {len(personal_frames)} personal images")

    # Save combined annotations
    processed_dir = DATA_DIRS['processed']
    annotations_file = os.path.join(processed_dir, "combined_annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(all_frames, f, indent=2)

    print(f"Saved annotations to: {annotations_file}")

    # Step 5: Create train/val/test splits
    print("\nStep 5: Creating train/val/test splits...")
    split_counts = create_train_val_test_splits(all_frames)
    
    print("\nDataset preparation complete!")
    return all_frames, split_counts

# =====================================================================
# PART 3: Model Architecture
# =====================================================================

class DogEmotionModel:
    """Class for building dog emotion recognition models"""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        """
        Initialize the model builder

        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config = load_config(config_path)

    def _create_image_branch(self, input_shape, backbone="mobilenetv2"):
        """
        Create the image processing branch of the model

        Args:
            input_shape (tuple): Image input shape (height, width, channels)
            backbone (str): Backbone model for transfer learning

        Returns:
            tuple: (input_tensor, output_tensor)
        """
        input_tensor = Input(shape=input_shape, name="image_input")

        # Select backbone based on configuration
        if backbone.lower() == "mobilenetv2":
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "resnet50":
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "efficientnetb0":
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze early layers for transfer learning
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False

        # Connect input to backbone
        x = base_model(input_tensor)

        # Add custom layers on top of backbone
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"])(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)

        return input_tensor, x

    def _create_behavior_branch(self, behavior_size):
        """
        Create the behavioral indicator branch of the model

        Args:
            behavior_size (int): Number of behavioral indicator features

        Returns:
            tuple: (input_tensor, output_tensor)
        """
        input_tensor = Input(shape=(behavior_size,), name="behavior_input")

        x = Dense(128, activation="relu")(input_tensor)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)

        return input_tensor, x

    def _create_video_branch(self, input_shape, backbone="mobilenetv2", sequence_length=16):
        """
        Create the video processing branch of the model

        Args:
            input_shape (tuple): Frame input shape (height, width, channels)
            backbone (str): Backbone model for transfer learning
            sequence_length (int): Number of frames in the sequence

        Returns:
            tuple: (input_tensor, output_tensor)
        """
        # Input shape is (sequence_length, height, width, channels)
        input_tensor = Input(shape=(sequence_length, *input_shape), name="video_input")

        # Select backbone based on configuration
        if backbone.lower() == "mobilenetv2":
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "resnet50":
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "efficientnetb0":
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze early layers for transfer learning
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False

        # Apply CNN to each frame in the sequence
        frame_features = TimeDistributed(base_model)(input_tensor)
        frame_features = TimeDistributed(GlobalAveragePooling2D())(frame_features)

        # Process the sequence of frame features with LSTM
        x = Bidirectional(LSTM(256, return_sequences=True))(frame_features)
        x = Bidirectional(LSTM(128))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"])(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)

        return input_tensor, x

    def _calculate_behavior_size(self):
        """
        Calculate the number of behavioral indicator features based on matrix data

        Returns:
            int: Number of behavioral features
        """
        # Try to load from a file
        behavior_map_path = os.path.join(DATA_DIRS['matrix'], "primary_behavior_matrix.json")

        if os.path.exists(behavior_map_path):
            try:
                with open(behavior_map_path, 'r') as f:
                    behavior_map = json.load(f)
                    
                # Count total behaviors across all categories
                behavior_count = 0
                for category in behavior_map.get('behavior_categories', []):
                    behavior_count += len(category.get('behaviors', []))
                    
                if behavior_count > 0:
                    return behavior_count
            except:
                pass

        # Default size if file not found or is empty
        return 64  # Assume 64 behavioral indicators by default

    def _create_emotion_head(self, x, num_emotions=None):
        """
        Create the emotion classification head

        Args:
            x: Input tensor
            num_emotions (int): Number of emotion classes

        Returns:
            tf.Tensor: Output tensor
        """
        if num_emotions is None:
            num_emotions = len(CLASS_NAMES)
            
        emotion_output = Dense(num_emotions, activation="softmax", name="emotion_output")(x)
        return emotion_output

    def _create_confidence_head(self, x):
        """
        Create the confidence scoring head

        Args:
            x: Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        confidence_output = Dense(1, activation="sigmoid", name="confidence_output")(x)
        return confidence_output

    def create_model(self, model_type="image", num_emotions=None, behavior_size=None):
        """
        Create the full model architecture

        Args:
            model_type (str): 'image' or 'video'
            num_emotions (int): Number of emotion classes
            behavior_size (int): Number of behavioral indicator features

        Returns:
            tf.keras.Model: Compiled model
        """
        # Set up image size from config
        input_shape = tuple(self.config["model"]["image_size"])
        
        # Default number of emotions if not provided
        if num_emotions is None:
            num_emotions = len(CLASS_NAMES)

        # Calculate behavior size if not provided
        if behavior_size is None:
            behavior_size = self._calculate_behavior_size()

        # Get backbone type from config
        backbone = self.config["model"]["backbone"]

        # Create model branches
        if model_type.lower() == "image":
            image_input, image_features = self._create_image_branch(input_shape, backbone)
            behavior_input, behavior_features = self._create_behavior_branch(behavior_size)

            # Combine features
            combined = Concatenate()([image_features, behavior_features])

        elif model_type.lower() == "video":
            sequence_length = self.config["model"].get("sequence_length", 16)
            video_input, video_features = self._create_video_branch(input_shape, backbone, sequence_length)
            behavior_input, behavior_features = self._create_behavior_branch(behavior_size)

            # Combine features
            combined = Concatenate()([video_features, behavior_features])

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add fusion layers
        x = Dense(256, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 4)(x)

        # Create output heads
        emotion_output = self._create_emotion_head(x, num_emotions)
        confidence_output = self._create_confidence_head(x)

        # Create model with appropriate inputs
        if model_type.lower() == "image":
            model = Model(
                inputs=[image_input, behavior_input],
                outputs=[emotion_output, confidence_output]
            )
        else:  # video
            model = Model(
                inputs=[video_input, behavior_input],
                outputs=[emotion_output, confidence_output]
            )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config["model"]["learning_rate"]),
            loss={
                "emotion_output": "categorical_crossentropy",
                "confidence_output": "binary_crossentropy"
            },
            metrics={
                "emotion_output": ["accuracy", "top_k_categorical_accuracy"],
                "confidence_output": ["accuracy"]
            },
            loss_weights={
                "emotion_output": 1.0,
                "confidence_output": 0.2
            }
        )

        return model

    def create_region_attention_model(self, num_emotions=None, behavior_size=None, num_regions=5):
        """
        Create a model with attention to different regions of dog (eyes, ears, mouth, tail, body)

        Args:
            num_emotions (int): Number of emotion classes
            behavior_size (int): Number of behavioral indicator features
            num_regions (int): Number of body regions to model

        Returns:
            tf.keras.Model: Compiled model with region attention
        """
        # Default number of emotions if not provided
        if num_emotions is None:
            num_emotions = len(CLASS_NAMES)
            
        # Set up image size from config
        input_shape = tuple(self.config["model"]["image_size"])

        # Calculate behavior size if not provided
        if behavior_size is None:
            behavior_size = self._calculate_behavior_size()

        # Get backbone type from config
        backbone = self.config["model"]["backbone"]

        # Main image input
        image_input = Input(shape=input_shape, name="image_input")

        # Region coordinates inputs (each region has [x, y, width, height])
        region_inputs = []
        region_features = []

        # Create a mini-model for the backbone
        if backbone.lower() == "mobilenetv2":
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "resnet50":
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        elif backbone.lower() == "efficientnetb0":
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze early layers for transfer learning
        for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
            layer.trainable = False

        # Process full image
        full_features = base_model(image_input)
        full_features = GlobalAveragePooling2D()(full_features)

        # Process each region
        for i in range(num_regions):
            # Input for region coordinates [x, y, width, height]
            region_input = Input(shape=(4,), name=f"region_{i}_input")
            region_inputs.append(region_input)

            # Extract region features using a Lambda layer with custom cropping function
            from tensorflow.keras.layers import Lambda

            def extract_region(inputs):
                img, coords = inputs
                # Get coordinates
                x, y, w, h = coords[0]

                # Convert to integers
                x = tf.cast(x * tf.cast(tf.shape(img)[2], dtype=tf.float32), dtype=tf.int32)
                y = tf.cast(y * tf.cast(tf.shape(img)[1], dtype=tf.float32), dtype=tf.int32)
                w = tf.cast(w * tf.cast(tf.shape(img)[2], dtype=tf.float32), dtype=tf.int32)
                h = tf.cast(h * tf.cast(tf.shape(img)[1], dtype=tf.float32), dtype=tf.int32)

                # Ensure coordinates are valid
                x = tf.maximum(0, x)
                y = tf.maximum(0, y)
                w = tf.minimum(w, tf.shape(img)[2] - x)
                h = tf.minimum(h, tf.shape(img)[1] - y)

                # Extract region
                region = tf.image.crop_to_bounding_box(img, y, x, h, w)

                # Resize to standard size
                region = tf.image.resize(region, [input_shape[0], input_shape[1]])

                return region

            # Extract and process region
            region = Lambda(extract_region)([image_input, region_input])
            region_feature = base_model(region)
            region_feature = GlobalAveragePooling2D()(region_feature)
            region_features.append(region_feature)

        # Behavioral input
        behavior_input, behavior_features = self._create_behavior_branch(behavior_size)

        # Combine features from full image, regions, and behavior
        all_features = [full_features] + region_features + [behavior_features]
        combined = Concatenate()(all_features)

        # Add fusion layers
        x = Dense(512, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"])(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config["model"]["dropout_rate"] / 2)(x)

        # Create output heads
        emotion_output = self._create_emotion_head(x, num_emotions)
        confidence_output = self._create_confidence_head(x)

        # Create full model
        model = Model(
            inputs=[image_input] + region_inputs + [behavior_input],
            outputs=[emotion_output, confidence_output]
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config["model"]["learning_rate"]),
            loss={
                "emotion_output": "categorical_crossentropy",
                "confidence_output": "binary_crossentropy"
            },
            metrics={
                "emotion_output": ["accuracy", "top_k_categorical_accuracy"],
                "confidence_output": ["accuracy"]
            },
            loss_weights={
                "emotion_output": 1.0,
                "confidence_output": 0.2
            }
        )

        return model

    def create_model_with_uncertainty(self, model_type="image", num_emotions=None, behavior_size=None):
        """
        Create a model with uncertainty estimation using Monte Carlo Dropout

        Args:
            model_type (str): 'image' or 'video'
            num_emotions (int): Number of emotion classes
            behavior_size (int): Number of behavioral indicator features

        Returns:
            tf.keras.Model: Compiled model with uncertainty estimation
        """
        # Default number of emotions if not provided
        if num_emotions is None:
            num_emotions = len(CLASS_NAMES)
            
        # Create base model
        model = self.create_model(model_type, num_emotions, behavior_size)

        # Create a wrapper function for Monte Carlo Dropout inference
        def mc_dropout_predict(inputs, n_samples=10):
            """Perform Monte Carlo Dropout inference"""
            # Enable dropout at inference time
            tf.keras.backend.set_learning_phase(1)

            # Initialize arrays for predictions
            emotion_preds = []
            confidence_preds = []

            # Perform multiple forward passes
            for _ in range(n_samples):
                emotion_pred, confidence_pred = model.predict(inputs)
                emotion_preds.append(emotion_pred)
                confidence_preds.append(confidence_pred)

            # Stack predictions
            emotion_preds = np.stack(emotion_preds, axis=0)
            confidence_preds = np.stack(confidence_preds, axis=0)

            # Calculate mean and standard deviation
            emotion_mean = np.mean(emotion_preds, axis=0)
            emotion_std = np.std(emotion_preds, axis=0)
            confidence_mean = np.mean(confidence_preds, axis=0)
            confidence_std = np.std(confidence_preds, axis=0)

            # Disable dropout for normal inference
            tf.keras.backend.set_learning_phase(0)

            return (emotion_mean, emotion_std), (confidence_mean, confidence_std)

        # Attach the function to the model for easy access
        model.mc_dropout_predict = mc_dropout_predict

        return model

# =====================================================================
# PART 4: Data Generator and Training
# =====================================================================

class DogEmotionDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for dog emotion recognition model"""
    
    def __init__(self, data_dir, split='train', img_size=(224, 224), batch_size=32, 
                 shuffle=True, augment=False, behavior_size=64, use_behaviors=True):
        """
        Initialize the data generator
        
        Args:
            data_dir (str): Base directory containing data
            split (str): 'train', 'val', or 'test'
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data between epochs
            augment (bool): Whether to apply data augmentation
            behavior_size (int): Size of behavior feature vector
            use_behaviors (bool): Whether to use behavior features
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.behavior_size = behavior_size
        self.use_behaviors = use_behaviors
        
        # Set proper split directory name
        if split == 'val':
            split_dir_name = 'val_by_class'
        elif split == 'test':
            split_dir_name = 'test_by_class'
        else:
            split_dir_name = 'train_by_class'
            
        self.split_dir = os.path.join(data_dir, split_dir_name)
        
        # Load image paths and labels
        self.samples = []
        self.emotion_to_idx = {}
        
        # Map emotions to indices
        for idx, emotion in enumerate(sorted(CLASS_NAMES)):
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")
            self.emotion_to_idx[emotion] = idx
            
            # Find all images for this emotion
            emotion_dir = os.path.join(self.split_dir, safe_emotion)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(emotion_dir, img_file)
                        self.samples.append((img_path, emotion))
        
        # Shuffle if needed
        if self.shuffle:
            random.shuffle(self.samples)
            
        print(f"Created {split} generator with {len(self.samples)} samples")
        
        # Set up augmentation if needed
        if self.augment:
            self.img_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
        # Try to load behavioral matrix
        self.behavior_matrix = self._load_behavior_matrix()
    
    def _load_behavior_matrix(self):
        """Load behavioral matrix from JSON file if available"""
        behavior_map_path = os.path.join(DATA_DIRS['matrix'], "primary_behavior_matrix.json")
        if os.path.exists(behavior_map_path):
            try:
                with open(behavior_map_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch_samples = self.samples[start_idx:end_idx]
        batch_size = len(batch_samples)
        
        # Initialize batch arrays
        batch_images = np.zeros((batch_size, self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        batch_emotions = np.zeros((batch_size, len(self.emotion_to_idx)), dtype=np.float32)
        batch_behaviors = np.zeros((batch_size, self.behavior_size), dtype=np.float32)
        batch_confidence = np.ones((batch_size, 1), dtype=np.float32)  # Default confidence of 1.0
        
        # Fill batch data
        for i, (img_path, emotion) in enumerate(batch_samples):
            try:
                # Load and preprocess image
                img = self._load_and_preprocess_image(img_path)
                batch_images[i] = img
                
                # One-hot encode emotion
                emotion_idx = self.emotion_to_idx.get(emotion, 0)
                batch_emotions[i, emotion_idx] = 1.0
                
                # Generate behavioral features (placeholder for now)
                # In a real implementation, you would extract these from annotations or matrix
                if self.use_behaviors and self.behavior_matrix:
                    # Try to extract some behavioral features based on emotion
                    behavior_states = self.behavior_matrix.get('behavioral_states', [])
                    state_id = None
                    
                    # Find the state ID matching the emotion
                    for state in behavior_states:
                        if state['name'] == emotion:
                            state_id = state['id']
                            break
                    
                    # Generate behavioral features from the matrix if we found the state
                    if state_id:
                        behavior_idx = 0
                        for category in self.behavior_matrix.get('behavior_categories', []):
                            for behavior in category.get('behaviors', []):
                                if behavior_idx < self.behavior_size:
                                    mapping = behavior.get('state_mapping', {})
                                    if state_id in mapping:
                                        batch_behaviors[i, behavior_idx] = mapping[state_id]
                                    behavior_idx += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Lower confidence if there was an error
                batch_confidence[i] = 0.5
        
        # Return inputs and outputs
        inputs = {
            'image_input': batch_images,
            'behavior_input': batch_behaviors
        }
        outputs = {
            'emotion_output': batch_emotions,
            'confidence_output': batch_confidence
        }
        
        return inputs, outputs
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch if needed"""
        if self.shuffle:
            random.shuffle(self.samples)
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess an image"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply augmentation if needed
        if self.augment:
            img = self.img_gen.random_transform(img)
        
        return img

def train_model(model, data_dir, config, model_name="dog_emotion"):
    """
    Train the model
    
    Args:
        model (tf.keras.Model): Model to train
        data_dir (str): Directory containing the data
        config (dict): Configuration dictionary
        model_name (str): Name for saved model
        
    Returns:
        tuple: (trained_model, history)
    """
    # Set up parameters
    batch_size = config['model']['batch_size']
    img_size = tuple(config['model']['image_size'][:2])  # Height, width only
    epochs = config['training'].get('epochs', 50)
    
    # Create data generators
    train_gen = DogEmotionDataGenerator(
        data_dir=data_dir,
        split='train',
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )
    
    val_gen = DogEmotionDataGenerator(
        data_dir=data_dir,
        split='val',
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(base_dir, config['training']['checkpoint_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logs_dir = os.path.join(base_dir, config['training']['logs_dir'])
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamp for unique folder names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}")
    log_dir = os.path.join(logs_dir, f"{model_name}_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_emotion_output_accuracy',
            patience=config['model']['early_stopping_patience'],
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_emotion_output_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_emotion_output_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Save regular checkpoints
        ModelCheckpoint(
            os.path.join(model_dir, 'model_epoch_{epoch:02d}.h5'),
            save_freq='epoch',
            verbose=0
        ),
        
        # TensorBoard logs
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Train the model
    print(f"Starting training for {model_name}...")
    print(f"Using {len(train_gen)} training batches and {len(val_gen)} validation batches")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.json")
    
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(val) for val in values]
        
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {history_path}")
    
    # Plot training history
    plot_training_history(history, model_name, model_dir)
    
    return model, history

def plot_training_history(history, model_name, save_dir):
    """Plot and save training history"""
    # Create the figure
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['emotion_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_emotion_output_accuracy'], label='Validation Accuracy')
    plt.title('Emotion Recognition Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['emotion_output_loss'], label='Training Loss')
    plt.plot(history.history['val_emotion_output_loss'], label='Validation Loss')
    plt.title('Emotion Recognition Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    history_plot_path = os.path.join(save_dir, f