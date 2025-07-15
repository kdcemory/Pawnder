import os
import glob
import shutil
import re
import random
from lxml import etree as ET
import json
import argparse
import pandas as pd
import numpy as np
import logging
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PawnderProcessor")

def convert_json_to_csv(json_path, csv_path):
    """Convert annotations from JSON to CSV format"""
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Check if annotations is empty
        if not annotations:
            print(f"Warning: No annotations found in {json_path}. Creating empty CSV with headers.")
            # Create an empty DataFrame with default columns
            df = pd.DataFrame(columns=['image_path', 'emotional_state', 'source'])
            df.to_csv(csv_path, index=False)
            return True
            
        # Convert to DataFrame
        rows = []
        for filename, data in annotations.items():
            # Extract base data
            row = {
                'image_path': filename,
                'emotional_state': data.get('emotions', {}).get('primary_emotion', 'Unknown')
            }
            
            # Extract other fields if present
            for key, value in data.items():
                if key not in ['emotions']:
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            row[f"{key}_{subkey}"] = subvalue
                    else:
                        row[key] = value
            
            # Extract behavioral indicators if present
            if 'behavioral_indicators' in data:
                for behavior, value in data['behavioral_indicators'].items():
                    row[f"behavior_{behavior}"] = value
            
            rows.append(row)
        
        # Check if rows is empty
        if not rows:
            print(f"Warning: No valid rows extracted from {json_path}. Creating empty CSV with headers.")
            df = pd.DataFrame(columns=['image_path', 'emotional_state', 'source'])
        else:
            # Create DataFrame from rows
            df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Converted {json_path} to {csv_path} with {len(df)} entries")
        return True
    
    except Exception as e:
        print(f"Error converting JSON to CSV: {e}")
        # Create empty CSV with default headers as fallback
        try:
            df = pd.DataFrame(columns=['image_path', 'emotional_state', 'source'])
            df.to_csv(csv_path, index=False)
            print(f"Created empty CSV file with default headers at {csv_path}")
            return True
        except Exception as e2:
            print(f"Failed to create empty CSV file: {e2}")
            return False

def setup_argparse():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description="Process dog emotion datasets")
    parser.add_argument("--force_reprocess", action="store_true", 
                        help="Force reprocessing of all datasets from scratch")
    parser.add_argument("--only_splits", action="store_true", 
                        help="Only recreate train/val/test splits from existing processed data")
    parser.add_argument("--stanford_only", action="store_true",
                        help="Process only the Stanford dataset (skip videos and personal images)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip processing and only perform analysis on existing data")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations during analysis")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with extra logging")
    return parser.parse_args()

def setup_paths_and_directories(base_dir="c:\\Users\\kelly\\Documents\\GitHub\\Pawnder"):
    """Setup paths and create output directories - EXACTLY like original process_dataset.py"""
    
    print(f"Using base directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Define key paths - exactly from original process_dataset.py
    # Check for different data directory structures
    data_dir_options = [
        os.path.join(base_dir, "Data"),
        os.path.join(base_dir, "Data", "raw"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "raw")
    ]
    
    data_root = None
    for data_option in data_dir_options:
        if os.path.exists(data_option):
            data_root = data_option
            print(f"Found data directory: {data_root}")
            break
    
    if data_root is None:
        # Create default data structure
        data_root = os.path.join(base_dir, "Data")
        print(f"Creating new data directory: {data_root}")
        os.makedirs(data_root, exist_ok=True)
    
    # Define paths - CORRECTED FOR YOUR ACTUAL STRUCTURE
    dog_emotion_analyzer_dir = os.path.join(base_dir, "Dog Emotion Analyzer")  # From original
    videos_dir = os.path.join(data_root, "raw", "Videos")  # YOUR ACTUAL PATH
    personal_images_dir = os.path.join(data_root, "raw", "personal_dataset")
    stanford_dir = os.path.join(data_root, "raw", "stanford_dog_pose")
    stanford_images_dir = os.path.join(stanford_dir, "Images")
    stanford_annotations_dir = os.path.join(data_root, "raw", "stanford_annotations")
    matrix_dir = os.path.join(data_root, "Data", "Matrix")
    processed_dir = os.path.join(data_root, "processed")
    combined_frames_dir = os.path.join(data_root, "processed", "all_frames")

    # Look for emotions file in multiple locations including instances_*.json
    emotions_file_options = [
        os.path.join(data_root, "interim", "emotions_only.json"),
        os.path.join(data_root, "processed", "emotions_only.json"),
        os.path.join(data_root, "emotions_only.json"),
        os.path.join(base_dir, "emotions_only.json")
    ]
    
    # Add search for instances_*.json files
    instances_patterns = [
        os.path.join(data_root, "**", "instances_*.json"),
        os.path.join(base_dir, "**", "instances_*.json")
    ]
    
    for pattern in instances_patterns:
        matches = glob.glob(pattern, recursive=True)
        emotions_file_options.extend(matches)
    
    emotions_file = None
    for emotions_option in emotions_file_options:
        if os.path.exists(emotions_option):
            emotions_file = emotions_option
            print(f"Found emotions file: {emotions_file}")
            break

    # Output directories
    output_dir = os.path.join(data_root, "processed")
    combined_frames_dir = os.path.join(output_dir, "all_frames")
    cache_dir = os.path.join(output_dir, "cache")
    
    # Create output directories
    os.makedirs(combined_frames_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Cache files
    video_cache = os.path.join(cache_dir, "video_annotations.json")
    stanford_cache = os.path.join(cache_dir, "stanford_annotations.json")
    personal_cache = os.path.join(cache_dir, "personal_annotations.json")
    combined_annotations_file = os.path.join(output_dir, "combined_annotations.json")
    
    # Check if files/directories actually exist and provide alternatives if needed
    print("\nChecking paths...")
    
    # Check videos directory
    if not os.path.exists(videos_dir):
        alt_videos = [
            os.path.join(data_root, "Raw", "Videos"),  # YOUR ACTUAL PATH - prioritize this
            os.path.join(data_root, "Raw", "personal_dataset", "videos"),
            os.path.join(data_root, "Raw", "videos"),
            os.path.join(data_root, "videos")
        ]
        for alt in alt_videos:
            if os.path.exists(alt):
                videos_dir = alt
                print(f"Using alternative videos directory: {videos_dir}")
                break
    
    # Check personal images directory  
    if not os.path.exists(personal_images_dir):
        alt_personal = [
            os.path.join(data_root, "Raw", "personal"),
            os.path.join(data_root, "personal_images"),
            os.path.join(data_root, "personal_dataset"),
            os.path.join(data_root, "personal_dataset", "images"),
        ]
        for alt in alt_personal:
            if os.path.exists(alt):
                personal_images_dir = alt
                print(f"Using alternative personal images directory: {personal_images_dir}")
                break
    
    # Check emotions file
    if emotions_file is None or not os.path.exists(emotions_file):
        alt_emotions = [
            os.path.join(data_root, "processed", "emotions_only.json"),
            os.path.join(data_root, "emotions_only.json"),
            os.path.join(base_dir, "emotions_only.json"),
            os.path.join(data_root, "processed", "combined_annotations.json"),
            os.path.join(data_root, "combined_annotations.json"),
            os.path.join(data_root, "interim", "emotions_only.json"),
            os.path.join(data_root, "interim", "combined_annotations.json"),
            os.path.join(data_root, "interim", "fixed_emotions.json")
        ]
        
        # Add search for instances_*.json files
        instances_patterns = [
            os.path.join(data_root, "**", "instances_*.json"),
            os.path.join(base_dir, "**", "instances_*.json")
        ]
        
        for pattern in instances_patterns:
            matches = glob.glob(pattern, recursive=True)
            alt_emotions.extend(matches)
        
        for alt in alt_emotions:
            if os.path.exists(alt):
                emotions_file = alt
                print(f"Using alternative emotions file: {emotions_file}")
                break
    
    # Print summary
    print("\n" + "="*50)
    print("PATHS SUMMARY")
    print("="*50)
    print(f"Base directory: {base_dir}")
    print(f"Data root: {data_root}")
    print(f"Videos directory: {videos_dir} {'✓' if os.path.exists(videos_dir) else '✗'}")
    print(f"Personal images: {personal_images_dir} {'✓' if os.path.exists(personal_images_dir) else '✗'}")
    print(f"Stanford directory: {stanford_dir} {'✓' if os.path.exists(stanford_dir) else '✗'}")
    print(f"Stanford images: {stanford_images_dir} {'✓' if os.path.exists(stanford_images_dir) else '✗'}")
    print(f"Stanford annotations: {stanford_annotations_dir} {'✓' if os.path.exists(stanford_annotations_dir) else '✗'}")
    print(f"Emotions file: {emotions_file} {'✓' if emotions_file and os.path.exists(emotions_file) else '✗'}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    # Return all paths as a dictionary - matching original structure
    return {
        "base_dir": base_dir,
        "dog_emotion_analyzer_dir": dog_emotion_analyzer_dir,
        "videos_dir": videos_dir,
        "personal_images_dir": personal_images_dir,
        "stanford_dir": stanford_dir,
        "stanford_images_dir": stanford_images_dir,
        "stanford_annotations_dir": stanford_annotations_dir,
        "emotions_file": emotions_file,
        "output_dir": output_dir,
        "processed_dir": processed_dir,
        "combined_frames_dir": combined_frames_dir,
        "cache_dir": cache_dir,
        "video_cache": video_cache,
        "stanford_cache": stanford_cache,
        "personal_cache": personal_cache,
        "combined_annotations_file": combined_annotations_file,
        "matrix_dir": matrix_dir,
        "combined_annotations_csv": os.path.join(output_dir, "combined_annotations.csv")
    }

def setup_emotion_mapping():
    """Setup emotion mapping and class names - from original"""
    # Emotion mapping from old to new categories
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
        "Possessive, Territorial, Dominant": "Aggressive/Threatening",
        "Fear or Aggression": "Fearful/Anxious",
        "Pain": "Stressed"
    }

    # Define emotion classes
    CLASS_NAMES = ["Happy/Playful", "Relaxed", "Submissive/Appeasement",
                   "Curiosity/Alertness", "Stressed", "Fearful/Anxious",
                   "Aggressive/Threatening"]

    # Safe class names for directories
    SAFE_CLASS_NAMES = [emotion.replace("/", "_").replace("\\", "_") for emotion in CLASS_NAMES]
    CLASS_MAP = dict(zip(CLASS_NAMES, SAFE_CLASS_NAMES))
    
    return {
        "EMOTION_MAPPING": EMOTION_MAPPING,
        "CLASS_NAMES": CLASS_NAMES,
        "SAFE_CLASS_NAMES": SAFE_CLASS_NAMES,
        "CLASS_MAP": CLASS_MAP
    }

def load_behavior_matrix(matrix_path=None):
    """
    Load the behavior matrix from JSON file, converting format if needed
    
    Args:
        matrix_path: Optional path to matrix file
        
    Returns:
        dict: Behavior matrix data in consistent format
    """
    try:
        # Try to load from specified path or search for matrix files
        if matrix_path and os.path.exists(matrix_path):
            matrix_file = matrix_path
        else:
            # Search for behavior matrix files
            possible_paths = [
                'use_behavior_matrix.json',
                'behavior_matrix.json',
                'updated_behavior_matrix.json',
                'primary_behavior_matrix.json',
                'data/use_behavior_matrix.json',
                'data/behavior_matrix.json',
                '../use_behavior_matrix.json',
                '../behavior_matrix.json',
                'config/behavior_matrix.json'
            ]
            
            matrix_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    matrix_file = path
                    break
            
            if not matrix_file:
                logger.warning("Behavior matrix file not found")
                return None
        
        with open(matrix_file, 'r') as f:
            raw_matrix = json.load(f)
        
        logger.info(f"Loaded behavior matrix from {matrix_file}")
        
        # Convert to consistent format
        matrix = convert_behavior_matrix_format(raw_matrix)
        
        return matrix
        
    except Exception as e:
        logger.error(f"Error loading behavior matrix: {str(e)}")
        return None

def convert_behavior_matrix_format(raw_matrix):
    """
    Convert behavior matrix to consistent format
    
    Args:
        raw_matrix: Raw matrix data from JSON
        
    Returns:
        dict: Standardized matrix format
    """
    # Handle different input formats
    if "behavioral indicators" in raw_matrix:
        # Format from use_behavior_matrix.json
        emotions_key = "peimary emotions" if "peimary emotions" in raw_matrix else "primary emotions"
        if emotions_key not in raw_matrix:
            emotions_key = "emotions"
        
        emotions = raw_matrix.get(emotions_key, [
            "Happy/Playful", "Relaxed", "Submissive/Appeasement",
            "Curiosity/Alertness", "Stressed", "Fearful/Anxious",
            "Aggressive/Threatening"
        ])
        
        behaviors = raw_matrix["behavioral indicators"]
        
    elif "behavioral_states" in raw_matrix and "behavior_categories" in raw_matrix:
        # Format from primary_behavior_matrix.json
        states = raw_matrix["behavioral_states"]
        categories = raw_matrix["behavior_categories"]
        
        # Create emotion names list
        emotions = [state["name"] for state in states]
        
        # Create behaviors dictionary
        behaviors = {}
        
        # Process each category and its behaviors
        for category in categories:
            for behavior in category.get("behaviors", []):
                behavior_id = behavior["id"]
                
                # Create mapping for each state
                state_mapping = {}
                for state in states:
                    state_id = state["id"]
                    state_name = state["name"]
                    
                    # Check if this behavior has a mapping for this state
                    if "state_mapping" in behavior and state_id in behavior["state_mapping"]:
                        value = behavior["state_mapping"][state_id]
                        state_mapping[state_name] = value
                    else:
                        state_mapping[state_name] = 0
                
                # Add to behaviors dictionary
                behaviors[behavior_id] = state_mapping
    
    elif "emotions" in raw_matrix and "behaviors" in raw_matrix:
        # Already in standard format
        emotions = raw_matrix["emotions"]
        behaviors = raw_matrix["behaviors"]
    
    else:
        logger.warning("Unknown behavior matrix format, using default")
        emotions = [
            "Happy/Playful", "Relaxed", "Submissive/Appeasement",
            "Curiosity/Alertness", "Stressed", "Fearful/Anxious",
            "Aggressive/Threatening"
        ]
        behaviors = {}
    
    # Create standardized matrix
    matrix = {
        "emotions": emotions,
        "behaviors": behaviors
    }
    
    logger.info(f"Converted matrix with {len(behaviors)} behaviors and {len(emotions)} emotions")
    
    return matrix

def normalize_behavior_name(behavior_name):
    """
    Normalize behavior name to match annotation format
    
    Args:
        behavior_name: Original behavior name
        
    Returns:
        str: Normalized behavior name
    """
    # Convert to lowercase and replace special characters
    normalized = behavior_name.lower()
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('/', '_')
    normalized = normalized.replace('-', '_')
    normalized = normalized.replace('(', '').replace(')', '')
    normalized = normalized.replace('&', 'and')
    
    return normalized

def ensure_split_directories(paths):
    """Ensure all split directories exist"""
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(paths["processed_dir"], split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'annotations'), exist_ok=True)

def create_class_directories(paths, emotion_config):
    """Create emotion class directories for all splits - from original"""
    # Create output directories for each split
    for split in ["all_by_class", "train_by_class", "val_by_class", "test_by_class"]:
        split_dir = os.path.join(paths["processed_dir"], split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create directories for each emotion class
        for safe_name in emotion_config["SAFE_CLASS_NAMES"]:
            os.makedirs(os.path.join(split_dir, safe_name), exist_ok=True)
        
        # Create unknown directory
        os.makedirs(os.path.join(split_dir, "unknown"), exist_ok=True)

def find_all_annotation_files(personal_images_dir):
    """Find all annotation files in the personal dataset directory"""
    logger.info(f"Searching for annotation files in {personal_images_dir}")
    
    annotation_files = []
    
    # Search patterns for annotation files
    search_patterns = [
        "*.xml",
        "*.json",
        "annotations.xml",
        "annotations.json"
    ]
    
    # Search in the main directory and subdirectories
    search_dirs = [
        personal_images_dir,
        os.path.join(personal_images_dir, "annotations"),
        os.path.join(personal_images_dir, "annotation"),
        os.path.join(personal_images_dir, "labels"),
        os.path.join(personal_images_dir, "cvat"),
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        logger.info(f"Searching in directory: {search_dir}")
        
        for pattern in search_patterns:
            files = glob.glob(os.path.join(search_dir, pattern))
            annotation_files.extend(files)
            
            # Also search recursively
            recursive_files = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
            annotation_files.extend(recursive_files)
    
    # Remove duplicates
    annotation_files = list(set(annotation_files))
    
    logger.info(f"Found {len(annotation_files)} annotation files:")
    for file in annotation_files:
        logger.info(f"  - {file}")
    
    return annotation_files

def find_all_stanford_annotation_files(stanford_annotations_dir, stanford_dir):
    """Find all Stanford annotation files"""
    logger.info(f"Searching for Stanford annotation files")
    
    annotation_files = []
    
    # Search patterns for Stanford annotation files
    search_patterns = [
        "*.json",
        "instances_*.json",
        "stanford_*.json",
        "annotations.json"
    ]
    
    # Search directories
    search_dirs = [
        stanford_annotations_dir,
        stanford_dir,
        os.path.join(stanford_dir, "annotations"),
        os.path.join(stanford_dir, "Annotation"),
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        logger.info(f"Searching Stanford annotations in directory: {search_dir}")
        
        for pattern in search_patterns:
            files = glob.glob(os.path.join(search_dir, pattern))
            annotation_files.extend(files)
            
            # Also search recursively
            recursive_files = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
            annotation_files.extend(recursive_files)
    
    # Remove duplicates
    annotation_files = list(set(annotation_files))
    
    logger.info(f"Found {len(annotation_files)} Stanford annotation files:")
    for file in annotation_files:
        logger.info(f"  - {file}")
    
    return annotation_files

def parse_personal_xml_enhanced(xml_file, emotion_config, personal_images_dir):
    """Enhanced parsing of personal dataset XML file with better duplicate handling"""
    logger.info(f"Parsing personal XML file: {xml_file}")
    
    # First, get list of actual images in personal dataset
    actual_images = set()
    
    # Search for images in multiple subdirectories
    image_search_dirs = [
        personal_images_dir,
        os.path.join(personal_images_dir, "images"),
        os.path.join(personal_images_dir, "img"),
        os.path.join(personal_images_dir, "pics"),
    ]
    
    for search_dir in image_search_dirs:
        if os.path.exists(search_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
                for img_file in glob.glob(os.path.join(search_dir, ext)):
                    actual_images.add(os.path.basename(img_file))
                # Also search recursively
                for img_file in glob.glob(os.path.join(search_dir, "**", ext), recursive=True):
                    actual_images.add(os.path.basename(img_file))
    
    logger.info(f"Found {len(actual_images)} actual images in personal dataset")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotations = {}
        processed_count = 0
        skipped_no_image = 0
        skipped_no_emotion = 0
        duplicates_found = 0
        
        # Process all images in the XML
        images = root.findall('.//image')
        logger.info(f"Found {len(images)} images in XML")
        
        for image in tqdm(images, desc="Processing personal images"):
            image_id = image.get('id', '0')
            image_name = image.get('name', f'image_{image_id}')
            base_image_name = os.path.basename(image_name)
            
            # FILTER 1: Only process images that actually exist in personal dataset
            if base_image_name not in actual_images:
                # Check if it might be in a subdirectory path or with different case
                found_match = False
                for actual_img in actual_images:
                    if actual_img.lower() == base_image_name.lower() or base_image_name in actual_img:
                        base_image_name = actual_img
                        found_match = True
                        break
                
                if not found_match:
                    skipped_no_image += 1
                    continue
            
            # Process all boxes in this image
            boxes = image.findall('.//box')
            
            for box_idx, box in enumerate(boxes):
                # Extract all attributes from this box
                attributes = box.findall('.//attribute')
                
                # Check for primary emotion FIRST
                primary_emotion = None
                behavioral_indicators = {}
                
                for attr in attributes:
                    name = attr.get('name', '').strip()
                    value = attr.text.strip() if attr.text else ''
                    
                    if not name or not value:
                        continue
                    
                    # Check for primary emotion
                    if name == "Primary Emotion" or ("primary" in name.lower() and "emotion" in name.lower()):
                        # Apply emotion mapping
                        if value in emotion_config["EMOTION_MAPPING"]:
                            value = emotion_config["EMOTION_MAPPING"][value]
                        primary_emotion = value
                    else:
                        # Store as behavioral indicator - normalize the name to match matrix
                        clean_name = normalize_behavior_name(name)
                        behavioral_indicators[clean_name] = value
                
                # FILTER 2: Only keep annotations with primary emotions
                if not primary_emotion:
                    skipped_no_emotion += 1
                    continue
                
                # Create unique frame ID (check for duplicates)
                base_frame_id = f"personal_{base_image_name}_{box_idx}"
                frame_id = base_frame_id
                counter = 0
                
                # Handle duplicates by adding a counter
                while frame_id in annotations:
                    counter += 1
                    frame_id = f"{base_frame_id}_dup_{counter}"
                    duplicates_found += 1
                
                annotation = {
                    "image_name": base_image_name,
                    "image_id": image_id,
                    "box_index": box_idx,
                    "source": "personal",
                    "emotions": {"primary_emotion": primary_emotion},
                    "behavioral_indicators": behavioral_indicators
                }
                
                annotations[frame_id] = annotation
                processed_count += 1
        
        logger.info(f"Personal dataset processing results:")
        logger.info(f"  - Processed annotations: {processed_count}")
        logger.info(f"  - Skipped (no image file): {skipped_no_image}")
        logger.info(f"  - Skipped (no emotion): {skipped_no_emotion}")
        logger.info(f"  - Duplicates handled: {duplicates_found}")
        
        return annotations
        
    except Exception as e:
        logger.error(f"Error parsing personal XML: {e}")
        return {}

def combine_personal_annotations(personal_images_dir, emotion_config):
    """Combine all personal annotation files and remove duplicates"""
    logger.info("Combining all personal annotation files")
    
    annotation_files = find_all_annotation_files(personal_images_dir)
    
    if not annotation_files:
        logger.warning("No personal annotation files found")
        return {}
    
    combined_annotations = {}
    file_stats = {}
    
    for annotation_file in annotation_files:
        logger.info(f"Processing annotation file: {annotation_file}")
        
        try:
            if annotation_file.endswith('.xml'):
                # Parse XML file
                file_annotations = parse_personal_xml_enhanced(annotation_file, emotion_config, personal_images_dir)
            elif annotation_file.endswith('.json'):
                # Parse JSON file
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                # Convert JSON format to our standard format
                file_annotations = {}
                
                # Handle different JSON formats
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict) and "emotions" in value:
                            file_annotations[key] = value
                        elif isinstance(value, dict) and "primary_emotion" in value:
                            # Convert to standard format
                            file_annotations[key] = {
                                "emotions": {"primary_emotion": value["primary_emotion"]},
                                "source": "personal"
                            }
            else:
                logger.warning(f"Unsupported annotation file format: {annotation_file}")
                continue
            
            file_stats[annotation_file] = len(file_annotations)
            
            # Merge with combined annotations, handling duplicates
            duplicates_in_file = 0
            for key, annotation in file_annotations.items():
                if key in combined_annotations:
                    # Handle duplicate by adding file suffix
                    original_key = key
                    counter = 1
                    while key in combined_annotations:
                        key = f"{original_key}_file{counter}"
                        counter += 1
                    duplicates_in_file += 1
                
                combined_annotations[key] = annotation
            
            logger.info(f"  - Added {len(file_annotations)} annotations")
            if duplicates_in_file > 0:
                logger.info(f"  - Handled {duplicates_in_file} duplicates")
                
        except Exception as e:
            logger.error(f"Error processing {annotation_file}: {e}")
    
    logger.info(f"Combined personal annotations summary:")
    logger.info(f"  - Total files processed: {len(annotation_files)}")
    logger.info(f"  - Total annotations: {len(combined_annotations)}")
    logger.info(f"  - File breakdown:")
    for file, count in file_stats.items():
        logger.info(f"    - {os.path.basename(file)}: {count} annotations")
    
    return combined_annotations

def combine_stanford_annotations(stanford_annotations_dir, stanford_dir, emotion_config):
    """Combine all Stanford annotation files and remove duplicates"""
    logger.info("Combining all Stanford annotation files")
    
    annotation_files = find_all_stanford_annotation_files(stanford_annotations_dir, stanford_dir)
    
    if not annotation_files:
        logger.warning("No Stanford annotation files found")
        return {}
    
    combined_annotations = {}
    file_stats = {}
    
    for annotation_file in annotation_files:
        logger.info(f"Processing Stanford annotation file: {annotation_file}")
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            file_annotations = {}
            
            # Handle COCO format
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                # Create mapping from image_id to filename
                image_map = {}
                for image in data["images"]:
                    image_map[image["id"]] = image["file_name"]
                
                # Process annotations
                for ann in data["annotations"]:
                    if "image_id" not in ann or ann["image_id"] not in image_map:
                        continue
                        
                    image_filename = image_map[ann["image_id"]]
                    
                    # Look for emotion in attributes
                    emotion = None
                    if "attributes" in ann and ann["attributes"]:
                        for attr_name, attr_value in ann["attributes"].items():
                            if "emotion" in attr_name.lower() or "primary" in attr_name.lower():
                                emotion = attr_value
                                break
                    
                    # Also check for category_id mapping
                    if not emotion and "category_id" in ann and "categories" in data:
                        for cat in data["categories"]:
                            if cat["id"] == ann["category_id"]:
                                if "emotion" in cat.get("name", "").lower():
                                    emotion = cat["name"]
                                    break
                    
                    # FILTER: Only keep annotations with emotions
                    if emotion:
                        # Apply emotion mapping
                        if emotion in emotion_config["EMOTION_MAPPING"]:
                            emotion = emotion_config["EMOTION_MAPPING"][emotion]
                        
                        file_annotations[image_filename] = {
                            "emotions": {"primary_emotion": emotion},
                            "image_id": ann["image_id"],
                            "annotation_id": ann.get("id", 0),
                            "source": "stanford"
                        }
            
            # Handle direct emotion mapping format
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and "emotions" in value:
                        # Check if has primary emotion
                        if "primary_emotion" in value["emotions"]:
                            file_annotations[key] = value
                    elif isinstance(value, str):
                        # Direct emotion mapping
                        emotion = value
                        if emotion in emotion_config["EMOTION_MAPPING"]:
                            emotion = emotion_config["EMOTION_MAPPING"][emotion]
                        file_annotations[key] = {
                            "emotions": {"primary_emotion": emotion},
                            "source": "stanford"
                        }
            
            file_stats[annotation_file] = len(file_annotations)
            
            # Merge with combined annotations, handling duplicates
            duplicates_in_file = 0
            for key, annotation in file_annotations.items():
                if key in combined_annotations:
                    # Handle duplicate by adding file suffix
                    original_key = key
                    counter = 1
                    while key in combined_annotations:
                        key = f"{original_key}_file{counter}"
                        counter += 1
                    duplicates_in_file += 1
                
                combined_annotations[key] = annotation
            
            logger.info(f"  - Added {len(file_annotations)} Stanford annotations")
            if duplicates_in_file > 0:
                logger.info(f"  - Handled {duplicates_in_file} duplicates")
                
        except Exception as e:
            logger.error(f"Error processing Stanford file {annotation_file}: {e}")
    
    logger.info(f"Combined Stanford annotations summary:")
    logger.info(f"  - Total files processed: {len(annotation_files)}")
    logger.info(f"  - Total annotations: {len(combined_annotations)}")
    logger.info(f"  - File breakdown:")
    for file, count in file_stats.items():
        logger.info(f"    - {os.path.basename(file)}: {count} annotations")
    
    return combined_annotations

# Keep all the original video processing functions unchanged
def parse_xml_annotation(xml_file, emotion_mapping=None):
    """Parse CVAT XML annotations - from original process_dataset.py - UNCHANGED"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get video/task name from source
        source_name = os.path.basename(os.path.dirname(xml_file))
        for meta in root.findall('.//meta'):
            for task in meta.findall('.//task'):
                for source in task.findall('.//source'):
                    source_name = source.text
                    if source_name:
                        # Remove extension if present
                        source_name = os.path.splitext(source_name)[0]

        if not source_name:
            source_name = os.path.basename(os.path.dirname(xml_file))
            
        # Extract annotations
        annotations = {}
        
        # First try to process track-based format (for videos)
        tracks = root.findall('.//track')
        if tracks:
            print(f"Processing track-based annotations for {source_name}")
            for track in tracks:
                for box in track.findall('.//box'):
                    frame_num = int(box.get('frame', 0))
                    frame_id = f"frame_{frame_num:06d}"
                    
                    # Create annotation entry
                    annotation = {
                        "video_name": source_name,
                        "frame": frame_num,
                        "original_format": "xml",
                        "source": "video_frames",
                        "emotions": {},
                        "behavioral_indicators": {}
                    }
                    
                    # Extract all attributes
                    for attr in box.findall('.//attribute'):
                        name = attr.get('name')
                        value = attr.text
                        
                        if name == "Primary Emotion":
                            # Map emotion if mapping is provided
                            if emotion_mapping and value in emotion_mapping:
                                value = emotion_mapping[value]
                            annotation["emotions"]["primary_emotion"] = value
                        else:
                            # Treat as behavioral indicator - normalize the name
                            clean_name = normalize_behavior_name(name)
                            annotation["behavioral_indicators"][clean_name] = value
                    
                    # Only add if we have a primary emotion
                    if "primary_emotion" in annotation["emotions"]:
                        annotations[frame_id] = annotation
        
        # If no tracks found or no annotations extracted, try image-based format
        if not tracks or not annotations:
            images = root.findall('.//image')
            if images:
                print(f"Processing image-based annotations for {source_name}")
                for image in images:
                    image_id = image.get('id', '0')
                    image_name = image.get('name', f'image_{image_id}')
                    
                    # Process all boxes in this image
                    boxes = image.findall('.//box')
                    if not boxes:
                        # Also check for boxes as direct children
                        boxes = image.findall('box')
                    
                    for box in boxes:
                        # Use image ID or name to create a frame ID
                        frame_id = f"frame_{image_id}"
                        
                        # Create annotation entry
                        annotation = {
                            "image_name": image_name,
                            "video_name": source_name,
                            "frame": int(image_id) if image_id.isdigit() else 0,
                            "original_format": "xml",
                            "source": "image_frames",
                            "emotions": {},
                            "behavioral_indicators": {}
                        }
                        
                        # Extract all attributes
                        attributes = box.findall('.//attribute')
                        if not attributes:
                            # Also check for attributes as direct children
                            attributes = box.findall('attribute')
                            
                        for attr in attributes:
                            name = attr.get('name')
                            value = attr.text
                            
                            if name == "Primary Emotion":
                                # Map emotion if mapping is provided
                                if emotion_mapping and value in emotion_mapping:
                                    value = emotion_mapping[value]
                                annotation["emotions"]["primary_emotion"] = value
                            else:
                                # Process as behavioral indicator - normalize the name
                                clean_name = normalize_behavior_name(name)
                                annotation["behavioral_indicators"][clean_name] = value
                        
                        # Only add if we have a primary emotion
                        if "primary_emotion" in annotation["emotions"]:
                            annotations[frame_id] = annotation
        
        print(f"Extracted {len(annotations)} annotations from {xml_file}")
        return annotations, source_name

    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {str(e)}")
        return {}, None

def process_video_folder(video_folder, paths, emotion_config):
    """Process a video folder with images and annotations - UNCHANGED"""
    folder_name = os.path.basename(video_folder)
    print(f"\nProcessing video folder: {folder_name}")

    # Find annotation file
    annotation_files = glob.glob(os.path.join(video_folder, "*.xml"))
    if not annotation_files:
        print(f"  No XML annotation file found in {video_folder}")
        return 0, {}

    # Use the first XML file found
    xml_file = annotation_files[0]
    annotations, video_name = parse_xml_annotation(xml_file, emotion_config["EMOTION_MAPPING"])

    if not video_name:
        video_name = folder_name

    # Create a short unique identifier for this video
    video_id = ''.join(e for e in video_name if e.isalnum())[:8]

    # Find image directory
    image_dir = os.path.join(video_folder, "images")
    if not os.path.exists(image_dir):
        print(f"  No images directory found in {video_folder}")
        return 0, {}

    # Count frames by emotion
    emotion_counts = defaultdict(int)
    for frame_id, annotation in annotations.items():
        if "emotions" in annotation and "primary_emotion" in annotation["emotions"]:
            emotion = annotation["emotions"]["primary_emotion"]
            emotion_counts[emotion] += 1

    print(f"  Found {len(annotations)} annotated frames")
    print("  Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {emotion}: {count}")

    # Get all files in the images directory
    all_files = os.listdir(image_dir)
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

        # Try all possible frame formats
        formats = [
            f"frame_{frame_num:06d}.png",
            f"frame_{frame_num:05d}.png",
            f"frame_{frame_num:04d}.png",
            f"frame_{frame_num:03d}.png",
            f"frame_{frame_num}.png",
            f"frame_{frame_num:06d}.jpg",
            f"frame_{frame_num:05d}.jpg",
            f"frame_{frame_num:04d}.jpg",
            f"frame_{frame_num:03d}.jpg",
            f"frame_{frame_num}.jpg",
            f"frame_{frame_num:06d}.PNG",
            f"frame_{frame_num:05d}.PNG",
            f"frame_{frame_num:04d}.PNG",
            f"frame_{frame_num:03d}.PNG",
            f"frame_{frame_num}.PNG"
        ]

        # Find matching file
        src_path = None
        
        # First check if we have in our frame number map
        if frame_num in filename_map and filename_map[frame_num]:
            src_filename = filename_map[frame_num][0]
            src_path = os.path.join(image_dir, src_filename)
        else:
            # Try direct formats
            for format in formats:
                format_path = os.path.join(image_dir, format)
                if os.path.exists(format_path):
                    src_path = format_path
                    break

        if src_path and os.path.exists(src_path):
            # Get emotion and create safe version
            emotion = annotation["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            # Create new filename with consistent format
            new_filename = f"video_{video_id}_{frame_id}.png"
            dst_path = os.path.join(paths["combined_frames_dir"], new_filename)

            # Copy to combined directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dst_path)

                # Copy to class directory
                class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
                class_path = os.path.join(class_dir, new_filename)
                os.makedirs(os.path.dirname(class_path), exist_ok=True)
                shutil.copy2(src_path, class_path)

                # Add to processed frames - ADD VIDEO IDENTIFIER
                processed_frames[new_filename] = {
                    "emotions": {"primary_emotion": emotion},
                    "video_name": video_name,
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "original_path": src_path,
                    "source": "video_frames",
                    "video_source": video_name,  # Add this for splitting logic
                    "behavioral_indicators": annotation.get("behavioral_indicators", {})
                }

                processed_count += 1
            except Exception as e:
                print(f"  Error copying {src_path}: {str(e)}")
                missing_count += 1
        else:
            missing_count += 1

    print(f"  Processed {processed_count} frames, {missing_count} frames missing")
    return processed_count, processed_frames

def process_video_folders(paths, emotion_config, force_reprocess):
    """Process all video folders - UNCHANGED"""
    video_cache = paths["video_cache"]
    
    # Check if we've already processed videos and should use cache
    if os.path.exists(video_cache) and not force_reprocess:
        with open(video_cache, 'r') as f:
            video_frames = json.load(f)
        
        # Calculate the total frames from the loaded data
        total_video_frames = 0
        if isinstance(video_frames, dict):
            total_video_frames = len(video_frames)
        elif isinstance(video_frames, list):
            total_video_frames = len(video_frames)
        
        print(f"Loaded {total_video_frames} cached video frames from {video_cache}")
        return video_frames, total_video_frames
    
    # Process videos from scratch
    print("Processing video folders from scratch...")
    if not os.path.exists(paths["videos_dir"]):
        print(f"Videos directory not found at {paths['videos_dir']}")
        return {}, 0
    
    # Find all video folders
    video_folders = []
    for item in os.listdir(paths["videos_dir"]):
        folder_path = os.path.join(paths["videos_dir"], item)
        if os.path.isdir(folder_path):
            video_folders.append(folder_path)

    print(f"Found {len(video_folders)} video folders")

    # Process each video folder
    video_frames = {}
    total_video_frames = 0

    for video_folder in tqdm(video_folders, desc="Processing video folders"):
        count, frames = process_video_folder(video_folder, paths, emotion_config)
        total_video_frames += count
        video_frames.update(frames)

    print(f"\nProcessed {total_video_frames} video frames from {len(video_folders)} folders")
    
    # Cache the processed frames
    with open(video_cache, 'w') as f:
        json.dump(video_frames, f, indent=2)
    print(f"Cached video frames to {video_cache}")
    
    return video_frames, total_video_frames

def process_personal_annotations_enhanced(paths, emotion_config, force_reprocess=False, debug=False):
    """Enhanced processing of personal annotations with file combination and duplicate removal"""
    personal_cache = paths["personal_cache"]
    
    # Check cache first
    if os.path.exists(personal_cache) and not force_reprocess:
        logger.info(f"Loading cached personal annotations from {personal_cache}")
        try:
            with open(personal_cache, 'r') as f:
                personal_annotations = json.load(f)
            logger.info(f"Loaded {len(personal_annotations)} cached personal annotations")
            return personal_annotations, len(personal_annotations)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}, processing from scratch")
    
    logger.info("Processing personal dataset from scratch with enhanced combination")
    
    # Combine all annotation files
    combined_annotations = combine_personal_annotations(paths["personal_images_dir"], emotion_config)
    
    if not combined_annotations:
        logger.warning("No valid personal annotations found")
        return {}, 0
    
    # Get list of actual images for validation
    actual_images = set()
    image_search_dirs = [
        paths["personal_images_dir"],
        os.path.join(paths["personal_images_dir"], "images"),
        os.path.join(paths["personal_images_dir"], "img"),
    ]
    
    for search_dir in image_search_dirs:
        if os.path.exists(search_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
                for img_file in glob.glob(os.path.join(search_dir, ext)):
                    actual_images.add(os.path.basename(img_file))
                for img_file in glob.glob(os.path.join(search_dir, "**", ext), recursive=True):
                    actual_images.add(os.path.basename(img_file))
    
    # Process and copy images to output directories
    processed_annotations = {}
    processed_count = 0
    missing_images = 0
    
    for annotation_id, annotation_data in tqdm(combined_annotations.items(), desc="Processing personal images"):
        # Skip if no emotion
        if "emotions" not in annotation_data or "primary_emotion" not in annotation_data["emotions"]:
            continue
        
        # Get image name
        image_name = None
        if "image_name" in annotation_data:
            image_name = annotation_data["image_name"]
        else:
            # Try to extract from annotation_id
            if "_" in annotation_id:
                parts = annotation_id.split("_")
                for part in parts:
                    if "." in part and any(ext in part.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        image_name = part
                        break
        
        if not image_name:
            continue
        
        # Check if image exists
        if image_name not in actual_images:
            missing_images += 1
            continue
        
        # Find the actual image path
        image_path = None
        for search_dir in image_search_dirs:
            potential_path = os.path.join(search_dir, image_name)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
            # Also search recursively
            for img_file in glob.glob(os.path.join(search_dir, "**", image_name), recursive=True):
                image_path = img_file
                break
            if image_path:
                break
        
        if not image_path:
            missing_images += 1
            continue
        
        # Copy image to output directories
        emotion = annotation_data["emotions"]["primary_emotion"]
        safe_emotion = emotion.replace("/", "_").replace("\\", "_")
        
        # Create new filename
        new_filename = f"personal_{processed_count:06d}_{image_name}"
        dst_path = os.path.join(paths["combined_frames_dir"], new_filename)
        
        try:
            # Copy to combined directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(image_path, dst_path)
            
            # Copy to class directory
            class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
            class_path = os.path.join(class_dir, new_filename)
            os.makedirs(os.path.dirname(class_path), exist_ok=True)
            shutil.copy2(image_path, class_path)
            
            # Add to processed annotations
            processed_annotations[new_filename] = {
                "emotions": {"primary_emotion": emotion},
                "original_id": annotation_id,
                "original_path": image_path,
                "source": "personal"
            }
            
            # Copy behavioral indicators if present
            if "behavioral_indicators" in annotation_data:
                processed_annotations[new_filename]["behavioral_indicators"] = annotation_data["behavioral_indicators"]
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error copying {image_path}: {e}")
    
    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in processed_annotations.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1
    
    logger.info(f"Personal annotations processing results:")
    logger.info(f"  - Total combined annotations: {len(combined_annotations)}")
    logger.info(f"  - Successfully processed: {processed_count}")
    logger.info(f"  - Missing images: {missing_images}")
    logger.info(f"  - Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    - {emotion}: {count}")
    
    # Cache results
    if processed_annotations:
        os.makedirs(os.path.dirname(personal_cache), exist_ok=True)
        with open(personal_cache, 'w') as f:
            json.dump(processed_annotations, f, indent=2)
        logger.info(f"Cached personal annotations to {personal_cache}")
    
    return processed_annotations, len(processed_annotations)

def process_stanford_dataset_enhanced(paths, emotion_config, force_reprocess, debug=False):
    """Enhanced processing of Stanford dataset with file combination and duplicate removal"""
    stanford_cache = paths["stanford_cache"]
    
    # Check cache first
    if os.path.exists(stanford_cache) and not force_reprocess:
        logger.info(f"Loading cached Stanford annotations from {stanford_cache}")
        try:
            with open(stanford_cache, 'r') as f:
                stanford_annotations = json.load(f)
            logger.info(f"Loaded {len(stanford_annotations)} cached Stanford annotations")
            return stanford_annotations, len(stanford_annotations)
        except Exception as e:
            logger.warning(f"Error loading cache: {e}, processing from scratch")
    
    logger.info("Processing Stanford dataset from scratch with enhanced combination")
    
    # Combine all Stanford annotation files
    combined_annotations = combine_stanford_annotations(
        paths["stanford_annotations_dir"], 
        paths["stanford_dir"], 
        emotion_config
    )
    
    if not combined_annotations:
        logger.warning("No valid Stanford annotations found")
        return {}, 0
    
    # Build Stanford images map
    if not os.path.exists(paths["stanford_images_dir"]):
        logger.warning(f"Stanford images directory not found: {paths['stanford_images_dir']}")
        return {}, 0
    
    stanford_images = {}
    breed_dirs = [d for d in os.listdir(paths["stanford_images_dir"]) 
                  if os.path.isdir(os.path.join(paths["stanford_images_dir"], d))]
    
    logger.info(f"Found {len(breed_dirs)} Stanford breed directories")
    
    for breed_dir in tqdm(breed_dirs, desc="Scanning Stanford images"):
        breed_path = os.path.join(paths["stanford_images_dir"], breed_dir)
        image_files = glob.glob(os.path.join(breed_path, "*.jpg"))
        
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            stanford_images[image_name] = image_path
            
            # Also index by base name without extension
            base_name = os.path.splitext(image_name)[0]
            stanford_images[base_name] = image_path
    
    logger.info(f"Found {len(stanford_images)} Stanford images")
    
    # Match annotations to images and copy them
    processed_annotations = {}
    processed_count = 0
    not_found_count = 0
    
    for img_key, annotation_data in tqdm(combined_annotations.items(), desc="Processing Stanford annotations"):
        # Skip if no emotion
        if "emotions" not in annotation_data or "primary_emotion" not in annotation_data["emotions"]:
            continue
        
        # Find corresponding image file with detailed matching
        image_path = None
        
        # Try multiple matching strategies
        matching_strategies = [
            img_key,  # Direct match
            os.path.basename(img_key),  # Just filename
            os.path.splitext(img_key)[0],  # Without extension
            os.path.splitext(os.path.basename(img_key))[0],  # Basename without extension
        ]
        
        for strategy in matching_strategies:
            if strategy in stanford_images:
                image_path = stanford_images[strategy]
                break
        
        if not image_path:
            # Try partial matching for similar names
            for stanford_name, stanford_path in stanford_images.items():
                if img_key in stanford_name or stanford_name in img_key:
                    image_path = stanford_path
                    break
        
        if image_path and os.path.exists(image_path):
            emotion = annotation_data["emotions"]["primary_emotion"]
            
            # Create new filename
            new_filename = f"stanford_{processed_count:06d}_{os.path.basename(image_path)}"
            
            # Copy to combined directory
            dst_path = os.path.join(paths["combined_frames_dir"], new_filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                shutil.copy2(image_path, dst_path)
                
                # Copy to class directory
                safe_emotion = emotion.replace("/", "_").replace("\\", "_")
                class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
                class_path = os.path.join(class_dir, new_filename)
                os.makedirs(os.path.dirname(class_path), exist_ok=True)
                shutil.copy2(image_path, class_path)
                
                # Add to processed annotations
                processed_annotations[new_filename] = {
                    "emotions": {"primary_emotion": emotion},
                    "original_key": img_key,
                    "original_path": image_path,
                    "source": "stanford"
                }
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error copying {image_path}: {e}")
        else:
            not_found_count += 1
            if debug:
                logger.debug(f"Could not find image for: {img_key}")
    
    logger.info(f"Stanford processing results:")
    logger.info(f"  - Total combined annotations: {len(combined_annotations)}")
    logger.info(f"  - Images found and processed: {processed_count}")
    logger.info(f"  - Images not found: {not_found_count}")
    
    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in processed_annotations.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1
    
    logger.info(f"  - Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    - {emotion}: {count}")
    
    # Cache results
    if processed_annotations:
        os.makedirs(os.path.dirname(stanford_cache), exist_ok=True)
        with open(stanford_cache, 'w') as f:
            json.dump(processed_annotations, f, indent=2)
        logger.info(f"Cached Stanford annotations to {stanford_cache}")
    
    return processed_annotations, len(processed_annotations)

def generate_emotions_only_json(all_annotations, paths):
    """
    Generate emotions_only.json from the combined annotations
    
    Args:
        all_annotations: Dictionary of all processed annotations
        paths: Dictionary containing file paths
    """
    logger.info("Generating emotions_only.json file")
    
    emotions_only = {}
    
    # Process each entry in the combined annotations
    for key, data in all_annotations.items():
        # Create a simplified entry for emotions_only
        emotion_entry = {}
        
        # Extract primary emotion
        if 'emotions' in data and 'primary_emotion' in data['emotions']:
            emotion_entry['emotions'] = {
                'primary_emotion': data['emotions']['primary_emotion']
            }
        elif 'primary_emotion' in data:
            # Handle case where primary_emotion might be at top level
            emotion_entry['emotions'] = {
                'primary_emotion': data['primary_emotion']
            }
        else:
            # Skip entries without emotion data
            continue
        
        # Add source information
        if 'source' in data:
            emotion_entry['source'] = data['source']
        else:
            # Try to determine source from other fields
            if 'video_name' in data or 'video_id' in data:
                emotion_entry['source'] = 'video_frames'
            elif 'stanford' in key.lower():
                emotion_entry['source'] = 'stanford'
            elif 'personal' in key.lower():
                emotion_entry['source'] = 'personal'
            else:
                emotion_entry['source'] = 'unknown'
        
        # Add video information if available
        if 'video_name' in data:
            emotion_entry['video_name'] = data['video_name']
        if 'video_id' in data:
            emotion_entry['video_id'] = data['video_id']
        if 'video_source' in data:
            emotion_entry['video_source'] = data['video_source']
        
        # Add behavioral indicators if present
        if 'behavioral_indicators' in data and data['behavioral_indicators']:
            emotion_entry['behavioral_indicators'] = data['behavioral_indicators']
        
        # Add frame information if available
        if 'frame_id' in data:
            emotion_entry['frame_id'] = data['frame_id']
        if 'frame' in data:
            emotion_entry['frame'] = data['frame']
        
        # Add original path information
        if 'original_path' in data:
            emotion_entry['original_path'] = data['original_path']
        
        emotions_only[key] = emotion_entry
    
    # Save the emotions_only.json in multiple locations
    emotions_files = [
        os.path.join(paths["output_dir"], "emotions_only.json"),
        os.path.join(paths["base_dir"], "Data", "interim", "emotions_only.json"),
        os.path.join(paths["base_dir"], "emotions_only.json")
    ]
    
    for emotions_file in emotions_files:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(emotions_file), exist_ok=True)
            
            with open(emotions_file, 'w') as f:
                json.dump(emotions_only, f, indent=2)
            logger.info(f"Saved emotions_only.json to {emotions_file}")
        except Exception as e:
            logger.warning(f"Could not save emotions_only.json to {emotions_file}: {e}")
    
    # Print statistics
    emotion_counts = defaultdict(int)
    source_counts = defaultdict(int)
    
    for entry in emotions_only.values():
        # Count emotions
        if 'emotions' in entry and 'primary_emotion' in entry['emotions']:
            emotion = entry['emotions']['primary_emotion']
            emotion_counts[emotion] += 1
        
        # Count sources
        if 'source' in entry:
            source = entry['source']
            source_counts[source] += 1
    
    logger.info(f"Generated emotions_only.json with {len(emotions_only)} entries")
    
    logger.info("Emotion distribution in emotions_only.json:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percent = count / len(emotions_only) * 100 if emotions_only else 0
        logger.info(f"  - {emotion}: {count} ({percent:.1f}%)")
    
    logger.info("Source distribution in emotions_only.json:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        percent = count / len(emotions_only) * 100 if emotions_only else 0
        logger.info(f"  - {source}: {count} ({percent:.1f}%)")
    
    return emotions_only

def combine_and_save_annotations(video_annotations, personal_annotations, stanford_annotations, paths):
    """Combine all annotations and save to file"""
    all_annotations = {}
    all_annotations.update(video_annotations)
    all_annotations.update(personal_annotations)
    all_annotations.update(stanford_annotations)
    
    logger.info(f"Combined annotations summary:")
    logger.info(f"  - Total: {len(all_annotations)} annotations")
    logger.info(f"  - Video: {len(video_annotations)} annotations")
    logger.info(f"  - Personal: {len(personal_annotations)} annotations")
    logger.info(f"  - Stanford: {len(stanford_annotations)} annotations")
    
    # Count by emotion
    emotions = defaultdict(int)
    for _, data in all_annotations.items():
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            emotion = data["emotions"]["primary_emotion"]
            emotions[emotion] += 1
    
    logger.info(f"Emotion distribution:")
    for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        percent = count / len(all_annotations) * 100 if all_annotations else 0
        logger.info(f"  - {emotion}: {count} ({percent:.1f}%)")
    
    # Save combined annotations
    combined_file = paths["combined_annotations_file"]
    with open(combined_file, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    logger.info(f"Saved combined annotations to {combined_file}")
    
    # Generate emotions_only.json
    generate_emotions_only_json(all_annotations, paths)
    
    # Convert to DataFrame and save as CSV
    rows = []
    for key, data in all_annotations.items():
        row = {'image_path': key}
        
        # Add emotion
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            row['primary_emotion'] = data["emotions"]["primary_emotion"]
        
        # Add source
        if "source" in data:
            row['source'] = data["source"]
        
        # Add behavior indicators with normalized names
        if "behavioral_indicators" in data:
            for behavior, value in data["behavioral_indicators"].items():
                # Normalize behavior name to match matrix format
                normalized_behavior = normalize_behavior_name(behavior)
                row[f"behavior_{normalized_behavior}"] = value
        
        # Add other fields
        for field, value in data.items():
            if field not in ["emotions", "behavioral_indicators"] and not isinstance(value, (dict, list)):
                row[field] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = paths["combined_annotations_csv"]
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved combined annotations CSV to {csv_file}")
    
    return all_annotations

def create_train_val_test_splits_with_video_constraint(all_annotations, paths, emotion_config):
    """Create train/validation/test splits ensuring video frames stay together"""
    logger.info("Creating train/validation/test splits with video frame constraint")
    
    # Group annotations by source and video
    video_groups = defaultdict(list)  # For video frames from same video
    individual_items = []  # For non-video items
    
    for key, data in all_annotations.items():
        if "emotions" not in data or "primary_emotion" not in data["emotions"]:
            continue
            
        emotion = data["emotions"]["primary_emotion"]
        # Map to standardized emotions if needed
        if emotion in emotion_config["EMOTION_MAPPING"]:
            emotion = emotion_config["EMOTION_MAPPING"][emotion]
        
        # Check if this is from a video
        if data.get("source") == "video_frames" and "video_source" in data:
            # Group by video source
            video_source = data["video_source"]
            video_groups[video_source].append((key, data, emotion))
        else:
            # Individual items (personal, stanford)
            individual_items.append((key, data, emotion))
    
    logger.info(f"Grouping results:")
    logger.info(f"  - Video groups: {len(video_groups)} videos")
    logger.info(f"  - Individual items: {len(individual_items)}")
    
    # Print video group sizes
    for video_name, items in video_groups.items():
        logger.info(f"    - Video '{video_name}': {len(items)} frames")
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Collect all items for splitting
    train_items = []
    val_items = []
    test_items = []
    
    # 1. Split video groups as units (entire videos go to one split)
    video_names = list(video_groups.keys())
    random.shuffle(video_names)
    
    video_train_count = int(len(video_names) * train_ratio)
    video_val_count = int(len(video_names) * val_ratio)
    
    train_videos = video_names[:video_train_count]
    val_videos = video_names[video_train_count:video_train_count + video_val_count]
    test_videos = video_names[video_train_count + video_val_count:]
    
    # Add video frames to respective splits
    for video_name in train_videos:
        train_items.extend(video_groups[video_name])
    
    for video_name in val_videos:
        val_items.extend(video_groups[video_name])
    
    for video_name in test_videos:
        test_items.extend(video_groups[video_name])
    
    logger.info(f"Video split assignment:")
    logger.info(f"  - Train videos: {len(train_videos)} ({sum(len(video_groups[v]) for v in train_videos)} frames)")
    logger.info(f"  - Validation videos: {len(val_videos)} ({sum(len(video_groups[v]) for v in val_videos)} frames)")
    logger.info(f"  - Test videos: {len(test_videos)} ({sum(len(video_groups[v]) for v in test_videos)} frames)")
    
    # 2. Split individual items by emotion (stratified)
    by_emotion = defaultdict(list)
    for key, data, emotion in individual_items:
        by_emotion[emotion].append((key, data, emotion))
    
    # Stratified split for individual items
    for emotion, items in by_emotion.items():
        random.shuffle(items)
        total = len(items)
        
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_items.extend(items[:train_count])
        val_items.extend(items[train_count:train_count+val_count])
        test_items.extend(items[train_count+val_count:])
    
    # Print final split statistics
    logger.info(f"Final split statistics:")
    logger.info(f"  - Train: {len(train_items)} items")
    logger.info(f"  - Validation: {len(val_items)} items")
    logger.info(f"  - Test: {len(test_items)} items")
    
    # Convert splits to dictionaries
    train_annotations = {key: data for key, data, emotion in train_items}
    val_annotations = {key: data for key, data, emotion in val_items}
    test_annotations = {key: data for key, data, emotion in test_items}
    
    # Save split annotations
    splits = [
        ("train", train_annotations),
        ("validation", val_annotations),
        ("test", test_annotations)
    ]
    
    for split_name, split_data in splits:
        # Create directory
        split_dir = os.path.join(paths["processed_dir"], split_name)
        annotations_dir = os.path.join(split_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(annotations_dir, "annotations.json")
        with open(json_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Convert to DataFrame and save CSV
        rows = []
        for key, data in split_data.items():
            row = {'image_path': key}
            
            # Add emotion
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                row['primary_emotion'] = data["emotions"]["primary_emotion"]
            
            # Add source
            if "source" in data:
                row['source'] = data["source"]
            
            # Add video information if present
            if "video_source" in data:
                row['video_source'] = data["video_source"]
            
            # Add behavior indicators with normalized names
            if "behavioral_indicators" in data:
                for behavior, value in data["behavioral_indicators"].items():
                    normalized_behavior = normalize_behavior_name(behavior)
                    row[f"behavior_{normalized_behavior}"] = value
            
            # Add other fields
            for field, value in data.items():
                if field not in ["emotions", "behavioral_indicators"] and not isinstance(value, (dict, list)):
                    row[field] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(split_dir, "annotations.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved {split_name} annotations: {len(split_data)} items")
        
        # Count by emotion for this split
        emotions = defaultdict(int)
        sources = defaultdict(int)
        for _, data in split_data.items():
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                emotions[emotion] += 1
            if "source" in data:
                source = data["source"]
                sources[source] += 1
        
        logger.info(f"{split_name.capitalize()} emotion distribution:")
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            percent = count / len(split_data) * 100 if split_data else 0
            logger.info(f"  - {emotion}: {count} ({percent:.1f}%)")
        
        logger.info(f"{split_name.capitalize()} source distribution:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percent = count / len(split_data) * 100 if split_data else 0
            logger.info(f"  - {source}: {count} ({percent:.1f}%)")
    
    return {
        "train": train_annotations,
        "validation": val_annotations,
        "test": test_annotations
    }

def analyze_behavior_patterns(all_annotations):
    """Analyze behavior patterns in the annotations"""
    logger.info("Analyzing behavior patterns")
    
    # Convert to DataFrame for analysis
    rows = []
    for key, data in all_annotations.items():
        row = {'id': key}
        
        # Add emotion
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            row['primary_emotion'] = data["emotions"]["primary_emotion"]
        
        # Add source
        if "source" in data:
            row['source'] = data["source"]
        
        # Add behavior indicators with normalized names
        if "behavioral_indicators" in data:
            for behavior, value in data["behavioral_indicators"].items():
                normalized_behavior = normalize_behavior_name(behavior)
                row[f"behavior_{normalized_behavior}"] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Analyze behavior columns
    behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
    logger.info(f"Found {len(behavior_cols)} behavior columns")
    
    # Count behaviors
    behavior_counts = {}
    for col in behavior_cols:
        non_null = df[col].notna().sum()
        behavior_counts[col] = non_null
    
    # Print top behaviors
    logger.info("Top behaviors by prevalence:")
    sorted_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
    for behavior, count in sorted_behaviors[:10]:
        percent = count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  - {behavior}: {count} ({percent:.1f}%)")
    
    return df

def create_behavior_matrix(df, output_file):
    """Create a behavior matrix that maps behaviors to emotions"""
    logger.info("Creating behavior matrix")
    
    # Check if a use_behavior_matrix.json exists already
    matrix_dir = os.path.dirname(output_file)
    use_matrix_path = os.path.join(matrix_dir, "use_behavior_matrix.json")
    
    # Also check current directory and parent directories
    search_paths = [
        use_matrix_path,
        "use_behavior_matrix.json",
        "../use_behavior_matrix.json",
        "../../use_behavior_matrix.json"
    ]
    
    existing_matrix = None
    for matrix_path in search_paths:
        if os.path.exists(matrix_path):
            logger.info(f"Found existing behavior matrix at {matrix_path}")
            existing_matrix = load_behavior_matrix(matrix_path)
            break
    
    if existing_matrix:
        logger.info("Using existing behavior matrix")
        
        # Save in our output location for consistency
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(existing_matrix, f, indent=2)
        
        logger.info(f"Saved behavior matrix to {output_file}")
        logger.info(f"Matrix contains {len(existing_matrix.get('behaviors', {}))} behaviors")
        
        return existing_matrix
    
    # Create new matrix if none exists
    logger.info("Creating new behavior matrix from data")
    
    # Standard emotions
    standard_emotions = [
        "Happy/Playful", 
        "Relaxed", 
        "Submissive/Appeasement",
        "Curiosity/Alertness", 
        "Stressed", 
        "Fearful/Anxious",
        "Aggressive/Threatening"
    ]
    
    # Get behavior columns
    behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
    if not behavior_cols:
        logger.warning("No behavior columns found")
        return {}
    
    # Get all emotions in the data
    emotions = df['primary_emotion'].unique().tolist() if 'primary_emotion' in df.columns else []
    
    # Create matrix structure
    matrix = {
        "emotions": standard_emotions,
        "behaviors": {}
    }
    
    # For each behavior, determine which emotions it's associated with
    for behavior in behavior_cols:
        short_name = behavior.replace('behavior_', '')
        
        # Skip if too few instances
        total_count = df[behavior].notna().sum()
        if total_count < 10:  # Skip behaviors with fewer than 10 instances
            continue
        
        # Initialize behavior entry
        matrix["behaviors"][short_name] = {emotion: 0 for emotion in standard_emotions}
        
        # Check each emotion
        for emotion in emotions:
            # Skip non-standard emotions
            if emotion not in standard_emotions:
                continue
            
            # Get subset for this emotion
            emotion_subset = df[df['primary_emotion'] == emotion]
            
            if len(emotion_subset) > 0:
                # For categorical behavior
                if df[behavior].dtype == 'object':
                    # Count occurrences of each value
                    value_counts = emotion_subset[behavior].value_counts(dropna=True)
                    
                    # Look for positive indicators
                    pos_values = sum(count for value, count in value_counts.items() 
                                 if str(value).lower() not in ['false', '0', 'no', 'none', 'nan'])
                    
                    # Calculate proportion
                    prop = pos_values / len(emotion_subset)
                    
                    # If over 15%, consider associated with this emotion
                    if prop >= 0.15:
                        matrix["behaviors"][short_name][emotion] = 1
                
                # For boolean or numeric behavior
                else:
                    # Calculate proportion of True/positive values
                    if df[behavior].dtype == bool:
                        prop = emotion_subset[behavior].mean()
                    else:
                        prop = (emotion_subset[behavior] > 0).mean()
                    
                    # If over 15%, consider associated with this emotion
                    if prop >= 0.15:
                        matrix["behaviors"][short_name][emotion] = 1
    
    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(matrix, f, indent=2)
    
    logger.info(f"Created behavior matrix with {len(matrix['behaviors'])} behaviors")
    logger.info(f"Saved to {output_file}")
    
    return matrix

def visualize_emotion_distribution(df, output_dir):
    """Create visualization of emotion distribution"""
    if 'primary_emotion' not in df.columns:
        logger.warning("No primary_emotion column found, skipping visualization")
        return
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Count emotions
    emotion_counts = df['primary_emotion'].value_counts()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = emotion_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Dog Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    
    # Add count labels
    for i, count in enumerate(emotion_counts):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(vis_dir, "emotion_distribution.png"), dpi=300)
    logger.info(f"Saved emotion distribution visualization")
    
    # Close the plot to free memory
    plt.close()

def visualize_behavior_relationships(df, output_dir):
    """Create visualizations of behavior-emotion relationships"""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Find behavior columns
    behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
    
    if not behavior_cols or 'primary_emotion' not in df.columns:
        logger.warning("Missing required columns for behavior relationship visualization")
        return
    
    # Count non-null values in each behavior column
    behavior_counts = {}
    for col in behavior_cols:
        non_null = df[col].notna().sum()
        behavior_counts[col] = non_null
    
    # Take top 5 behaviors
    top_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Create visualizations for each top behavior
    for behavior, _ in top_behaviors:
        try:
            logger.info(f"Creating visualization for {behavior}")
            
            # Create cross-tab
            cross_tab = pd.crosstab(df['primary_emotion'], df[behavior], normalize='index') * 100
            
            # Plot
            plt.figure(figsize=(12, 6))
            cross_tab.plot(kind='bar', stacked=True)
            plt.title(f'{behavior} by Emotion')
            plt.xlabel('Emotion')
            plt.ylabel('Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(vis_dir, f"{behavior}_by_emotion.png"), dpi=300)
            
            # Close the plot to free memory
            plt.close()
        except Exception as e:
            logger.error(f"Error creating visualization for {behavior}: {str(e)}")
    
    logger.info(f"Saved behavior relationship visualizations")

def process_datasets(args, paths, emotion_config):
    """Process all datasets and create combined annotations"""
    # Create output directory
    os.makedirs(paths["matrix_dir"], exist_ok=True)
    
    # Ensure split directories exist
    ensure_split_directories(paths)
    create_class_directories(paths, emotion_config)
    
    # Process each dataset
    if args.stanford_only:
        # Process only Stanford dataset
        video_annotations = {}
        personal_annotations = {}
        stanford_annotations, _ = process_stanford_dataset_enhanced(paths, emotion_config, args.force_reprocess, args.debug)
    else:
        # Process all datasets
        video_annotations, _ = process_video_folders(paths, emotion_config, args.force_reprocess)
        personal_annotations, _ = process_personal_annotations_enhanced(paths, emotion_config, args.force_reprocess, args.debug)
        stanford_annotations, _ = process_stanford_dataset_enhanced(paths, emotion_config, args.force_reprocess, args.debug)
    
    # Combine all annotations
    all_annotations = combine_and_save_annotations(
        video_annotations, personal_annotations, stanford_annotations, paths
    )
    
    # Create train/validation/test splits with video constraint
    splits = create_train_val_test_splits_with_video_constraint(all_annotations, paths, emotion_config)
    
    return all_annotations, splits

def analyze_datasets(all_annotations, paths, visualize=False):
    """Analyze datasets after processing"""
    # Analyze behavior patterns
    df = analyze_behavior_patterns(all_annotations)
    
    # Create behavior matrix
    if df is not None and 'primary_emotion' in df.columns:
        create_behavior_matrix(df, os.path.join(paths["matrix_dir"], "behavior_matrix.json"))
    
    # Create visualizations if requested
    if visualize and df is not None:
        visualize_emotion_distribution(df, paths["processed_dir"])
        visualize_behavior_relationships(df, paths["processed_dir"])
    
    return df

def main():
    """Main function"""
    # Parse command line arguments
    try:
        args = setup_argparse()
    except:
        # Create default arguments for Jupyter
        class DefaultArgs:
            def __init__(self):
                self.force_reprocess = False
                self.only_splits = False
                self.stanford_only = False
                self.analyze_only = False
                self.visualize = True
                self.debug = False
        args = DefaultArgs()
    
    # Setup paths
    paths = setup_paths_and_directories()
    
    # Setup emotion mapping
    emotion_config = setup_emotion_mapping()
    
    logger.info("=" * 80)
    logger.info("Pawnder Data Processing and Analysis - Enhanced Version")
    logger.info("=" * 80)
    
    all_annotations = None
    df = None
    
    # Check if we're only doing analysis
    if args.analyze_only:
        logger.info("Analysis-only mode: Loading existing annotations")
        
        # Load combined annotations
        combined_file = paths["combined_annotations_file"]
        if os.path.exists(combined_file):
            with open(combined_file, 'r') as f:
                all_annotations = json.load(f)
            logger.info(f"Loaded {len(all_annotations)} annotations from {combined_file}")
            
            # Analyze the data
            df = analyze_datasets(all_annotations, paths, args.visualize)
        else:
            logger.error(f"Combined annotations file not found: {combined_file}")
    
    # Check if we're only creating splits
    elif args.only_splits:
        logger.info("Splits-only mode: Loading existing annotations and creating splits")
        
        # Load combined annotations
        combined_file = paths["combined_annotations_file"]
        if os.path.exists(combined_file):
            with open(combined_file, 'r') as f:
                all_annotations = json.load(f)
            logger.info(f"Loaded {len(all_annotations)} annotations from {combined_file}")
            
            # Create train/validation/test splits with video constraint
            splits = create_train_val_test_splits_with_video_constraint(all_annotations, paths, emotion_config)
            
            # Analyze the data
            if args.visualize:
                df = analyze_datasets(all_annotations, paths, args.visualize)
        else:
            logger.error(f"Combined annotations file not found: {combined_file}")
    
    # Full processing
    else:
        logger.info("Full processing mode")
        
        # Process all datasets
        all_annotations, splits = process_datasets(args, paths, emotion_config)
        
        # Analyze the data
        if args.visualize:
            df = analyze_datasets(all_annotations, paths, args.visualize)
    
    logger.info("=" * 80)
    logger.info("Enhanced processing complete!")
    logger.info("=" * 80)
    
    return {
        'annotations': all_annotations,
        'df': df,
        'paths': paths,
        'emotion_config': emotion_config
    }

if __name__ == "__main__":
    main()