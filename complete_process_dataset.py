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

def setup_paths_and_directories(base_dir="C:\\Users\\thepf\\pawnder"):
    """Setup paths and create output directories - EXACTLY like original process_dataset.py"""
    
    print(f"Using base directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Define key paths - exactly from original process_dataset.py
    # Check for different data directory structures
    data_dir_options = [
        os.path.join(base_dir, "Data"),
        os.path.join(base_dir, "Data", "Raw"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "Raw")
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
    videos_dir = os.path.join(data_root, "Raw", "Videos")  # YOUR ACTUAL PATH
    personal_images_dir = os.path.join(data_root, "Raw", "personal_dataset")
    stanford_dir = os.path.join(data_root, "Raw", "stanford_dog_pose")
    stanford_images_dir = os.path.join(stanford_dir, "Images")
    stanford_annotations_dir = os.path.join(data_root, "Raw", "stanford_annotations")
    matrix_dir = os.path.join(data_root, "Data", "Matrix")
    processed_dir = os.path.join(base_dir, "processed")
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

def debug_personal_xml(xml_file, personal_images_dir):
    """Debug what's in the personal XML vs actual images"""
    logger.info("=== DEBUGGING PERSONAL DATASET ===")
    
    # Check actual images in directory
    images_dir = os.path.join(personal_images_dir, "images")
    if os.path.exists(images_dir):
        actual_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            actual_images.extend(glob.glob(os.path.join(images_dir, ext)))
            actual_images.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        logger.info(f"Actual images in {images_dir}: {len(actual_images)}")
        if actual_images:
            logger.info(f"Sample actual images: {[os.path.basename(f) for f in actual_images[:5]]}")
    
    # Parse XML to see what's in there
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        images = root.findall('.//image')
        logger.info(f"Images in XML: {len(images)}")
        
        # Sample some image names from XML
        sample_xml_images = []
        for i, image in enumerate(images[:10]):
            image_name = image.get('name', f'unnamed_{i}')
            sample_xml_images.append(image_name)
        
        logger.info(f"Sample XML image names: {sample_xml_images}")
        
        # Check how many have emotions
        images_with_emotions = 0
        images_with_personal_match = 0
        
        for image in images[:100]:  # Check first 100
            image_name = image.get('name', '')
            boxes = image.findall('.//box')
            
            has_emotion = False
            for box in boxes:
                attributes = box.findall('.//attribute')
                for attr in attributes:
                    name = attr.get('name', '').strip()
                    if name == "Primary Emotion" or ("primary" in name.lower() and "emotion" in name.lower()):
                        has_emotion = True
                        break
                if has_emotion:
                    break
            
            if has_emotion:
                images_with_emotions += 1
                
            # Check if this might be a personal image
            if any(keyword in image_name.lower() for keyword in ['personal', 'dataset']) or \
               any(actual_name in image_name for actual_name in [os.path.basename(f) for f in actual_images[:10]]):
                images_with_personal_match += 1
        
        logger.info(f"Images with emotions (first 100 checked): {images_with_emotions}")
        logger.info(f"Images matching personal dataset (first 100 checked): {images_with_personal_match}")
        
    except Exception as e:
        logger.error(f"Error debugging XML: {e}")

def debug_stanford_coco(json_file, stanford_images_dir):
    """Debug Stanford COCO file"""
    logger.info("=== DEBUGGING STANFORD DATASET ===")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"COCO file keys: {list(data.keys())}")
        logger.info(f"Number of images: {len(data.get('images', []))}")
        logger.info(f"Number of annotations: {len(data.get('annotations', []))}")
        
        # Sample image data
        if data.get('images'):
            sample_image = data['images'][0]
            logger.info(f"Sample image data: {sample_image}")
        
        # Sample annotation data
        if data.get('annotations'):
            sample_ann = data['annotations'][0]
            logger.info(f"Sample annotation data: {sample_ann}")
            
            # Check for attributes in annotations
            attrs_found = 0
            for ann in data['annotations'][:10]:
                if 'attributes' in ann:
                    attrs_found += 1
                    logger.info(f"Sample attributes: {ann['attributes']}")
                    break
            
            logger.info(f"Annotations with attributes: {attrs_found}/{len(data['annotations'])}")
        
        # Check actual Stanford images
        if os.path.exists(stanford_images_dir):
            breed_dirs = [d for d in os.listdir(stanford_images_dir) 
                         if os.path.isdir(os.path.join(stanford_images_dir, d))]
            logger.info(f"Actual breed directories: {len(breed_dirs)}")
            
            # Sample some actual image files
            if breed_dirs:
                sample_breed = breed_dirs[0]
                breed_path = os.path.join(stanford_images_dir, sample_breed)
                breed_images = glob.glob(os.path.join(breed_path, "*.jpg"))
                logger.info(f"Sample breed '{sample_breed}' has {len(breed_images)} images")
                if breed_images:
                    logger.info(f"Sample image names: {[os.path.basename(f) for f in breed_images[:3]]}")
        
    except Exception as e:
        logger.error(f"Error debugging COCO file: {e}")

def parse_personal_xml_fixed(xml_file, emotion_config, personal_images_dir):
    """Parse personal dataset XML file with filtering for actual personal images"""
    logger.info(f"Parsing personal XML file: {xml_file}")
    
    # First, get list of actual images in personal dataset
    images_dir = os.path.join(personal_images_dir, "images")
    actual_images = set()
    
    if os.path.exists(images_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_file in glob.glob(os.path.join(images_dir, ext)):
                actual_images.add(os.path.basename(img_file))
            for img_file in glob.glob(os.path.join(images_dir, ext.upper())):
                actual_images.add(os.path.basename(img_file))
    
    logger.info(f"Found {len(actual_images)} actual images in personal dataset")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotations = {}
        processed_count = 0
        skipped_count = 0
        
        # Process all images in the XML
        images = root.findall('.//image')
        logger.info(f"Found {len(images)} images in XML")
        
        for image in tqdm(images, desc="Processing personal images"):
            image_id = image.get('id', '0')
            image_name = image.get('name', f'image_{image_id}')
            
            # FILTER: Only process images that actually exist in personal dataset
            base_image_name = os.path.basename(image_name)
            if base_image_name not in actual_images:
                # Check if it might be in a subdirectory path
                if not any(actual_img in image_name for actual_img in actual_images):
                    skipped_count += 1
                    continue
            
            # Process all boxes in this image
            boxes = image.findall('.//box')
            
            for box_idx, box in enumerate(boxes):
                # Create unique frame ID
                frame_id = f"personal_{base_image_name}_{box_idx}"
                
                annotation = {
                    "image_name": base_image_name,
                    "image_id": image_id,
                    "box_index": box_idx,
                    "source": "personal",
                    "emotions": {},
                    "behavioral_indicators": {}
                }
                
                # Extract all attributes from this box
                attributes = box.findall('.//attribute')
                
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
                        annotation["emotions"]["primary_emotion"] = value
                    else:
                        # Store as behavioral indicator
                        clean_name = name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')
                        annotation["behavioral_indicators"][clean_name] = value
                
                # Only keep annotations with primary emotions
                if "primary_emotion" in annotation["emotions"]:
                    annotations[frame_id] = annotation
                    processed_count += 1
        
        logger.info(f"Processed {processed_count} personal annotations from {len(actual_images)} images")
        logger.info(f"Skipped {skipped_count} images not in personal dataset")
        return annotations
        
    except Exception as e:
        logger.error(f"Error parsing personal XML: {e}")
        return {}

def load_stanford_coco_annotations_fixed(json_file):
    """Load Stanford annotations from COCO format JSON with better debugging"""
    try:
        logger.info(f"Loading Stanford COCO annotations from: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            logger.warning(f"Invalid COCO format in {json_file}")
            return {}
        
        # Create mapping from image_id to filename
        image_map = {}
        for image in data["images"]:
            image_map[image["id"]] = image["file_name"]
        
        logger.info(f"Found {len(image_map)} images in COCO file")
        
        # Process annotations with detailed debugging
        emotions = {}
        emotions_found = 0
        no_attributes = 0
        no_emotion_attr = 0
        
        for ann in data["annotations"]:
            if "image_id" not in ann or ann["image_id"] not in image_map:
                continue
                
            image_filename = image_map[ann["image_id"]]
            
            # Look for emotion in attributes
            emotion = None
            if "attributes" in ann:
                if not ann["attributes"]:
                    no_attributes += 1
                else:
                    for attr_name, attr_value in ann["attributes"].items():
                        logger.debug(f"Checking attribute: {attr_name} = {attr_value}")
                        if "emotion" in attr_name.lower() or "primary" in attr_name.lower():
                            emotion = attr_value
                            emotions_found += 1
                            break
                    
                    if not emotion:
                        no_emotion_attr += 1
                        # Log what attributes we did find
                        logger.debug(f"No emotion found in attributes: {list(ann['attributes'].keys())}")
            else:
                no_attributes += 1
            
            # Also check for category_id mapping (common in COCO)
            if not emotion and "category_id" in ann:
                # Check if categories define emotions
                if "categories" in data:
                    for cat in data["categories"]:
                        if cat["id"] == ann["category_id"]:
                            if "emotion" in cat.get("name", "").lower():
                                emotion = cat["name"]
                                emotions_found += 1
                                break
            
            if emotion:
                emotions[image_filename] = {
                    "emotions": {"primary_emotion": emotion},
                    "image_id": ann["image_id"],
                    "annotation_id": ann.get("id", 0)
                }
        
        logger.info(f"Stanford annotation debugging:")
        logger.info(f"  - Total annotations: {len(data['annotations'])}")
        logger.info(f"  - Annotations with attributes: {len(data['annotations']) - no_attributes}")
        logger.info(f"  - Annotations with empty attributes: {no_attributes}")
        logger.info(f"  - Annotations with emotion attributes: {emotions_found}")
        logger.info(f"  - Annotations without emotion attributes: {no_emotion_attr}")
        logger.info(f"  - Final emotions extracted: {len(emotions)}")
        
        return emotions
        
    except Exception as e:
        logger.error(f"Error loading Stanford COCO annotations: {e}")
        return {}

def parse_xml_annotation(xml_file, emotion_mapping=None):
    """Parse CVAT XML annotations - from original process_dataset.py"""
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
                            # Treat as behavioral indicator
                            clean_name = name.lower().replace(' ', '_').replace('/', '_')
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
                                # Process as behavioral indicator
                                clean_name = name.lower().replace(' ', '_').replace('/', '_')
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
    """Process a video folder with images and annotations - from original process_dataset.py"""
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
            except Exception as e:
                print(f"  Error copying {src_path}: {str(e)}")
                missing_count += 1
        else:
            missing_count += 1

    print(f"  Processed {processed_count} frames, {missing_count} frames missing")
    return processed_count, processed_frames

def process_video_folders(paths, emotion_config, force_reprocess):
    """Process all video folders - exactly from original process_dataset.py"""
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

def process_personal_annotations_fixed(paths, emotion_config, force_reprocess=False, debug=False):
    """Process personal annotations with proper filtering"""
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
    
    logger.info("Processing personal dataset from scratch")
    
    # Find the XML file
    xml_locations = [
        os.path.join(paths["personal_images_dir"], "annotations", "annotations.xml"),
        os.path.join(paths["personal_images_dir"], "annotations.xml"),
        os.path.join(paths["base_dir"], "Data", "Raw", "personal_annotations", "annotations.xml"),
        os.path.join(paths["base_dir"], "Data", "Raw", "annotations.xml")
    ]
    
    xml_file = None
    for xml_path in xml_locations:
        if os.path.exists(xml_path):
            xml_file = xml_path
            logger.info(f"Found personal XML file: {xml_file}")
            break
    
    if not xml_file:
        logger.warning("No personal XML annotation file found")
        return {}, 0
    
    # Debug if requested
    if debug:
        debug_personal_xml(xml_file, paths["personal_images_dir"])
    
    # Parse the XML file with filtering
    annotations = parse_personal_xml_fixed(xml_file, emotion_config, paths["personal_images_dir"])
    
    if not annotations:
        logger.warning("No valid personal annotations extracted")
        return {}, 0
    
    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in annotations.items():
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            emotion = data["emotions"]["primary_emotion"]
            emotion_counts[emotion] += 1
    
    logger.info(f"Personal annotations summary:")
    logger.info(f"  Total annotations: {len(annotations)}")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {emotion}: {count}")
    
    # Cache results
    if annotations:
        os.makedirs(os.path.dirname(personal_cache), exist_ok=True)
        with open(personal_cache, 'w') as f:
            json.dump(annotations, f, indent=2)
        logger.info(f"Cached personal annotations to {personal_cache}")
    
    return annotations, len(annotations)

def process_stanford_dataset_fixed(paths, emotion_config, force_reprocess, debug=False):
    """Process Stanford dataset with debugging"""
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
    
    logger.info("Processing Stanford dataset from scratch")
    
    # Find Stanford emotion annotation file
    emotion_file_locations = [
        os.path.join(paths["stanford_annotations_dir"], "stanford_annotations.json"),
        os.path.join(paths["stanford_annotations_dir"], "instances_Validation.json"),
        os.path.join(paths["stanford_dir"], "stanford_annotations.json"),
    ]
    
    emotion_file = None
    for emotion_path in emotion_file_locations:
        if os.path.exists(emotion_path):
            emotion_file = emotion_path
            logger.info(f"Found Stanford emotion file: {emotion_file}")
            break
    
    if not emotion_file:
        logger.warning("No Stanford emotion annotation file found")
        return {}, 0
    
    # Debug if requested
    if debug:
        debug_stanford_coco(emotion_file, paths["stanford_images_dir"])
    
    # Load emotions from COCO format
    emotion_annotations = load_stanford_coco_annotations_fixed(emotion_file)
    if not emotion_annotations:
        logger.warning("No valid Stanford emotion annotations loaded")
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
    
    # Match emotions to images and copy them
    stanford_annotations = {}
    processed_count = 0
    not_found_count = 0
    
    for img_key, emotion_data in tqdm(emotion_annotations.items(), desc="Processing Stanford emotions"):
        # Skip if no emotion
        if "emotions" not in emotion_data or "primary_emotion" not in emotion_data["emotions"]:
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
            emotion = emotion_data["emotions"]["primary_emotion"]
            
            # Apply emotion mapping
            if emotion in emotion_config["EMOTION_MAPPING"]:
                emotion = emotion_config["EMOTION_MAPPING"][emotion]
            
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
                
                # Add to annotations
                stanford_annotations[new_filename] = {
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
    
    logger.info(f"Stanford image matching results:")
    logger.info(f"  - Images found and processed: {processed_count}")
    logger.info(f"  - Images not found: {not_found_count}")
    
    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in stanford_annotations.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1
    
    logger.info(f"Stanford annotations summary:")
    logger.info(f"  Total annotations: {len(stanford_annotations)}")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {emotion}: {count}")
    
    # Cache results
    if stanford_annotations:
        os.makedirs(os.path.dirname(stanford_cache), exist_ok=True)
        with open(stanford_cache, 'w') as f:
            json.dump(stanford_annotations, f, indent=2)
        logger.info(f"Cached Stanford annotations to {stanford_cache}")
    
    return stanford_annotations, len(stanford_annotations)

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
        
        # Add behavior indicators
        if "behavioral_indicators" in data:
            for behavior, value in data["behavioral_indicators"].items():
                row[f"behavior_{behavior}"] = value
        
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

def create_train_val_test_splits(all_annotations, paths, emotion_config):
    """Create train/validation/test splits"""
    logger.info("Creating train/validation/test splits")
    
    # Group by emotion
    by_emotion = defaultdict(list)
    for key, data in all_annotations.items():
        if "emotions" in data and "primary_emotion" in data["emotions"]:
            emotion = data["emotions"]["primary_emotion"]
            # Map to standardized emotions if needed
            if emotion in emotion_config["EMOTION_MAPPING"]:
                emotion = emotion_config["EMOTION_MAPPING"][emotion]
            by_emotion[emotion].append((key, data))
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Create splits
    train_items = []
    val_items = []
    test_items = []
    
    # Stratified split by emotion
    for emotion, items in by_emotion.items():
        random.shuffle(items)
        total = len(items)
        
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_items.extend(items[:train_count])
        val_items.extend(items[train_count:train_count+val_count])
        test_items.extend(items[train_count+val_count:])
    
    # Print split statistics
    logger.info(f"Split statistics:")
    logger.info(f"  - Train: {len(train_items)} items")
    logger.info(f"  - Validation: {len(val_items)} items")
    logger.info(f"  - Test: {len(test_items)} items")
    
    # Convert splits to dictionaries
    train_annotations = {key: data for key, data in train_items}
    val_annotations = {key: data for key, data in val_items}
    test_annotations = {key: data for key, data in test_items}
    
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
            
            # Add behavior indicators
            if "behavioral_indicators" in data:
                for behavior, value in data["behavioral_indicators"].items():
                    row[f"behavior_{behavior}"] = value
            
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
        for _, data in split_data.items():
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                emotions[emotion] += 1
        
        logger.info(f"{split_name.capitalize()} emotion distribution:")
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            percent = count / len(split_data) * 100 if split_data else 0
            logger.info(f"  - {emotion}: {count} ({percent:.1f}%)")
    
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
        
        # Add behavior indicators
        if "behavioral_indicators" in data:
            for behavior, value in data["behavioral_indicators"].items():
                row[f"behavior_{behavior}"] = value
        
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
    
    # Check if a primary_behavior_matrix.json exists already
    matrix_dir = os.path.dirname(output_file)
    primary_matrix_path = os.path.join(matrix_dir, "primary_behavior_matrix.json")
    
    if os.path.exists(primary_matrix_path):
        logger.info(f"Found existing primary_behavior_matrix.json at {primary_matrix_path}")
        
        try:
            with open(primary_matrix_path, 'r') as f:
                primary_matrix = json.load(f)
            
            # Check if it has the expected structure
            if "behavioral_states" in primary_matrix and "behavior_categories" in primary_matrix:
                logger.info("Using existing primary_behavior_matrix.json")
                
                # Convert to our matrix format for compatibility
                states = primary_matrix["behavioral_states"]
                categories = primary_matrix["behavior_categories"]
                
                # Create state names list
                state_names = [state["name"] for state in states]
                
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
                
                # Create matrix
                matrix = {
                    "emotions": state_names,
                    "behaviors": behaviors
                }
                
                # Save in our format for compatibility
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(matrix, f, indent=2)
                
                logger.info(f"Converted primary matrix with {len(behaviors)} behaviors")
                logger.info(f"Saved to {output_file}")
                
                return matrix
        except Exception as e:
            logger.error(f"Error processing primary_behavior_matrix.json: {e}")
            # Continue with creating a new matrix
    
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
        return
    
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
        stanford_annotations, _ = process_stanford_dataset_fixed(paths, emotion_config, args.force_reprocess, args.debug)
    else:
        # Process all datasets
        video_annotations, _ = process_video_folders(paths, emotion_config, args.force_reprocess)
        personal_annotations, _ = process_personal_annotations_fixed(paths, emotion_config, args.force_reprocess, args.debug)
        stanford_annotations, _ = process_stanford_dataset_fixed(paths, emotion_config, args.force_reprocess, args.debug)
    
    # Combine all annotations
    all_annotations = combine_and_save_annotations(
        video_annotations, personal_annotations, stanford_annotations, paths
    )
    
    # Create train/validation/test splits
    splits = create_train_val_test_splits(all_annotations, paths, emotion_config)
    
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
    logger.info("Pawnder Data Processing and Analysis")
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
            
            # Create train/validation/test splits
            splits = create_train_val_test_splits(all_annotations, paths, emotion_config)
            
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
    logger.info("Processing complete!")
    logger.info("=" * 80)
    
    return {
        'annotations': all_annotations,
        'df': df,
        'paths': paths,
        'emotion_config': emotion_config
    }

if __name__ == "__main__":
    main()
                