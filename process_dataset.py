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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataProcessor")

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

def ensure_split_directories():
    """Ensure all split directories exist"""
    # Define paths using the main path structure
    paths = setup_paths_and_directories()
    base_dir = paths["output_dir"]
    
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'annotations'), exist_ok=True)
    
    print("Created all required split directories")

def validate_image_paths(annotations_path, base_dir=None):
    """Validate that all image paths in annotations exist"""
    # Determine format based on file extension
    if annotations_path.endswith('.json'):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        image_paths = list(annotations.keys())  # Filenames are the keys in your JSON structure
    elif annotations_path.endswith('.csv'):
        df = pd.read_csv(annotations_path)
        if 'image_path' in df.columns:
            image_paths = df['image_path'].tolist()
        else:
            print(f"Warning: No 'image_path' column found in {annotations_path}")
            return False
    else:
        print(f"Unsupported annotations format: {annotations_path}")
        return False
    
    # Determine base directory if not provided
    if base_dir is None:
        # Default: look for images in all_frames directory
        base_dir = os.path.join(os.path.dirname(os.path.dirname(annotations_path)), 'all_frames')
    
    # Check each path
    missing_files = []
    for path in image_paths:
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
            missing_files.append(path)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} image files referenced in annotations are missing")
        print(f"First 5 missing files: {missing_files[:5]}")
        return False
    else:
        print(f"All {len(image_paths)} image paths validated successfully")
        return True

def setup_argparse():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description="Process dog emotion datasets")
    parser.add_argument("--force_reprocess", action="store_true", 
                        help="Force reprocessing of all datasets from scratch")
    parser.add_argument("--only_splits", action="store_true", 
                        help="Only recreate train/val/test splits from existing processed data")
    parser.add_argument("--stanford_only", action="store_true",
                        help="Process only the Stanford dataset (skip videos and personal images)")
    return parser.parse_args()
    
def setup_paths_and_directories():
    """Setup paths and create output directories"""
    # Define key paths
    base_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
    dog_emotion_analyzer_dir = "/content/drive/MyDrive/Colab Notebooks/Dog Emotion Analyzer"
    videos_dir = os.path.join(base_dir, "Data/Raw/personal_dataset/videos")
    personal_images_dir = os.path.join(base_dir, "Data/Raw/personal_dataset/images")
    stanford_dir = os.path.join(base_dir, "Data/Raw/stanford_dog_pose")
    stanford_images_dir = os.path.join(stanford_dir, "Images")
    emotions_file = os.path.join(base_dir, "Data/interim/emotions_only.json")

    # Output directories
    output_dir = os.path.join(base_dir, "Data/processed")
    combined_frames_dir = os.path.join(output_dir, "all_frames")
    cache_dir = os.path.join(output_dir, "cache")
    
    # Create output directories
    os.makedirs(combined_frames_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache files
    video_cache = os.path.join(cache_dir, "video_annotations.json")
    stanford_cache = os.path.join(cache_dir, "stanford_annotations.json")
    personal_cache = os.path.join(cache_dir, "personal_annotations.json")
    combined_annotations_file = os.path.join(output_dir, "combined_annotations.json")
    
    # Return all paths as a dictionary
    return {
        "base_dir": base_dir,
        "dog_emotion_analyzer_dir": dog_emotion_analyzer_dir,
        "videos_dir": videos_dir,
        "personal_images_dir": personal_images_dir,
        "stanford_dir": stanford_dir,
        "stanford_images_dir": stanford_images_dir,
        "emotions_file": emotions_file,
        "output_dir": output_dir,
        "combined_frames_dir": combined_frames_dir,
        "cache_dir": cache_dir,
        "video_cache": video_cache,
        "stanford_cache": stanford_cache,
        "personal_cache": personal_cache,
        "combined_annotations_file": combined_annotations_file
    }

def setup_emotion_mapping():
    """Setup emotion mapping and class names"""
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

def create_class_directories(paths, emotion_config):
    """Create emotion class directories for all splits"""
    # Create output directories for each split
    for split in ["all_by_class", "train_by_class", "val_by_class", "test_by_class"]:
        split_dir = os.path.join(paths["output_dir"], split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create directories for each emotion class
        for safe_name in emotion_config["SAFE_CLASS_NAMES"]:
            os.makedirs(os.path.join(split_dir, safe_name), exist_ok=True)
        
        # Create unknown directory
        os.makedirs(os.path.join(split_dir, "unknown"), exist_ok=True)

def load_emotion_annotations(paths):
    """Load emotion annotations from JSON file"""
    print(f"Loading annotations from {paths['emotions_file']}")
    try:
        with open(paths["emotions_file"], 'r') as f:
            emotion_annotations = json.load(f)
        print(f"Loaded {len(emotion_annotations)} annotations")
        
        # Print emotion distribution
        emotion_counts = defaultdict(int)
        for img_id, data in emotion_annotations.items():
            if "emotions" in data and "primary_emotion" in data["emotions"]:
                emotion = data["emotions"]["primary_emotion"]
                emotion_counts[emotion] += 1

        print("\nEmotion distribution in annotations:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} frames")
            
        return emotion_annotations
    except Exception as e:
        print(f"Error loading emotion annotations: {str(e)}")
        return {}

def find_stanford_breed_directories(paths):
    """Find all Stanford breed directories"""
    # Check if stanford_images_dir exists
    if not os.path.exists(paths["stanford_images_dir"]):
        print(f"Error: Stanford images directory not found at {paths['stanford_images_dir']}")
        return []
    
    # Find all directories with the Stanford pattern (starting with 'n')
    breed_dirs = []
    for item in os.listdir(paths["stanford_images_dir"]):
        breed_path = os.path.join(paths["stanford_images_dir"], item)
        if os.path.isdir(breed_path) and item.startswith('n'):  # Stanford uses n* format for breed directories
            breed_dirs.append(breed_path)
    
    print(f"Found {len(breed_dirs)} Stanford breed directories")
    return breed_dirs

def load_stanford_split_annotations(paths, split, emotion_config):
    """Load Stanford annotations for a specific split"""
    # Try to find annotations file in the Dog Emotion Analyzer directory
    anno_dir = os.path.join(paths["dog_emotion_analyzer_dir"], "raw_data", "AWS S3download", split)
    annotations = {}
    
    # Get the emotion mapping
    EMOTION_MAPPING = emotion_config["EMOTION_MAPPING"]
    
    # Try different potential file paths
    potential_paths = [
        os.path.join(anno_dir, "annotations.csv"),
        os.path.join(anno_dir, f"{split}_annotations.csv"),
        os.path.join(paths["stanford_dir"], f"{split}_annotations.csv")
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Check if this is a valid annotation file
                if 'id' in df.columns and 'emotion' in df.columns:
                    for _, row in df.iterrows():
                        # Extract the ID (could be full image ID or just number)
                        img_id = str(row['id'])
                        emotion = row['emotion']
                        
                        # Map to standard emotion if needed
                        if emotion in EMOTION_MAPPING:
                            emotion = EMOTION_MAPPING[emotion]
                        
                        annotations[img_id] = emotion
                    
                    print(f"Loaded {len(annotations)} annotations from {path}")
                    return annotations
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    print(f"No Stanford annotations found for split: {split}")
    return annotations

def check_general_emotions_for_stanford(paths, emotion_annotations):
    """Extract Stanford images from general emotion annotations"""
    stanford_emotions = {}
    
    for img_id, annotation in emotion_annotations.items():
        # Check if this is likely a Stanford image (pattern n02085620_10074)
        basename = os.path.basename(img_id)
        basename_no_ext = os.path.splitext(basename)[0]
        
        if re.match(r'n\d+_\d+', basename_no_ext):
            # Get the emotion
            if isinstance(annotation, dict) and "emotions" in annotation:
                # If emotions is a dict with primary_emotion key
                if isinstance(annotation["emotions"], dict) and "primary_emotion" in annotation["emotions"]:
                    emotion = annotation["emotions"]["primary_emotion"]
                    stanford_emotions[basename_no_ext] = emotion
                # If emotions is a string
                elif isinstance(annotation["emotions"], str):
                    emotion = annotation["emotions"]
                    stanford_emotions[basename_no_ext] = emotion
            # Direct emotion string
            elif isinstance(annotation, str):
                emotion = annotation
                stanford_emotions[basename_no_ext] = emotion
    
    print(f"Found {len(stanford_emotions)} Stanford images in general emotion annotations")
    return stanford_emotions

def parse_xml_annotation(xml_file, emotion_mapping=None):
    """Parse CVAT XML annotations for both videos (track-based) and images (image-based)"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get video/task name from source
        source_name = None
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
                            # Process as behavioral indicator
                            annotation["behavioral_indicators"][name] = value
                    
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
                    image_name = image.get('name', f'frame_{image_id}')
                    
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
                                annotation["behavioral_indicators"][name] = value
                        
                        # Only add if we have a primary emotion
                        if "primary_emotion" in annotation["emotions"]:
                            annotations[frame_id] = annotation
        
        # If still no annotations, check for standalone boxes
        if not annotations:
            print(f"Checking for standalone boxes in {source_name}")
            for box in root.findall('.//box'):
                # If box is a direct child of root or another non-track, non-image element
                if box.getparent().tag not in ['track', 'image']:
                    frame_num = int(box.get('frame', 0))
                    frame_id = f"frame_{frame_num:06d}"
                    
                    # Create annotation entry
                    annotation = {
                        "video_name": source_name,
                        "frame": frame_num,
                        "original_format": "xml",
                        "source": "standalone_boxes",
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
                            # Process as behavioral indicator
                            annotation["behavioral_indicators"][name] = value
                    
                    # Only add if we have a primary emotion
                    if "primary_emotion" in annotation["emotions"]:
                        annotations[frame_id] = annotation
        
        print(f"Extracted {len(annotations)} annotations from {xml_file}")
        return annotations, source_name

    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {str(e)}")
        return {}, Nonedef process_video_folder(video_folder, paths, emotion_config):
    """Process a video folder with images and annotations"""
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
            f"frame_{frame_num}.jpg"
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
    """Process all video folders"""
    video_cache = paths["video_cache"]
    
    # Check if we've already processed videos and should use cache
    if os.path.exists(video_cache) and not force_reprocess:
        with open(video_cache, 'r') as f:
            video_frames = json.load(f)
        
        # Calculate the total frames from the loaded data
        total_video_frames = 0
        if isinstance(video_frames, dict):
            # If video_frames is a dictionary of videos
            if all(isinstance(v, dict) for v in video_frames.values()):
                total_video_frames = len(video_frames)
            # If video_frames is a dictionary of dictionaries
            else:
                total_video_frames = sum(1 for v in video_frames.values() if isinstance(v, dict))
        elif isinstance(video_frames, list):
            # If video_frames is a list of frames
            total_video_frames = len(video_frames)
        
        print(f"Loading {len(video_frames)} cached video frames from {video_cache} ({os.path.getsize(video_cache)} bytes)")
        print(f"Loaded {total_video_frames} cached video frames")
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

def process_personal_images(paths, emotion_config, emotion_annotations, force_reprocess):
    """Process personal dataset images"""
    personal_cache = paths["personal_cache"]
    
    # Check if we've already processed personal images and should use cache
    if os.path.exists(personal_cache) and not force_reprocess:
        print(f"Loading cached personal images from {personal_cache}")
        with open(personal_cache, 'r') as f:
            personal_frames = json.load(f)
        personal_count = len(personal_frames)
        print(f"Loaded {personal_count} cached personal images")
        return personal_frames, personal_count
        
    # Process personal images from scratch
    print("\nProcessing Personal Dataset Images from scratch")

    # Check if personal_images_dir exists
    if not os.path.exists(paths["personal_images_dir"]):
        print(f"Error: Personal images directory not found at {paths['personal_images_dir']}")
        return {}, 0

    # Build a map of all personal images
    personal_images = {}
    image_files = []

    # Look for images in the root and subdirectories
    for root, dirs, files in os.walk(paths["personal_images_dir"]):
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

    for img_id, annotation in tqdm(emotion_annotations.items(), desc="Processing personal annotations"):
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

        # Skip if image not found
        if not image_path:
            continue

        # Skip if it's a Stanford image
        is_stanford = re.match(r'n\d+_\d+', basename) is not None
        if is_stanford:
            continue

        # Skip if it's a video frame (already processed)
        is_video_frame = "frame_" in basename.lower()
        if is_video_frame:
            continue

        # Get emotion and create safe version
        emotion = annotation["emotions"]["primary_emotion"]
        safe_emotion = emotion.replace("/", "_").replace("\\", "_")

        # Create new filename
        new_filename = f"personal_{processed_count:06d}_{basename}{os.path.splitext(image_path)[1]}"
        dst_path = os.path.join(paths["combined_frames_dir"], new_filename)

        # Copy to combined directory
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy2(image_path, dst_path)

            # Copy to class directory
            class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
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
    
    # Cache the processed frames
    with open(personal_cache, 'w') as f:
        json.dump(processed_frames, f, indent=2)
    print(f"Cached personal frames to {personal_cache}")

    return processed_frames, processed_count

def process_stanford_dataset(paths, emotion_config, emotion_annotations, force_reprocess):
    """Process Stanford dog dataset images"""
    stanford_cache = paths["stanford_cache"]
    
    # Check if we've already processed Stanford images and should use cache
    if os.path.exists(stanford_cache) and not force_reprocess:
        print(f"Loading cached Stanford images from {stanford_cache}")
        with open(stanford_cache, 'r') as f:
            stanford_frames = json.load(f)
        stanford_count = len(stanford_frames)
        print(f"Loaded {stanford_count} cached Stanford images")
        return stanford_frames, stanford_count
        
    # Process Stanford images from scratch
    print("\nProcessing Stanford Dog Dataset from scratch")

    # Define additional paths for Stanford dataset
    stanford_images_dir = paths["stanford_images_dir"]
    processed_keypoints_path = os.path.join(paths["base_dir"], "Data/annotations/stanford_keypoints/stanford_keypoints.json")
    
    # Personal emotion annotations for Stanford
    train_emotions_path = os.path.join(paths["base_dir"], "Data/Raw/stanford_annotations/instances_Train.json")
    val_emotions_path = os.path.join(paths["base_dir"], "Data/Raw/stanford_annotations/instances_Validation.json")

    # 1. Load processed keypoint annotations if available
    keypoint_data = {}
    if os.path.exists(processed_keypoints_path):
        try:
            with open(processed_keypoints_path, 'r') as f:
                keypoint_data = json.load(f)
            print(f"Loaded {len(keypoint_data)} keypoint annotations")
        except Exception as e:
            print(f"Error loading keypoint data: {e}")

    # 2. Load emotion annotations with more flexible matching
    local_emotion_annotations = {}
    emotion_by_image_number = {}

    for path, split in [(train_emotions_path, 'train'), (val_emotions_path, 'validation')]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                # COCO format has 'images' and 'annotations' lists
                if 'images' in data and 'annotations' in data:
                    # Map image info
                    image_info = {}
                    for img in data['images']:
                        image_info[img['id']] = {
                            'file_name': img['file_name'],
                            'breed_id': None,
                            'image_number': None
                        }

                        # Extract breed ID and image number
                        filename = os.path.basename(img['file_name'])
                        match = re.match(r'(n\d+)_(\d+)\.', filename)
                        if match:
                            breed_id = match.group(1)
                            image_number = match.group(2)
                            image_info[img['id']]['breed_id'] = breed_id
                            image_info[img['id']]['image_number'] = image_number

                    # Process annotations
                    for ann in data['annotations']:
                        if 'image_id' in ann and 'attributes' in ann:
                            img_id = ann['image_id']
                            if img_id in image_info:
                                # Look for emotion attribute
                                emotion = None
                                for attr_name, attr_value in ann['attributes'].items():
                                    if 'emotion' in attr_name.lower() or 'primary' in attr_name.lower():
                                        emotion = attr_value
                                        break

                                if emotion:
                                    # Map to standardized emotion if needed
                                    if emotion in emotion_config["EMOTION_MAPPING"]:
                                        emotion = emotion_config["EMOTION_MAPPING"][emotion]
                                        
                                    # Store with full path
                                    local_emotion_annotations[image_info[img_id]['file_name']] = {
                                        'emotion': emotion,
                                        'split': split
                                    }

                                    # Also store by basename
                                    basename = os.path.basename(image_info[img_id]['file_name'])
                                    local_emotion_annotations[basename] = {
                                        'emotion': emotion,
                                        'split': split
                                    }

                                    # Store by breed_id+image_number
                                    if image_info[img_id]['breed_id'] and image_info[img_id]['image_number']:
                                        key = f"{image_info[img_id]['breed_id']}_{image_info[img_id]['image_number']}"
                                        emotion_by_image_number[key] = {
                                            'emotion': emotion,
                                            'split': split
                                        }

                print(f"Loaded annotations from {split}: {len(local_emotion_annotations)} entries, {len(emotion_by_image_number)} by image number")
            except Exception as e:
                print(f"Error loading {split} emotion annotations: {e}")

    # 3. Also check general emotion annotations from main process
    if emotion_annotations:
        print("Checking general emotion annotations for Stanford images...")
        stanford_count_in_general = 0
        
        for img_id, annotation in emotion_annotations.items():
            if "emotions" not in annotation or "primary_emotion" not in annotation["emotions"]:
                continue
                
            # Check if this is likely a Stanford image
            filename = os.path.basename(img_id)
            basename = os.path.splitext(filename)[0]
            
            # Stanford images often have format like n02085620_10074.jpg
            is_stanford = re.match(r'n\d+_\d+', basename) is not None
            
            if is_stanford:
                stanford_count_in_general += 1
                
                # Store in our local annotations if not already there
                if filename not in local_emotion_annotations:
                    local_emotion_annotations[filename] = {
                        'emotion': annotation["emotions"]["primary_emotion"],
                        'split': 'unknown'
                    }
                    
                # Also try storing by image number
                match = re.match(r'(n\d+)_(\d+)', basename)
                if match and match.group(1) and match.group(2):
                    key = f"{match.group(1)}_{match.group(2)}"
                    if key not in emotion_by_image_number:
                        emotion_by_image_number[key] = {
                            'emotion': annotation["emotions"]["primary_emotion"],
                            'split': 'unknown'
                        }
        
        print(f"Found {stanford_count_in_general} Stanford images in general emotion annotations")

    # 4. Build Stanford images map
    stanford_images = {}
    breed_dirs = []
    
    if os.path.exists(stanford_images_dir):
        for item in os.listdir(stanford_images_dir):
            breed_path = os.path.join(stanford_images_dir, item)
            if os.path.isdir(breed_path) and item.startswith('n'):  # Stanford uses n* format for breed directories
                breed_dirs.append(breed_path)

        print(f"Found {len(breed_dirs)} breed directories")

        for breed_dir in tqdm(breed_dirs, desc="Scanning Stanford directories"):
            breed_name = os.path.basename(breed_dir)
            image_files = glob.glob(os.path.join(breed_dir, "*.jpg")) + glob.glob(os.path.join(breed_dir, "*.png"))

            for image_path in image_files:
                image_name = os.path.basename(image_path)
                stanford_images[image_name] = image_path
                # Also store by breed_id+image_number
                match = re.match(r'(n\d+)_(\d+)\.', image_name)
                if match:
                    key = f"{match.group(1)}_{match.group(2)}"
                    stanford_images[key] = image_path

        print(f"Found {len(stanford_images)} total Stanford dog images")
    else:
        print(f"Stanford images directory not found at {stanford_images_dir}")
        return {}, 0

    # 5. Process images with both keypoints and emotions
    processed_frames = {}
    processed_count = 0
    matching_methods = {
        'direct_match': 0,
        'image_number_match': 0,
        'general_annotation_match': 0
    }

    # Process Stanford images
    if keypoint_data:
        # Process images with keypoints
        print("Processing Stanford images with keypoints...")
        
        # For each keypoint annotation
        for filename, keypoints in tqdm(keypoint_data.items(), desc="Processing Stanford keypoint annotations"):
            # Get the base filename and try to parse breed_id and image_number
            base_filename = os.path.basename(filename)
            image_id = filename  # Store for logging

            # Try to match directly
            emotion = None
            match_method = None

            # Check if this filename has a matching emotion annotation
            if base_filename in local_emotion_annotations:
                emotion = local_emotion_annotations[base_filename]['emotion']
                match_method = 'direct_match'

            # Try matching by breed_id + image_number
            if not emotion:
                match = re.match(r'(n\d+)_(\d+)(?:\..*)?$', base_filename)
                if match:
                    breed_id = match.group(1)
                    image_number = match.group(2)
                    key = f"{breed_id}_{image_number}"

                    if key in emotion_by_image_number:
                        emotion = emotion_by_image_number[key]['emotion']
                        match_method = 'image_number_match'

            # If still no match, check general annotations
            if not emotion:
                # Check original emotion_annotations
                for img_id, annotation in emotion_annotations.items():
                    if "emotions" in annotation and "primary_emotion" in annotation["emotions"]:
                        # Check if filenames match
                        other_filename = os.path.basename(img_id)
                        if other_filename == base_filename or os.path.splitext(other_filename)[0] == os.path.splitext(base_filename)[0]:
                            emotion = annotation["emotions"]["primary_emotion"]
                            match_method = 'general_annotation_match'
                            break

            # If no emotion match, just skip this image
            if not emotion:
                # No emotion found, so skip this image
                print(f"No emotion found for image {image_id}, skipping")
                continue  # Skip to the next iteration of the loop
            
            # If we have an emotion and keypoints, process the image
            if emotion:
                # Get image path - handle different formats
                image_path = None

                # Try direct match
                if base_filename in stanford_images:
                    image_path = stanford_images[base_filename]
                else:
                    # Try with/without extension
                    name_without_ext = os.path.splitext(base_filename)[0]
                    if name_without_ext in stanford_images:
                        image_path = stanford_images[name_without_ext]
                    elif f"{name_without_ext}.jpg" in stanford_images:
                        image_path = stanford_images[f"{name_without_ext}.jpg"]
                    elif f"{name_without_ext}.png" in stanford_images:
                        image_path = stanford_images[f"{name_without_ext}.png"]

                # If we found an image path, process it
                if image_path:
                    # Create new filename
                    new_filename = f"stanford_{processed_count:06d}_{os.path.basename(image_path)}"
                    dst_path = os.path.join(paths["combined_frames_dir"], new_filename)

                    # Create safe emotion name for directory
                    safe_emotion = emotion.replace("/", "_").replace("\\", "_")

                    try:
                        # Copy to combined directory
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(image_path, dst_path)

                        # Copy to class directory
                        class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
                        os.makedirs(class_dir, exist_ok=True)
                        class_path = os.path.join(class_dir, new_filename)
                        shutil.copy2(image_path, class_path)

                        # Add to processed frames
                        processed_frames[new_filename] = {
                            "emotions": {"primary_emotion": emotion},
                            "keypoints": keypoints,
                            "original_id": filename,
                            "original_path": image_path,
                            "source": "stanford"
                        }

                        processed_count += 1
                        if match_method:
                            matching_methods[match_method] += 1

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    else:
        # Alternative processing without keypoints
        print("Processing Stanford images without keypoints...")
        
        # Process any Stanford image that has an emotion annotation
        for img_id, annotation in tqdm(emotion_annotations.items(), desc="Processing Stanford annotations"):
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
            elif f"{basename}.jpg" in stanford_images:
                image_path = stanford_images[f"{basename}.jpg"]
            elif f"{basename}.png" in stanford_images:
                image_path = stanford_images[f"{basename}.png"]

            # Try matching by breed_id and image_number
            if not image_path:
                match = re.match(r'(n\d+)_(\d+)', basename)
                if match:
                    key = f"{match.group(1)}_{match.group(2)}"
                    if key in stanford_images:
                        image_path = stanford_images[key]

            if image_path:
                # Get emotion and create safe version
                emotion = annotation["emotions"]["primary_emotion"]
                safe_emotion = emotion.replace("/", "_").replace("\\", "_")

                # Create new filename with index for uniqueness
                new_filename = f"stanford_{processed_count:06d}_{os.path.basename(image_path)}"
                dst_path = os.path.join(paths["combined_frames_dir"], new_filename)

                # Copy to combined directory
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                try:
                    shutil.copy2(image_path, dst_path)

                    # Copy to class directory
                    class_dir = os.path.join(paths["output_dir"], "all_by_class", safe_emotion)
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
                    matching_methods['general_annotation_match'] += 1

                except Exception as e:
                    print(f"Error copying {image_path}: {e}")

    print(f"Processed {processed_count} Stanford images with emotion annotations")
    if matching_methods:
        print(f"Matching methods: {matching_methods}")

    # Count by emotion
    emotion_counts = defaultdict(int)
    for _, data in processed_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1

    print("Stanford emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count}")
    
    # Cache the processed frames
    with open(stanford_cache, 'w') as f:
        json.dump(processed_frames, f, indent=2)
    print(f"Cached Stanford frames to {stanford_cache}")
    
    return processed_frames, processed_count

def combine_and_save_annotations(video_frames, stanford_frames, personal_frames, paths):
    """Combine frames from all sources and save annotations"""
    # Combine all annotations
    all_frames = {}
    all_frames.update(video_frames)
    all_frames.update(stanford_frames)
    all_frames.update(personal_frames)

    total_frames = len(all_frames)
    print(f"\nTotal processed frames: {total_frames}")
    print(f"  - {len(video_frames)} video frames")
    print(f"  - {len(stanford_frames)} Stanford images")
    print(f"  - {len(personal_frames)} personal images")

    # Save combined annotations as JSON
    with open(paths["combined_annotations_file"], 'w') as f:
        json.dump(all_frames, f, indent=2)

    print(f"Saved annotations to: {paths['combined_annotations_file']}")
    
    # Convert JSON annotations to CSV format with existence check
    csv_path = os.path.join(paths["output_dir"], "combined_annotations.csv")
    if os.path.exists(paths["combined_annotations_file"]):
        convert_json_to_csv(paths["combined_annotations_file"], csv_path)
    else:
        print(f"Warning: JSON file not found: {paths['combined_annotations_file']}")
        # Create empty CSV with headers as a fallback
        df = pd.DataFrame(columns=['image_path', 'emotional_state', 'source'])
        df.to_csv(csv_path, index=False)
    
    # Count frames by emotion
    emotion_counts = defaultdict(int)
    for filename, data in all_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        emotion_counts[emotion] += 1

    print("\nTotal emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count} frames")
    
    return all_frames

def create_train_val_test_splits(all_frames, paths, emotion_config):
    """Create train/validation/test splits"""
    print("\nCreating train/val/test splits...")

    # Group by emotion
    frames_by_emotion = defaultdict(list)
    for filename, data in all_frames.items():
        emotion = data["emotions"]["primary_emotion"]
        safe_emotion = emotion.replace("/", "_").replace("\\", "_")
        frames_by_emotion[safe_emotion].append((filename, data))

    print("Emotion distribution for splitting:")
    for emotion, frames in sorted(frames_by_emotion.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {emotion}: {len(frames)} frames")

    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

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

    # Create split assignments for later use
    split_assignments = {}
    for filename, _ in train_files:
        split_assignments[filename] = 'train'
    for filename, _ in val_files:
        split_assignments[filename] = 'validation'
    for filename, _ in test_files:
        split_assignments[filename] = 'test'

    # Copy files to split directories and create split-specific annotation files
    split_counts = defaultdict(lambda: defaultdict(int))

    for file_list, split_dir, split_name in [
        (train_files, os.path.join(paths["output_dir"], "train_by_class"), "train"),
        (val_files, os.path.join(paths["output_dir"], "val_by_class"), "validation"),
        (test_files, os.path.join(paths["output_dir"], "test_by_class"), "test")
    ]:
        
        print(f"\nCreating {split_name} split with {len(file_list)} files...")

        # Create split directory and images subdirectory
        split_base_dir = os.path.join(paths["output_dir"], split_name)
        split_images_dir = os.path.join(split_base_dir, "images")
        os.makedirs(split_images_dir, exist_ok=True)
        
        # Create annotations subdirectory
        split_annotations_dir = os.path.join(split_base_dir, "annotations")
        os.makedirs(split_annotations_dir, exist_ok=True)
        
        # Create split-specific annotations dictionary
        split_annotations = {}

        for filename, data in tqdm(file_list, desc=f"Processing {split_name} split"):
            emotion = data["emotions"]["primary_emotion"]
            safe_emotion = emotion.replace("/", "_").replace("\\", "_")

            # Copy to class directory
            src_path = os.path.join(paths["combined_frames_dir"], filename)
            class_path = os.path.join(split_dir, safe_emotion, filename)

            # Also copy to split/images directory
            split_path = os.path.join(split_images_dir, filename)

            if os.path.exists(src_path):
                # Copy to class directory
                os.makedirs(os.path.dirname(class_path), exist_ok=True)
                shutil.copy2(src_path, class_path)
                
                # Copy to split/images directory
                os.makedirs(os.path.dirname(split_path), exist_ok=True)
                shutil.copy2(src_path, split_path)
                
                # Add to split annotations
                split_annotations[filename] = data
                
                # Update counts
                split_counts[split_name][safe_emotion] += 1

        # Save split-specific annotations as JSON
        split_json_path = os.path.join(split_annotations_dir, "annotations.json")
        with open(split_json_path, 'w') as f:
            json.dump(split_annotations, f, indent=2)
            
        # Convert to CSV - add the existence check here
        split_csv_path = os.path.join(split_base_dir, "annotations.csv")
        if os.path.exists(split_json_path):
            convert_json_to_csv(split_json_path, split_csv_path)
        else:
            print(f"Warning: JSON file not found: {split_json_path}")
            # Create empty CSV with headers as a fallback
            df = pd.DataFrame(columns=['image_path', 'emotional_state', 'source'])
            df.to_csv(split_csv_path, index=False)
        
        print(f"  Saved {split_name} annotations with {len(split_annotations)} entries")

    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print(f"Total images: {len(all_frames)}")

    # Print split statistics
    for split_name in ["train", "validation", "test"]:
        emotion_counts = split_counts[split_name]
        total = sum(emotion_counts.values())

        print(f"\n{split_name.capitalize()} split:")
        print(f"  Total: {total} images")
        print("  Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percent = count / total * 100 if total > 0 else 0
            print(f"    {emotion}: {count} ({percent:.1f}%)")
    
    # Validate image paths in the split CSV files
    print("\nValidating file paths...")
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(paths["output_dir"], split)
        annotations_path = os.path.join(split_dir, "annotations.csv")
        
        if os.path.exists(annotations_path):
            print(f"Validating {split} annotations...")
            images_dir = os.path.join(split_dir, "images")
            validate_image_paths(annotations_path, images_dir)
        else:
            print(f"Warning: Annotations file not found: {annotations_path}")
    
    return split_counts

def main():
    """Main processing pipeline"""
    # Setup command line arguments
    args = setup_argparse()
    
    # Setup paths and directories
    paths = setup_paths_and_directories()
    
    # Setup emotion mapping and class names
    emotion_config = setup_emotion_mapping()
    
    # Ensure all split directories exist
    ensure_split_directories()
    
    # Create class directories
    create_class_directories(paths, emotion_config)
    
    # Check if we should only recreate splits
    if args.only_splits:
        # Load combined annotations
        if os.path.exists(paths["combined_annotations_file"]):
            print(f"Loading combined annotations from {paths['combined_annotations_file']}")
            with open(paths["combined_annotations_file"], 'r') as f:
                all_frames = json.load(f)
            print(f"Loaded {len(all_frames)} combined annotations")
            
            # Create train/val/test splits
            create_train_val_test_splits(all_frames, paths, emotion_config)
            print("\nDataset preparation complete (splits only)!")
            return
        else:
            print(f"Error: Combined annotations file not found at {paths['combined_annotations_file']}")
            print("Cannot create splits without combined annotations. Please run the full processing pipeline first.")
            return
    
    # Load emotion annotations
    emotion_annotations = load_emotion_annotations(paths)
    if not emotion_annotations:
        print("Error loading emotion annotations. Aborting.")
        return
    
    # Process datasets based on flags
    if args.stanford_only:
        # Process only Stanford dataset
        video_frames = {}
        personal_frames = {}
        stanford_frames, stanford_count = process_stanford_dataset(paths, emotion_config, emotion_annotations, args.force_reprocess)
    else:
        # Process all datasets
        video_frames, total_video_frames = process_video_folders(paths, emotion_config, args.force_reprocess)
        personal_frames, personal_count = process_personal_images(paths, emotion_config, emotion_annotations, args.force_reprocess)
        stanford_frames, stanford_count = process_stanford_dataset(paths, emotion_config, emotion_annotations, args.force_reprocess)
        
    # Combine and save annotations
    all_frames = combine_and_save_annotations(video_frames, stanford_frames, personal_frames, paths)
    
    # Create train/val/test splits
    create_train_val_test_splits(all_frames, paths, emotion_config)
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    main()