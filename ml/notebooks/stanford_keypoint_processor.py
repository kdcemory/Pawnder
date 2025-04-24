# stanford_keypoint_processor.py
# Processes Stanford Dog Pose keypoint data and updates annotation files

import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KeypointProcessor")

def load_keypoint_definitions(def_path):
    """Load keypoint definitions from CSV file"""
    if not os.path.exists(def_path):
        logger.warning(f"Keypoint definitions file not found: {def_path}")
        return None
    
    try:
        defs = pd.read_csv(def_path)
        logger.info(f"Loaded {len(defs)} keypoint definitions")
        return defs
    except Exception as e:
        logger.error(f"Error loading keypoint definitions: {e}")
        return None

def load_stanford_keypoints(base_dir, keypoint_defs=None):
    """
    Load keypoint data from the Stanford Dog Pose dataset
    
    Args:
        base_dir (str): Base directory containing the Stanford Dog Pose dataset
        keypoint_defs (pd.DataFrame): Keypoint definitions
        
    Returns:
        dict: Dictionary of keypoints indexed by image ID
    """
    train_dir = os.path.join(base_dir, "animal-pose-data/train")
    valid_dir = os.path.join(base_dir, "animal-pose-data/valid")
    
    keypoints = {}
    
    # Process both train and validation sets
    for data_dir, split in [(train_dir, "train"), (valid_dir, "valid")]:
        if not os.path.exists(data_dir):
            logger.warning(f"Directory not found: {data_dir}")
            continue
            
        # Load keypoint data
        labels_dir = os.path.join(data_dir, "labels")
        images_dir = os.path.join(data_dir, "images")
        
        if not os.path.exists(labels_dir):
            logger.warning(f"Labels directory not found: {labels_dir}")
            continue
            
        # Get all label files
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        logger.info(f"Found {len(label_files)} label files in {split} set")
        
        # Process each label file
        for label_file in tqdm(label_files, desc=f"Processing {split} keypoints"):
            try:
                # Get image ID from filename
                filename = os.path.basename(label_file)
                image_id = os.path.splitext(filename)[0]
                
                # Load entire content of the file
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                
                # Parse the keypoint data
                values = [float(x) for x in content.split()]
                
                # Organize keypoints based on the definitions
                structured_keypoints = []
                
                # If we have the definitions, use them to structure the data
                if keypoint_defs is not None and len(values) >= 3:
                    num_keypoints = len(keypoint_defs)
                    for i in range(num_keypoints):
                        # Each keypoint typically has 3 values: x, y, visibility
                        idx = i * 3
                        if idx + 2 < len(values):
                            x = values[idx]
                            y = values[idx + 1]
                            v = values[idx + 2]
                            
                            # Add this keypoint with its name if available
                            keypoint_name = f"keypoint_{i}"
                            if i < len(keypoint_defs):
                                if 'name' in keypoint_defs.columns:
                                    keypoint_name = keypoint_defs.iloc[i]['name']
                                else:
                                    keypoint_name = f"keypoint_{i}"
                            
                            structured_keypoints.append({
                                "id": i,
                                "name": keypoint_name,
                                "x": x,
                                "y": y,
                                "visibility": v
                            })
                else:
                    # No definitions available, just parse the raw values
                    # Assuming triplets of x, y, visibility
                    for i in range(0, len(values), 3):
                        if i + 2 < len(values):
                            structured_keypoints.append({
                                "id": i // 3,
                                "name": f"keypoint_{i // 3}",
                                "x": values[i],
                                "y": values[i + 1],
                                "visibility": values[i + 2]
                            })
                
                # Store keypoints if any were found
                if structured_keypoints:
                    # Check if corresponding image exists
                    img_path = os.path.join(images_dir, f"{image_id}.jpg")
                    has_image = os.path.exists(img_path)
                    
                    keypoints[image_id] = {
                        "keypoints": structured_keypoints,
                        "split": split,
                        "file": label_file,
                        "has_image": has_image
                    }
            
            except Exception as e:
                logger.warning(f"Error processing {label_file}: {str(e)}")
    
    # Count visible keypoints
    visible_counts = {}
    for img_id, data in keypoints.items():
        visible = sum(1 for kp in data["keypoints"] if kp["visibility"] > 0)
        if visible not in visible_counts:
            visible_counts[visible] = 0
        visible_counts[visible] += 1
    
    logger.info(f"Keypoint visibility distribution:")
    for count, occurrences in sorted(visible_counts.items()):
        logger.info(f"  {count} visible keypoints: {occurrences} images")
    
    logger.info(f"Loaded keypoints for {len(keypoints)} images from Stanford Dog Pose dataset")
    return keypoints

def update_annotations_with_keypoints(annotations_path, keypoints, output_path=None):
    """
    Update annotations CSV with keypoint data
    
    Args:
        annotations_path (str): Path to annotations CSV file
        keypoints (dict): Dictionary of keypoints by image ID
        output_path (str): Path to save updated annotations
        
    Returns:
        pd.DataFrame: Updated annotations
    """
    logger.info(f"Updating annotations at {annotations_path} with keypoints...")
    
    # Load annotations
    try:
        annotations = pd.read_csv(annotations_path)
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return None
    
    # Add keypoint columns
    annotations['has_keypoints'] = False
    annotations['keypoint_count'] = 0
    annotations['visible_keypoints'] = 0
    annotations['keypoint_split'] = None
    
    # Track matches
    match_count = 0
    
    # Update each row
    for idx, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Matching keypoints"):
        # Get image path
        if 'image_path' not in row:
            continue
            
        image_path = row['image_path']
        
        # Try different naming patterns to match with keypoints
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Direct match
        if image_id in keypoints:
            kps = keypoints[image_id]['keypoints']
            visible = sum(1 for kp in kps if kp["visibility"] > 0)
            
            annotations.at[idx, 'has_keypoints'] = True
            annotations.at[idx, 'keypoint_count'] = len(kps)
            annotations.at[idx, 'visible_keypoints'] = visible
            annotations.at[idx, 'keypoint_split'] = keypoints[image_id]['split']
            match_count += 1
            continue
        
        # Try with 'stanford_' prefix removed (if present)
        if image_id.startswith('stanford_'):
            stanford_id = image_id[9:]  # Remove 'stanford_' prefix
            if stanford_id in keypoints:
                kps = keypoints[stanford_id]['keypoints']
                visible = sum(1 for kp in kps if kp["visibility"] > 0)
                
                annotations.at[idx, 'has_keypoints'] = True
                annotations.at[idx, 'keypoint_count'] = len(kps)
                annotations.at[idx, 'visible_keypoints'] = visible
                annotations.at[idx, 'keypoint_split'] = keypoints[stanford_id]['split']
                match_count += 1
                continue
        
        # Try with original ID
        if 'original_id' in row and not pd.isna(row['original_id']):
            original_id = os.path.splitext(os.path.basename(row['original_id']))[0]
            if original_id in keypoints:
                kps = keypoints[original_id]['keypoints']
                visible = sum(1 for kp in kps if kp["visibility"] > 0)
                
                annotations.at[idx, 'has_keypoints'] = True
                annotations.at[idx, 'keypoint_count'] = len(kps)
                annotations.at[idx, 'visible_keypoints'] = visible
                annotations.at[idx, 'keypoint_split'] = keypoints[original_id]['split']
                match_count += 1
                continue
    
    logger.info(f"Matched keypoints for {match_count} out of {len(annotations)} annotations ({match_count/len(annotations)*100:.1f}%)")
    
    # Save updated annotations if output path provided
    if output_path:
        annotations.to_csv(output_path, index=False)
        logger.info(f"Saved updated annotations to {output_path}")
    
    return annotations

# Define paths
base_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
stanford_dir = f"{base_dir}/Data/Raw/stanford_dog_pose"
keypoint_def_path = f"{stanford_dir}/keypoint_definitions.csv"
annotations_dir = f"{base_dir}/Data/processed"
output_dir = f"{base_dir}/Data/annotations/stanford_keypoints"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load keypoint definitions
keypoint_defs = load_keypoint_definitions(keypoint_def_path)

# Load keypoints
keypoints = load_stanford_keypoints(stanford_dir, keypoint_defs)

if not keypoints:
    logger.error("No keypoints loaded. Exiting.")
else:
    logger.info(f"Successfully loaded {len(keypoints)} keypoint annotations")
    
    # Save keypoints for later use
    keypoint_file = os.path.join(output_dir, "stanford_keypoints.json")
    
    # Save in a format that can be serialized to JSON
    serializable_keypoints = {}
    for img_id, data in keypoints.items():
        serializable_keypoints[img_id] = {
            "keypoints": data["keypoints"],
            "split": data["split"],
            "file": data["file"],
            "has_image": data["has_image"]
        }
    
    with open(keypoint_file, 'w') as f:
        json.dump(serializable_keypoints, f)
    
    logger.info(f"Saved keypoints to {keypoint_file}")
    
    # Update annotations for each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(annotations_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        annotations_path = os.path.join(split_dir, 'annotations.csv')
        if not os.path.exists(annotations_path):
            logger.warning(f"Annotations file not found: {annotations_path}")
            continue
        
        # Update annotations
        updated = update_annotations_with_keypoints(
            annotations_path, 
            keypoints,
            output_path=annotations_path  # Overwrite original
        )
        
        if updated is not None:
            logger.info(f"Successfully updated {split} annotations")
            
    logger.info("Keypoint processing complete!")