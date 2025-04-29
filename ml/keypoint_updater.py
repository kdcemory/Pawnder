# Keypoint Updater
import os
import json
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KeypointUpdater")

def load_keypoint_annotations(keypoint_dir):
    """Load keypoint annotations from files"""
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

def update_annotations_with_keypoints(annotations_path, keypoints, output_path=None):
    """Update annotations CSV with keypoint data"""
    logger.info(f"Updating annotations at {annotations_path} with keypoints...")

    # Load annotations
    try:
        annotations = pd.read_csv(annotations_path)
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return None

    # Add keypoint columns
    annotations['has_keypoints'] = False
    annotations['keypoint_file'] = None

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
            annotations.at[idx, 'has_keypoints'] = True
            annotations.at[idx, 'keypoint_file'] = f"{image_id}.json"
            match_count += 1
            continue

        # Try with 'stanford_' prefix removed (if present)
        if image_id.startswith('stanford_'):
            stanford_id = image_id[9:]  # Remove 'stanford_' prefix
            if stanford_id in keypoints:
                annotations.at[idx, 'has_keypoints'] = True
                annotations.at[idx, 'keypoint_file'] = f"{stanford_id}.json"
                match_count += 1
                continue

        # Try with original ID
        if 'original_id' in row and not pd.isna(row['original_id']):
            original_id = os.path.splitext(os.path.basename(row['original_id']))[0]
            if original_id in keypoints:
                annotations.at[idx, 'has_keypoints'] = True
                annotations.at[idx, 'keypoint_file'] = f"{original_id}.json"
                match_count += 1
                continue

    logger.info(f"Matched keypoints for {match_count} out of {len(annotations)} annotations ({match_count/len(annotations)*100:.1f}%)")

    # Save updated annotations if output path provided
    if output_path:
        annotations.to_csv(output_path, index=False)
        logger.info(f"Saved updated annotations to {output_path}")

    return annotations

# Define your directories explicitly
base_dir = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
keypoint_dir = f"{base_dir}/Data/annotations/stanford_keypoints"
# Set to the processed directory which contains train, validation, test folders
annotations_dir = f"{base_dir}/Data/processed"

# Verify directories exist
if not os.path.exists(keypoint_dir):
    logger.error(f"Keypoint directory not found: {keypoint_dir}")
else:
    # Load keypoints
    keypoints = load_keypoint_annotations(keypoint_dir)

    if not keypoints:
        logger.error("No keypoints loaded. Exiting.")
    else:
        # Update annotations for each split
        for split in ['train', 'validation', 'test']:
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