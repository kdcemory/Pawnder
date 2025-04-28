# config.py
# Central configuration file for the Pawnder data pipeline

import os
import json
from pathlib import Path
import yaml

# Project root directory (in Colab) - GitHub repo location
PROJECT_ROOT = '/content/pawnder'

# Google Drive configuration
GDRIVE_ENABLED = True
GDRIVE_ROOT = '/content/drive/MyDrive/Colab Notebooks/Pawnder'

# Helper function to normalize paths
def normalize_path(path):
    """Normalize a path to avoid duplicates caused by different representations"""
    if isinstance(path, str) and path.startswith('s3://'):
        return path  # Don't normalize S3 paths
    return os.path.normpath(path)

# Data directories
DATA_ROOT = normalize_path(os.path.join(GDRIVE_ROOT, 'Data'))  # Data in Google Drive - note uppercase 'Data'

# Apply normalization to all paths
DATA_DIRS = {k: normalize_path(v) for k, v in {
    # Raw data directories
    'raw': os.path.join(DATA_ROOT, 'Raw'),
    'stanford_original': os.path.join(DATA_ROOT, 'Raw', 'stanford_dog_pose'),
    'stanford_annotations': os.path.join(DATA_ROOT, 'Raw', 'stanford_annotations'),
    'personal_dataset': os.path.join(DATA_ROOT, 'Raw', 'personal_dataset'),
    
    # Matrix data
    'matrix': os.path.join(DATA_ROOT, 'Matrix'),
    
    # Processed data directories
    'interim': os.path.join(DATA_ROOT, 'interim'),
    'processed': os.path.join(DATA_ROOT, 'processed'),
    'augmented': os.path.join(DATA_ROOT, 'augmented'),
    
    # Annotation directories
    'annotations': os.path.join(DATA_ROOT, 'annotations'),
    'cvat_exports': os.path.join(DATA_ROOT, 'annotations', 'cvat_exports'),
    'emotion_annotations': os.path.join(DATA_ROOT, 'annotations', 'emotion_annotations'),
    'merged_annotations': os.path.join(DATA_ROOT, 'annotations', 'merged_annotations'),
    'stanford_keypoints': os.path.join(DATA_ROOT, 'annotations', 'stanford_keypoints'),
    
    # Augmentation subdirectories
    'background_variations': os.path.join(DATA_ROOT, 'augmented', 'background_variations'),
    'lighting_variations': os.path.join(DATA_ROOT, 'augmented', 'lighting_variations'),
    'pose_variations': os.path.join(DATA_ROOT, 'augmented', 'pose_variations'),
    'visualizations': os.path.join(DATA_ROOT, 'augmented', 'visualizations'),
    
    # Model directories
    'models': os.path.join(GDRIVE_ROOT, 'Models'),
    
    # AWS directories - for hybrid environment
    'aws_data': 's3://pawnder-media-storage/training',
    'aws_models': 's3://pawnder-media-storage/models',
    'aws_results': 's3://pawnder-media-storage/results'
}.items()}

# AWS configuration for hybrid environment
AWS_CONFIG = {
    'region': 'us-east-1',  # Replace with your AWS region
    'bucket': 'pawnder-media-storage',
    'ec2_instance': 'pawnder-model-training',
    'use_spot': True  # Set to True to use spot instances for training
}

# Try to get AWS credentials from Colab secrets if available
try:
    from google.colab import userdata
    AWS_CONFIG['aws_access_key_id'] = userdata.get('AWS_ACCESS_KEY_ID')
    AWS_CONFIG['aws_secret_access_key'] = userdata.get('AWS_SECRET_ACCESS_KEY')
    print("AWS credentials loaded from Colab secrets")
except (ImportError, Exception) as e:
    # If not in Colab or secrets not available, leave them unset
    # (will fall back to environment variables or AWS credential file)
    print(f"Note: AWS credentials not loaded from Colab secrets")

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
    "Possessive, Territorial, Dominant": "Fearful/Anxious",
    "Fear or Aggression": "Aggressive/Threatening",
    "Pain": "Stressed"
}

# Secondary phrases for each primary emotion
SECONDARY_ANNOTATIONS = {
    "Happy/Playful": [
        "Play with me, please.",
        "Give me some attention",
        "Please, I'm a good dog",
        "I love you, I love you, I love you",
        "I'm friendly",
        "I'm ready",
        "More please",
        "I love this",
        "Don't stop",
        "This couch is perfect",
        "Ball! Ball! Ball! BALL!",
        "Again! Let's do it again!",
        "I'm so excited!",
        "Chase me if you can!",
        "Zoomies time!",
        "Best day ever!",
        "Let's go to the park!",
        "Your home!"
    ],
    "Relaxed": [
        "ZZZzzzz",
        "Just relaxing",
        "Nap time",
        "Just 5 more minutes",
        "Belly rubs please",
        "I love nap o'clock",
        "Sunbeam feels amazing",
        "Food? Im listening",
        "This is my happy spot",
        "Sigh… life is good",
        "Mmm, in my comfy spot",
        "Chilling out relaxing"
    ],
    "Submissive/Appeasement": [
        "I'm friendly",
        "Please like me",
        "Can we be friends",
        "Pretty please",
        "I'll do whatever you say",
        "Did I do something wrong",
        "Please don't be mad",
        "You're the boss",
        "Im a good dog, I promise",
        "We're cool, right?",
        "I'm sorry if I messed up",
        "Just checking we're ok"
    ],
    "Curiosity/Alertness": [
        "Whatcha Doing?",
        "Who's there?",
        "What was that?",
        "Did you hear that too?",
        "New dog? Must check",
        "Something new here…",
        "I smell something….",
        "This is overwhelming",
        "Too much happening",
        "I don't like this place"
    ],
    "Stressed": [
        "I need space",
        "Who are you?",
        "What are you doing?",
        "I don't feel good",
        "I don't like this",
        "I don't want to",
        "I've had enough",
        "Please don't leave",
        "This is overwhelming",
        "Get me out of here!"
    ],
    "Fearful/Anxious": [
        "What are we doing?",
        "Go away",
        "This scares me",
        "Please don't leave",
        "Something's wrong here",
        "What was that?!",
        "I'm scared",
        "I don't trust you!",
        "Too many strangers",
        "I don't know about you",
        "You look scary",
        "I'm intimidated"
    ],
    "Aggressive/Threatening": [
        "Go away",
        "This is mine, don't touch",
        "I feel threatened",
        "I'm the boss around here",
        "Final warning, buddy",
        "I'll defend what's mine",
        "Don't touch my human",
        "Cross this line. I dare.",
        "Don't test me",
        "I've warned you twice",
        "Don't think about it."
    ]
}

def ensure_directories():
    """Ensure all required directories exist with prevention of duplicate paths"""
    # Track created paths to avoid duplicates
    created_paths = set()
    
    # Create main data directories
    for name, dir_path in DATA_DIRS.items():
        # Skip S3 paths
        if isinstance(dir_path, str) and dir_path.startswith('s3://'):
            continue
            
        # Skip if we've already created this path or its parent
        if any(dir_path.startswith(existing) for existing in created_paths):
            continue
            
        # Create directory and add to tracked paths
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        created_paths.add(dir_path)
    
    # Create subdirectories for processed data - only if they don't exist
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(DATA_DIRS['processed'], split)
        if split_path not in created_paths:
            for subdir in ['images', 'annotations']:
                path = os.path.join(split_path, subdir)
                Path(path).mkdir(parents=True, exist_ok=True)
                created_paths.add(path)
    
    print(f"Directory structure created successfully! Created {len(created_paths)} directories.")

def get_aws_credentials():
    """Get AWS credentials from various sources"""
    # Try Colab secrets first
    try:
        from google.colab import userdata
        aws_access_key = userdata.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = userdata.get('AWS_SECRET_ACCESS_KEY')
        if aws_access_key and aws_secret_key:
            return aws_access_key, aws_secret_key
    except (ImportError, Exception):
        pass
    
    # Try environment variables next
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if aws_access_key and aws_secret_key:
        return aws_access_key, aws_secret_key
    
    # Try AWS config in this file
    if 'aws_access_key_id' in AWS_CONFIG and 'aws_secret_access_key' in AWS_CONFIG:
        return AWS_CONFIG['aws_access_key_id'], AWS_CONFIG['aws_secret_access_key']
    
    # Return None if no credentials found
    return None, None

def save_config_to_file():
    """Save configuration to a JSON file for reference"""
    import pandas as pd
    config_file = os.path.join(DATA_ROOT, 'config.json')
    config = {
        'project_root': PROJECT_ROOT,
        'gdrive': {
            'enabled': GDRIVE_ENABLED,
            'root': GDRIVE_ROOT
        },
        'data_dirs': {k: v for k, v in DATA_DIRS.items() if not (isinstance(v, str) and v.startswith('s3://'))},
        'aws': {k: v for k, v in DATA_DIRS.items() if isinstance(v, str) and v.startswith('s3://')},
        'emotion_mapping': EMOTION_MAPPING,
        'updated_at': pd.Timestamp.now().isoformat()
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")

# Setup AWS credentials from Colab secrets if available
def setup_aws_from_secrets():
    """Setup AWS credentials from Colab secrets"""
    aws_access_key, aws_secret_key = get_aws_credentials()
    if aws_access_key and aws_secret_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        if 'region' in AWS_CONFIG:
            os.environ['AWS_DEFAULT_REGION'] = AWS_CONFIG['region']
        return True
    return False

# If run directly, ensure directories exist
if __name__ == "__main__":
    import pandas as pd
    ensure_directories()
    save_config_to_file()
    setup_aws_from_secrets()
    print("Configuration initialized successfully!")