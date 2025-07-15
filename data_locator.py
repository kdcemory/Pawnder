"""
Data Locator Script for Pawnder App

This script finds the actual data locations being used.
"""

import os
import glob
from pathlib import Path

# Potential base directories
POTENTIAL_DIRS = [
    "/content/drive/MyDrive/Colab Notebooks/Pawnder",
    "/content/drive/MyDrive/Pawnder",
    "/content/drive/My Drive/Pawnder",
    "/content/Pawnder",
    "/content/drive/MyDrive/Colab Notebooks",
    "/content"
]

def find_directories():
    """Find the most likely data directories"""
    print("\n=== Looking for Pawnder directories ===")
    
    found_dirs = []
    
    for base_dir in POTENTIAL_DIRS:
        if os.path.exists(base_dir):
            print(f"\nFound potential base directory: {base_dir}")
            
            # Check for data subdirectories
            data_dir = os.path.join(base_dir, "Data")
            if os.path.exists(data_dir):
                print(f"  ✓ Found Data directory: {data_dir}")
                
                # Check for processed data
                processed_dir = os.path.join(data_dir, "processed")
                if os.path.exists(processed_dir):
                    print(f"  ✓ Found processed directory: {processed_dir}")
                    
                    # Check for class directories
                    for split in ['train_by_class', 'validation_by_class', 'test_by_class']:
                        split_dir = os.path.join(processed_dir, split)
                        if os.path.exists(split_dir):
                            print(f"    ✓ Found {split} directory: {split_dir}")
                            class_dirs = [d for d in os.listdir(split_dir) 
                                         if os.path.isdir(os.path.join(split_dir, d))]
                            print(f"      Contains classes: {', '.join(class_dirs[:5])}{'...' if len(class_dirs) > 5 else ''}")
                
                # Also check for direct class dirs
                for split in ['train', 'val', 'validation', 'test']:
                    split_dir = os.path.join(data_dir, split)
                    if os.path.exists(split_dir):
                        print(f"  ✓ Found {split} directory: {split_dir}")
            
            # Look for model directory
            model_dir = os.path.join(base_dir, "Models")
            if os.path.exists(model_dir):
                print(f"  ✓ Found Models directory: {model_dir}")
            
            # Add to found dirs if it seems to contain Pawnder data
            if os.path.exists(data_dir) or os.path.exists(model_dir):
                found_dirs.append(base_dir)
    
    # If no directories found, try searching for specific patterns
    if not found_dirs:
        print("\nNo standard Pawnder directories found. Searching for data patterns...")
        
        # Look for directories that might contain processed data
        for base_dir in POTENTIAL_DIRS:
            if os.path.exists(base_dir):
                # Look for potential data directories
                data_patterns = [
                    os.path.join(base_dir, "**/train_by_class"),
                    os.path.join(base_dir, "**/processed"),
                    os.path.join(base_dir, "**/Data")
                ]
                
                for pattern in data_patterns:
                    matches = glob.glob(pattern, recursive=True)
                    for match in matches:
                        print(f"Found potential data directory: {match}")
                        parent = os.path.dirname(match)
                        if parent not in found_dirs:
                            found_dirs.append(parent)
    
    return found_dirs

def scan_directory(directory, max_depth=3, current_depth=0):
    """Recursively scan a directory to find dataset-like structures"""
    if current_depth > max_depth:
        return
    
    try:
        items = os.listdir(directory)
        
        # Count image files
        image_files = [f for f in items if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) > 10:
            print(f"Found {len(image_files)} images in {directory}")
        
        # Count subdirectories
        subdirs = [d for d in items if os.path.isdir(os.path.join(directory, d))]
        
        # Look for class-like directories
        emotion_classes = ['happy', 'relaxed', 'stressed', 'fearful', 'aggressive']
        class_dirs = [d for d in subdirs if d.lower() in emotion_classes]
        if class_dirs:
            print(f"Found potential class directories in {directory}: {', '.join(class_dirs)}")
        
        # Recursively scan subdirectories
        for subdir in subdirs:
            scan_directory(os.path.join(directory, subdir), max_depth, current_depth + 1)
    
    except (PermissionError, FileNotFoundError):
        # Skip directories we can't access
        pass

def main():
    """Main function to locate data"""
    print("="*80)
    print("Pawnder Data Locator")
    print("="*80)
    
    # Find potential directories
    found_dirs = find_directories()
    
    if found_dirs:
        print("\n=== Recommended Directory Settings ===")
        most_likely_dir = found_dirs[0]
        print(f"BASE_DIR = \"{most_likely_dir}\"")
        print(f"PROCESSED_DIR = \"{os.path.join(most_likely_dir, 'Data/processed')}\"")
        print(f"MODEL_DIR = \"{os.path.join(most_likely_dir, 'Models')}\"")
        
        print("\nTo modify the scripts, update the base_dir variable in each file to:")
        print(f"self.base_dir = \"{most_likely_dir}\"")
    else:
        print("\nCould not find standard Pawnder directories.")
        print("Let's scan the current directory for dataset-like structures:")
        scan_directory(os.getcwd())

if __name__ == "__main__":
    main()