"""
Directory and Dataset Diagnosis Script for Pawnder App

This script analyzes your directory structure and dataset
to identify any issues before training.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Base directories to check
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/Pawnder"
PROCESSED_DIR = os.path.join(BASE_DIR, "Data/processed")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
MATRIX_DIR = os.path.join(BASE_DIR, "Data/Matrix")


# Class directories to check
CLASS_DIRS = {
    "train": os.path.join(PROCESSED_DIR, "train_by_class"),
    "validation": os.path.join(PROCESSED_DIR, "validation_by_class"),
    "test": os.path.join(PROCESSED_DIR, "test_by_class")
}

# Expected emotion classes based on Primary Behavior Matrix
EXPECTED_CLASSES = [
    "Happy", 
    "Relaxed", 
    "Submissive", 
    "Curiosity", 
    "Stressed", 
    "Fearful", 
    "Aggressive"
]

def check_base_directories():
    """Check if base directories exist"""
    print("\n=== Base Directory Check ===")
    
    dirs_to_check = {
        "Base directory": BASE_DIR,
        "Processed directory": PROCESSED_DIR,
        "Models directory": MODELS_DIR,
        "Matrix directory": MATRIX_DIR
    }
    
    all_exist = True
    for name, path in dirs_to_check.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓ Exists' if exists else 'X Missing'} -> {path}")
        all_exist = all_exist and exists
    
    return all_exist

def check_data_directories():
    """Check if data split directories exist and have proper structure"""
    print("\n=== Data Directory Structure Check ===")
    
    all_valid = True
    for split_name, split_dir in CLASS_DIRS.items():
        print(f"\n{split_name.capitalize()} directory: {split_dir}")
        
        if not os.path.exists(split_dir):
            print(f"  X Directory does not exist!")
            all_valid = False
            continue
        
        # Check for class subdirectories
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        
        print(f"  Found {len(class_dirs)} class directories")
        
        # Check for expected emotion classes
        missing_classes = set(EXPECTED_CLASSES) - set(class_dirs)
        if missing_classes:
            print(f"  ! Warning: Missing expected classes: {', '.join(missing_classes)}")
            all_valid = False
        
        # Check each class directory for images
        total_images = 0
        class_stats = {}
        
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            class_stats[class_name] = len(image_files)
            total_images += len(image_files)
            
            # Check if class directory is empty
            if len(image_files) == 0:
                print(f"  ! Warning: Class '{class_name}' has no images")
                all_valid = False
        
        # Print class statistics
        print(f"  Total images: {total_images}")
        if total_images == 0:
            print(f"  X No images found in any class directory!")
            all_valid = False
            continue
        
        print("  Class distribution:")
        for class_name, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_images * 100 if total_images > 0 else 0
            print(f"    {class_name}: {count} images ({percentage:.1f}%)")
    
    return all_valid

def check_primary_matrix():
    """Check if Primary Behavior Matrix file exists and is accessible"""
    print("\n=== Primary Behavior Matrix Check ===")
    
    # Possible matrix file locations
    matrix_files = [
        os.path.join(MATRIX_DIR, "Primary Behavior Matrix.xlsx"),
        os.path.join(MATRIX_DIR, "primary_behavior_matrix.json"),
        os.path.join(BASE_DIR, "Data/Matrix/Primary Behavior Matrix.xlsx"),
        os.path.join(BASE_DIR, "Data/Matrix/primary_behavior_matrix.json"),
        os.path.join(BASE_DIR, "data/Matrix/Primary Behavior Matrix.xlsx"),
        os.path.join(BASE_DIR, "data/Matrix/primary_behavior_matrix.json")
    ]
    
    found = False
    
    for matrix_path in matrix_files:
        if os.path.exists(matrix_path):
            print(f"✓ Found matrix file: {matrix_path}")
            found = True
            
            # Try to analyze the matrix file
            try:
                if matrix_path.endswith('.json'):
                    with open(matrix_path, 'r') as f:
                        matrix_data = json.load(f)
                    print(f"  Successfully loaded JSON matrix with {len(matrix_data)} entries")
                    
                    # Check for behavior columns
                    behavior_cols = [k for k in matrix_data.keys() 
                                   if isinstance(k, str) and k.startswith('behavior_')]
                    print(f"  Found {len(behavior_cols)} behavior features")
                    
                elif matrix_path.endswith('.xlsx'):
                    matrix_df = pd.read_excel(matrix_path)
                    print(f"  Successfully loaded Excel matrix with {len(matrix_df)} rows and {len(matrix_df.columns)} columns")
                    
                    # Check for behavior columns
                    behavior_cols = [col for col in matrix_df.columns 
                                   if isinstance(col, str) and col.startswith('behavior_')]
                    print(f"  Found {len(behavior_cols)} behavior features")
            
            except Exception as e:
                print(f"  ! Error analyzing matrix file: {str(e)}")
                found = False
    
    if not found:
        print("X Could not find or load Primary Behavior Matrix file")
        print("  Searched in the following locations:")
        for path in matrix_files:
            print(f"  - {path}")
    
    return found

def check_annotations():
    """Check for annotations file and analyze content"""
    print("\n=== Annotations File Check ===")
    
    # Possible annotations file locations
    annotation_files = [
        os.path.join(PROCESSED_DIR, "combined_annotations.json"),
        os.path.join(PROCESSED_DIR, "annotations.json"),
        os.path.join(BASE_DIR, "Data/processed/combined_annotations.json"),
        os.path.join(BASE_DIR, "Data/processed/annotations.json")
    ]
    
    found = False
    
    for annotations_path in annotation_files:
        if os.path.exists(annotations_path):
            print(f"✓ Found annotations file: {annotations_path}")
            found = True
            
            # Try to analyze the annotations file
            try:
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                
                print(f"  Successfully loaded annotations with {len(annotations)} entries")
                
                # Check a sample annotation
                if annotations:
                    first_key = next(iter(annotations))
                    print(f"  Sample annotation (key: {first_key}):")
                    first_entry = annotations[first_key]
                    
                    # Check for emotion data
                    if "emotions" in first_entry and "primary_emotion" in first_entry["emotions"]:
                        print(f"  ✓ Contains emotion data")
                        
                        # Count emotions
                        emotion_counts = defaultdict(int)
                        for img_id, data in annotations.items():
                            if "emotions" in data and "primary_emotion" in data["emotions"]:
                                emotion = data["emotions"]["primary_emotion"]
                                emotion_counts[emotion] += 1
                        
                        print(f"  Emotion distribution in annotations:")
                        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                            percentage = count / len(annotations) * 100
                            print(f"    {emotion}: {count} ({percentage:.1f}%)")
                    else:
                        print(f"  ! No emotion data found in annotations")
                    
                    # Check for behavior features
                    behavior_cols = [k for k in first_entry.keys() 
                                  if isinstance(k, str) and k.startswith('behavior_')]
                    if behavior_cols:
                        print(f"  ✓ Contains {len(behavior_cols)} behavior features: {', '.join(behavior_cols[:5])}{'...' if len(behavior_cols) > 5 else ''}")
                    else:
                        print(f"  ! No behavior features found in annotations")
            
            except Exception as e:
                print(f"  ! Error analyzing annotations file: {str(e)}")
                found = False
    
    if not found:
        print("X Could not find or load annotations file")
        print("  Searched in the following locations:")
        for path in annotation_files:
            print(f"  - {path}")
    
    return found

def visualize_class_distribution():
    """Visualize class distribution across splits"""
    print("\n=== Visualizing Class Distribution ===")
    
    # Collect class counts for each split
    all_stats = {}
    
    for split_name, split_dir in CLASS_DIRS.items():
        if not os.path.exists(split_dir):
            continue
        
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        
        class_stats = {}
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_stats[class_name] = len(image_files)
        
        all_stats[split_name] = class_stats
    
    # Generate plot
    if all_stats:
        try:
            # Get all unique classes
            all_classes = set()
            for split_stats in all_stats.values():
                all_classes.update(split_stats.keys())
            
            # Sort classes by standard order if possible
            if set(EXPECTED_CLASSES).issubset(all_classes):
                sorted_classes = EXPECTED_CLASSES.copy()
                for cls in sorted(all_classes):
                    if cls not in sorted_classes:
                        sorted_classes.append(cls)
            else:
                sorted_classes = sorted(all_classes)
            
            # Create plot data
            splits = list(all_stats.keys())
            x = range(len(sorted_classes))
            width = 0.8 / len(splits)
            
            fig, ax = plt.figure(figsize=(12, 6)), plt.axes()
            
            for i, split_name in enumerate(splits):
                split_stats = all_stats[split_name]
                counts = [split_stats.get(cls, 0) for cls in sorted_classes]
                offset = width * (i - len(splits)/2 + 0.5)
                ax.bar([pos + offset for pos in x], counts, width, label=split_name.capitalize())
            
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_classes, rotation=45, ha='right')
            ax.set_ylabel('Number of Images')
            ax.set_title('Class Distribution Across Data Splits')
            ax.legend()
            
            plt.tight_layout()
            
            # Save the figure
            output_dir = os.path.join(BASE_DIR, "Data/diagnostics")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "class_distribution.png")
            plt.savefig(output_path, dpi=300)
            
            print(f"✓ Visualization saved to {output_path}")
            print("  You can view it in your file browser.")
            
            # Display in notebook if running in one
            plt.show()
            
        except Exception as e:
            print(f"! Error creating visualization: {str(e)}")

def check_for_common_issues():
    """Check for common issues that could cause training problems"""
    print("\n=== Common Issues Check ===")
    
    # 1. Check for class imbalance
    print("\nChecking for class imbalance...")
    for split_name, split_dir in CLASS_DIRS.items():
        if not os.path.exists(split_dir):
            continue
        
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        
        class_counts = {}
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[class_name] = len(image_files)
        
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            
            if min_count == 0:
                print(f"  ! {split_name.capitalize()} set has empty classes")
            
            # Check for severe imbalance (>10x difference)
            if min_count > 0 and max_count / min_count > 10:
                print(f"  ! {split_name.capitalize()} set has severe class imbalance:")
                print(f"    Largest class: {max(class_counts.items(), key=lambda x: x[1])[0]} with {max_count} images")
                print(f"    Smallest class: {min(class_counts.items(), key=lambda x: x[1])[0]} with {min_count} images")
                print(f"    Ratio: {max_count/min_count:.1f}x")
                print("    Consider balancing classes or using class weights during training")
    
    # 2. Check for inconsistent class names across splits
    print("\nChecking for inconsistent class names...")
    all_class_sets = {}
    for split_name, split_dir in CLASS_DIRS.items():
        if not os.path.exists(split_dir):
            continue
        
        class_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        all_class_sets[split_name] = set(class_dirs)
    
    if len(all_class_sets) > 1:
        all_consistent = True
        reference_set = next(iter(all_class_sets.values()))
        
        for split_name, class_set in all_class_sets.items():
            if class_set != reference_set:
                all_consistent = False
                missing = reference_set - class_set
                extra = class_set - reference_set
                
                if missing:
                    print(f"  ! {split_name.capitalize()} set is missing classes: {', '.join(missing)}")
                if extra:
                    print(f"  ! {split_name.capitalize()} set has extra classes: {', '.join(extra)}")
        
        if all_consistent:
            print("  ✓ All splits have consistent class names")
    
    # 3. Check for very small datasets
    print("\nChecking for small dataset size...")
    for split_name, split_dir in CLASS_DIRS.items():
        if not os.path.exists(split_dir):
            continue
        
        total_images = 0
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(image_files)
        
        if total_images < 100:
            print(f"  ! {split_name.capitalize()} set has only {total_images} images")
            print("    Consider adding more data or using data augmentation")
        elif total_images < 500:
            print(f"  ! {split_name.capitalize()} set has {total_images} images")
            print("    Dataset is small for deep learning, consider adding more data or extensive augmentation")

def main():
    """Main function to run directory and dataset diagnosis"""
    print("="*80)
    print("Directory and Dataset Diagnosis for Pawnder App")
    print("="*80)
    
    all_checks_passed = True
    
    # Check base directories
    base_dirs_ok = check_base_directories()
    all_checks_passed = all_checks_passed and base_dirs_ok
    
    # Check data directories and classes
    data_dirs_ok = check_data_directories()
    all_checks_passed = all_checks_passed and data_dirs_ok
    
    # Check Primary Behavior Matrix
    matrix_ok = check_primary_matrix()
    all_checks_passed = all_checks_passed and matrix_ok
    
    # Check annotations
    annotations_ok = check_annotations()
    all_checks_passed = all_checks_passed and annotations_ok
    
    # Check for common issues
    check_for_common_issues()
    
    # Visualize class distribution
    if data_dirs_ok:
        visualize_class_distribution()
    
    print("\n" + "="*80)
    print("Diagnosis Summary:")
    if all_checks_passed:
        print("✓ All basic checks passed. Your directory structure appears ready for training.")
    else:
        print("! Some checks failed. Please review the issues above before training.")
    print("="*80)

if __name__ == "__main__":
    main()