# split_dataset.py
# Script to split data into train, validation, and test sets

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_output, val_output, test_output, 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  stratify=True, random_state=42):
    """Split dataset into train, validation and test sets"""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found")
        return None
    
    # Check that ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1 (got {train_ratio + val_ratio + test_ratio})")
        return None
    
    # Load annotations
    with open(input_file, 'r') as f:
        annotations = json.load(f)
    
    # Check for existing split information
    has_splits = False
    split_counts = {"train": 0, "val": 0, "validation": 0, "test": 0}
    
    for img_id, data in annotations.items():
        if "split" in data:
            has_splits = True
            split = data["split"]
            split_counts[split] = split_counts.get(split, 0) + 1
    
    # If annotations already have splits, respect them
    if has_splits and (split_counts["train"] > 0 or split_counts["val"] > 0 or 
                      split_counts["validation"] > 0 or split_counts["test"] > 0):
        print("Found existing split information in annotations, respecting those splits")
        
        train_annotations = {}
        val_annotations = {}
        test_annotations = {}
        
        for img_id, data in annotations.items():
            split = data.get("split", "train")  # Default to train if not specified
            
            if split == "train":
                train_annotations[img_id] = data
            elif split in ["val", "validation"]:
                val_annotations[img_id] = data
            elif split == "test":
                test_annotations[img_id] = data
            else:
                # Unknown split, put in train
                train_annotations[img_id] = data
    else:
        # No splits or insufficient splits, create new ones
        print("Creating new splits based on provided ratios")
        
        # Convert annotations to list of (id, annotation) tuples
        items = list(annotations.items())
        
        # If stratify is True, prepare stratification labels
        if stratify:
            stratify_labels = []
            for img_id, data in items:
                if "emotions" in data and "primary_emotion" in data["emotions"]:
                    label = data["emotions"]["primary_emotion"]
                else:
                    label = "unknown"
                stratify_labels.append(label)
        else:
            stratify_labels = None
        
        # First split: separate training set
        train_items, temp_items = train_test_split(
            items,
            train_size=train_ratio,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Update stratify_labels for second split if needed
        if stratify and len(temp_items) > 0:
            stratify_labels = []
            for img_id, data in temp_items:
                if "emotions" in data and "primary_emotion" in data["emotions"]:
                    label = data["emotions"]["primary_emotion"]
                else:
                    label = "unknown"
                stratify_labels.append(label)
        else:
            stratify_labels = None
        
        # Second split: separate validation and test sets
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_items, test_items = train_test_split(
            temp_items,
            train_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Convert back to dictionaries
        train_annotations = dict(train_items)
        val_annotations = dict(val_items)
        test_annotations = dict(test_items)
        
        # Update split information in annotations
        for img_id in train_annotations:
            train_annotations[img_id]["split"] = "train"
        
        for img_id in val_annotations:
            val_annotations[img_id]["split"] = "validation"
        
        for img_id in test_annotations:
            test_annotations[img_id]["split"] = "test"
    
    # Save split annotations
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    with open(train_output, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    os.makedirs(os.path.dirname(val_output), exist_ok=True)
    with open(val_output, 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    with open(test_output, 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    # Print statistics
    total = len(annotations)
    train_pct = len(train_annotations) / total * 100
    val_pct = len(val_annotations) / total * 100
    test_pct = len(test_annotations) / total * 100
    
    print(f"Dataset split complete:")
    print(f"  Total: {total} annotations")
    print(f"  Train: {len(train_annotations)} ({train_pct:.1f}%)")
    print(f"  Validation: {len(val_annotations)} ({val_pct:.1f}%)")
    print(f"  Test: {len(test_annotations)} ({test_pct:.1f}%)")
    
    return {
        "train": train_annotations,
        "validation": val_annotations,
        "test": test_annotations
    }