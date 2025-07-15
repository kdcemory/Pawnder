#!/usr/bin/env python3
"""
Behavior Matrix Fix Script
Save this as 'fix_behavior_matrix.py' and run with: python fix_behavior_matrix.py
"""

import os
import json
import pandas as pd

def find_primary_matrix():
    """Find the primary behavior matrix file"""
    possible_paths = [
        "primary_behavior_matrix.json",
        "Data/Matrix/primary_behavior_matrix.json",
        "Data/matrix/primary_behavior_matrix.json",
        "C:/Users/thepf/pawnder/Data/Matrix/primary_behavior_matrix.json",
        # Add more paths as needed
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, list current directory contents
    print("Primary behavior matrix not found. Current directory contents:")
    for item in os.listdir('.'):
        if 'matrix' in item.lower() or 'json' in item.lower():
            print(f"  {item}")
    
    return None

def main():
    print("="*60)
    print("BEHAVIOR MATRIX FIX SCRIPT")
    print("="*60)
    
    # Find the matrix file
    matrix_path = find_primary_matrix()
    
    if not matrix_path:
        print("❌ Could not find primary_behavior_matrix.json")
        print("Please ensure the file is in the current directory or update the path.")
        return
    
    print(f"✓ Found matrix file: {matrix_path}")
    
    try:
        # Load the matrix
        with open(matrix_path, 'r') as f:
            matrix = json.load(f)
        
        print(f"✓ Loaded matrix with {len(matrix['behavioral_states'])} states")
        
        # Extract emotions
        emotions = [state['name'] for state in matrix['behavioral_states']]
        print(f"✓ Emotions: {emotions}")
        
        # Convert behaviors
        behaviors = {}
        state_map = {state['id']: state['name'] for state in matrix['behavioral_states']}
        
        for category in matrix['behavior_categories']:
            for behavior in category['behaviors']:
                emotion_mapping = {}
                for state_id, value in behavior['state_mapping'].items():
                    if state_id in state_map:
                        emotion_mapping[state_map[state_id]] = value
                
                behaviors[behavior['id']] = emotion_mapping
                behaviors[behavior['name']] = emotion_mapping
        
        print(f"✓ Converted {len(behaviors)} behaviors")
        
        # Save corrected matrix
        corrected_matrix = {
            "emotions": emotions,
            "behaviors": behaviors
        }
        
        # Save to current directory
        output_path = "corrected_behavior_matrix.json"
        with open(output_path, 'w') as f:
            json.dump(corrected_matrix, f, indent=2)
        
        print(f"✅ SUCCESS! Saved corrected matrix to: {output_path}")
        
        # Show sample mappings
        print("\nSample behavior mappings:")
        for i, (behavior, mapping) in enumerate(list(behaviors.items())[:3]):
            print(f"\n{behavior}:")
            for emotion, value in mapping.items():
                indicator = "✓" if value == 1 else "○"
                print(f"  {indicator} {emotion}")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Copy this corrected_behavior_matrix.json to your project")
        print("2. Update your model to use this corrected matrix")
        print("3. Retrain your model - you should see much better performance!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error processing matrix: {e}")

if __name__ == "__main__":
    main()
