import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def analyze_dataset_quality(base_dir="C:/Users/kelly/Documents/GitHub/Pawnder"):
    """Analyze the quality and distribution of your dataset"""
    
    # Load the annotations
    processed_dir = os.path.join(base_dir, "Data/processed")
    
    splits = ['train', 'validation', 'test']
    all_data = []
    
    for split in splits:
        csv_path = os.path.join(processed_dir, split, "annotations.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['split'] = split
            all_data.append(df)
            print(f"{split.capitalize()} set: {len(df)} samples")
    
    if not all_data:
        print("No annotation files found!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 1. Class Distribution Analysis
    print("\n=== CLASS DISTRIBUTION ===")
    emotion_counts = combined_df['primary_emotion'].value_counts()
    print(emotion_counts)
    
    # Calculate class imbalance ratio
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 3:
        print("⚠️  HIGH CLASS IMBALANCE DETECTED!")
        print("This likely explains your low accuracy.")
    
    # 2. Split Distribution
    print(f"\n=== SPLIT DISTRIBUTION ===")
    for split in splits:
        split_data = combined_df[combined_df['split'] == split]
        if len(split_data) > 0:
            print(f"\n{split.capitalize()}:")
            split_emotions = split_data['primary_emotion'].value_counts()
            for emotion, count in split_emotions.items():
                print(f"  {emotion}: {count}")
    
    # 3. Source Distribution
    print(f"\n=== SOURCE DISTRIBUTION ===")
    if 'source' in combined_df.columns:
        source_counts = combined_df['source'].value_counts()
        print(source_counts)
    
    # 4. Behavior Features Analysis
    behavior_cols = [col for col in combined_df.columns if col.startswith('behavior_')]
    print(f"\n=== BEHAVIOR FEATURES ===")
    print(f"Found {len(behavior_cols)} behavior columns")
    
    if behavior_cols:
        # Check how many samples have behavior data
        behavior_data_count = combined_df[behavior_cols].notna().any(axis=1).sum()
        print(f"Samples with behavior data: {behavior_data_count}/{len(combined_df)} ({behavior_data_count/len(combined_df)*100:.1f}%)")
        
        if behavior_data_count < len(combined_df) * 0.5:
            print("⚠️  MISSING BEHAVIOR DATA!")
            print("Many samples lack behavioral features - this reduces model performance.")
    
    # 5. Visualization
    plt.figure(figsize=(15, 10))
    
    # Class distribution
    plt.subplot(2, 2, 1)
    emotion_counts.plot(kind='bar')
    plt.title('Emotion Class Distribution')
    plt.xticks(rotation=45)
    
    # Split distribution
    plt.subplot(2, 2, 2)
    split_counts = combined_df['split'].value_counts()
    split_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Train/Val/Test Split')
    
    # Source distribution
    if 'source' in combined_df.columns:
        plt.subplot(2, 2, 3)
        source_counts.plot(kind='bar')
        plt.title('Data Source Distribution')
        plt.xticks(rotation=45)
    
    # Behavior data availability
    if behavior_cols:
        plt.subplot(2, 2, 4)
        has_behavior = combined_df[behavior_cols].notna().any(axis=1)
        behavior_dist = has_behavior.value_counts()
        behavior_dist.plot(kind='pie', labels=['Missing Behavior', 'Has Behavior'], autopct='%1.1f%%')
        plt.title('Behavior Data Availability')
    
    plt.tight_layout()
    plt.show()
    
    return combined_df

# Run the analysis
df = analyze_dataset_quality()
