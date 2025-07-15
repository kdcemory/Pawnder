import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def analyze_video_lengths(video_folder):
    """
    Analyze all videos in a folder and subfolders recursively
    """
    
    video_stats = []
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    print("Analyzing videos...")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(video_folder):
        for video_file in files:
            if any(video_file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, video_file)
                
                try:
                    # Open video
                    cap = cv2.VideoCapture(video_path)
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # Get file size in MB
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    
                    video_stats.append({
                        'filename': video_file,
                        'full_path': video_path,
                        'subfolder': os.path.basename(root),
                        'duration_seconds': duration,
                        'frame_count': frame_count,
                        'fps': fps,
                        'file_size_mb': file_size_mb,
                        'frames_per_second': frame_count / duration if duration > 0 else 0
                    })
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    return pd.DataFrame(video_stats)

def categorize_videos(df, short_threshold=10, long_threshold=30):
    """
    Categorize videos by length and provide recommendations
    """
    
    # Add categories
    df['category'] = 'medium'
    df.loc[df['duration_seconds'] < short_threshold, 'category'] = 'short'
    df.loc[df['duration_seconds'] > long_threshold, 'category'] = 'long'
    
    # Add recommendations
    df['recommendation'] = 'keep_sample'  # Default for medium
    df.loc[df['category'] == 'short', 'recommendation'] = 'keep_all'
    df.loc[df['category'] == 'long', 'recommendation'] = 'remove'
    
    # Calculate sampling rates
    df['suggested_sampling_rate'] = 1  # Default
    df.loc[df['category'] == 'short', 'suggested_sampling_rate'] = 5  # Every 5th frame
    df.loc[df['category'] == 'medium', 'suggested_sampling_rate'] = 10  # Every 10th frame
    df.loc[df['category'] == 'long', 'suggested_sampling_rate'] = 0  # Remove
    
    return df

def print_analysis_summary(df):
    """
    Print detailed analysis summary
    """
    
    print("\n" + "="*60)
    print("VIDEO DATASET ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal videos analyzed: {len(df)}")
    print(f"Total duration: {df['duration_seconds'].sum():.1f} seconds ({df['duration_seconds'].sum()/60:.1f} minutes)")
    print(f"Total frames: {df['frame_count'].sum():,}")
    print(f"Total file size: {df['file_size_mb'].sum():.1f} MB")
    
    print(f"\nDuration Statistics:")
    print(f"  Mean duration: {df['duration_seconds'].mean():.1f} seconds")
    print(f"  Median duration: {df['duration_seconds'].median():.1f} seconds")
    print(f"  Min duration: {df['duration_seconds'].min():.1f} seconds")
    print(f"  Max duration: {df['duration_seconds'].max():.1f} seconds")
    
    print(f"\nFrame Count Statistics:")
    print(f"  Mean frames: {df['frame_count'].mean():.0f}")
    print(f"  Median frames: {df['frame_count'].median():.0f}")
    print(f"  Min frames: {df['frame_count'].min()}")
    print(f"  Max frames: {df['frame_count'].max()}")
    
    # Category breakdown
    category_summary = df.groupby('category').agg({
        'filename': 'count',
        'duration_seconds': ['sum', 'mean'],
        'frame_count': 'sum'
    }).round(1)
    
    print(f"\nCategory Breakdown:")
    print(f"  Short videos (<10s): {len(df[df['category'] == 'short'])} videos")
    print(f"  Medium videos (10-30s): {len(df[df['category'] == 'medium'])} videos") 
    print(f"  Long videos (>30s): {len(df[df['category'] == 'long'])} videos")
    
    # Recommendation summary
    recommendations = df['recommendation'].value_counts()
    print(f"\nRecommendations:")
    for rec, count in recommendations.items():
        print(f"  {rec.replace('_', ' ').title()}: {count} videos")
    
    # Calculate frame reduction
    original_frames = df['frame_count'].sum()
    
    # Calculate frames after filtering
    keep_all_frames = df[df['recommendation'] == 'keep_all']['frame_count'].sum()
    keep_sample_frames = (df[df['recommendation'] == 'keep_sample']['frame_count'] / 
                         df[df['recommendation'] == 'keep_sample']['suggested_sampling_rate']).sum()
    
    final_frames = keep_all_frames + keep_sample_frames
    
    print(f"\nFrame Reduction Impact:")
    print(f"  Original frames: {original_frames:,}")
    print(f"  Frames after filtering: {final_frames:,.0f}")
    print(f"  Reduction: {((original_frames - final_frames) / original_frames * 100):.1f}%")

def create_visualizations(df):
    """
    Create helpful visualizations
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Duration histogram
    axes[0,0].hist(df['duration_seconds'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=10, color='red', linestyle='--', label='Short threshold (10s)')
    axes[0,0].axvline(x=30, color='orange', linestyle='--', label='Long threshold (30s)')
    axes[0,0].set_xlabel('Duration (seconds)')
    axes[0,0].set_ylabel('Number of Videos')
    axes[0,0].set_title('Video Duration Distribution')
    axes[0,0].legend()
    
    # Frame count histogram
    axes[0,1].hist(df['frame_count'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Frame Count')
    axes[0,1].set_ylabel('Number of Videos')
    axes[0,1].set_title('Frame Count Distribution')
    
    # Category pie chart
    category_counts = df['category'].value_counts()
    axes[1,0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[1,0].set_title('Video Categories')
    
    # Recommendation pie chart
    rec_counts = df['recommendation'].value_counts()
    axes[1,1].pie(rec_counts.values, labels=[r.replace('_', ' ').title() for r in rec_counts.index], autopct='%1.1f%%')
    axes[1,1].set_title('Recommendations')
    
    plt.tight_layout()
    plt.show()

def save_filtering_results(df, output_file='video_analysis_results.csv'):
    """
    Save the analysis results to CSV for further processing
    """
    
    # Sort by recommendation and duration
    df_sorted = df.sort_values(['recommendation', 'duration_seconds'])
    df_sorted.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save separate lists for easy processing
    keep_all = df[df['recommendation'] == 'keep_all']['filename'].tolist()
    keep_sample = df[df['recommendation'] == 'keep_sample']['filename'].tolist()
    remove = df[df['recommendation'] == 'remove']['filename'].tolist()
    
    # Save as JSON for easy loading
    filtering_lists = {
        'keep_all': keep_all,
        'keep_sample': keep_sample,
        'remove': remove
    }
    
    with open('video_filtering_lists.json', 'w') as f:
        json.dump(filtering_lists, f, indent=2)
    
    print("Filtering lists saved to: video_filtering_lists.json")

def main():
    """
    Main function to run the complete analysis
    """
    
    # Get video folder path
    video_folder = r"C:\Users\thepf\pawnder\Data\Raw\Videos"

    if not os.path.exists(video_folder):
        print(f"Error: Folder not found: {video_folder}")
        return
    
    # Analyze videos
    df = analyze_video_lengths(video_folder)
    
    if df.empty:
        print("No videos found in the specified folder!")
        return
    
    # Categorize videos
    df = categorize_videos(df)
    
    # Print analysis
    print_analysis_summary(df)
    
    # Create visualizations
    try:
        create_visualizations(df)
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    # Save results
    save_filtering_results(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()