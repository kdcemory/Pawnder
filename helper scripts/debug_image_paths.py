"""
Debug Script to Identify Image Path Issues

Run this script to diagnose where your images are located and fix the path resolution.
"""

import os
import pandas as pd
import glob

def debug_image_locations(base_dir="C:\\Users\\kelly\\Documents\\GitHub\\Pawnder"):
    """
    Debug function to find where images are actually located
    """
    print("🔍 DEBUGGING IMAGE LOCATIONS")
    print("="*60)
    
    # 1. Check the CSV to see what image paths look like
    csv_path = os.path.join(base_dir, "Data", "processed", "train", "annotations.csv")
    
    if os.path.exists(csv_path):
        print(f"✅ Found CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"📊 CSV has {len(df)} rows")
        
        # Sample some image paths
        print("\n📁 Sample image paths from CSV:")
        for i, path in enumerate(df['image_path'].head(10)):
            print(f"  {i+1}. {path}")
    else:
        print(f"❌ CSV not found: {csv_path}")
        return
    
    # 2. Check different possible image directories
    print("\n📂 CHECKING POSSIBLE IMAGE DIRECTORIES:")
    
    possible_dirs = [
        os.path.join(base_dir, "Data", "processed", "train", "images"),
        os.path.join(base_dir, "Data", "processed", "all_frames"),
        os.path.join(base_dir, "Data", "processed", "combined_frames"),
        os.path.join(base_dir, "Data", "raw", "all_frames"),
        os.path.join(base_dir, "processed", "all_frames"),
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"✅ {dir_path}")
                print(f"   📊 Contains {len(image_files)} image files")
                if image_files:
                    print(f"   📄 Sample files:")
                    for f in image_files[:3]:
                        print(f"      - {f}")
            except Exception as e:
                print(f"❌ {dir_path} - Error: {e}")
        else:
            print(f"❌ {dir_path} - Does not exist")
    
    # 3. Search for image directories recursively
    print(f"\n🔍 SEARCHING FOR IMAGE DIRECTORIES IN {base_dir}:")
    
    for root, dirs, files in os.walk(base_dir):
        # Look for directories with many image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 100:  # Directories with substantial number of images
            print(f"📁 Found image directory: {root}")
            print(f"   📊 Contains {len(image_files)} images")
            
            # Show sample files
            print(f"   📄 Sample files:")
            for f in image_files[:3]:
                print(f"      - {f}")
    
    # 4. Try to match CSV paths with actual files
    print(f"\n🔗 CHECKING PATH MATCHING:")
    
    # Get sample paths from CSV
    sample_paths = df['image_path'].head(5).tolist()
    
    # Find the most likely image directory
    best_dir = None
    max_matches = 0
    
    for root, dirs, files in os.walk(base_dir):
        if len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) > 100:
            matches = 0
            for csv_path in sample_paths:
                filename = os.path.basename(csv_path)
                if filename in files:
                    matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_dir = root
            
            print(f"📁 {root}: {matches}/{len(sample_paths)} matches")
    
    if best_dir:
        print(f"\n🎯 BEST MATCH: {best_dir}")
        print(f"   ✅ Matches {max_matches}/{len(sample_paths)} sample files")
        
        # Test actual file access
        print(f"\n🧪 TESTING FILE ACCESS:")
        for csv_path in sample_paths:
            filename = os.path.basename(csv_path)
            full_path = os.path.join(best_dir, filename)
            
            if os.path.exists(full_path):
                print(f"   ✅ {filename} - Found")
            else:
                print(f"   ❌ {filename} - Missing")
                
                # Try to find similar files
                similar = [f for f in os.listdir(best_dir) if filename[:10] in f]
                if similar:
                    print(f"      🔍 Similar files: {similar[:3]}")
    
    return best_dir

def fix_training_script_paths(base_dir, correct_image_dir):
    """
    Generate code to fix the image path resolution in the training script
    """
    print(f"\n🔧 SUGGESTED FIX:")
    print("="*60)
    print("Replace the image directory section in your prepare_data_for_training method with:")
    print()
    
    # Create the fix code with proper string formatting
    print("# FIXED IMAGE DIRECTORY RESOLUTION")
    print("# Replace the image_directories section with this:")
    print()
    print(f'images_dir = r"{correct_image_dir}"')
    print('print(f"✅ Using fixed images directory: {images_dir}")')
    print()
    print("# Or if you want to handle different splits:")
    print("if split_name == 'train':")
    print(f'    images_dir = r"{correct_image_dir}"')
    print("else:")
    print("    # For validation and test, try the same pattern")
    alt_dir = correct_image_dir.replace('train', '{split_name}') if 'train' in correct_image_dir else correct_image_dir
    print(f'    images_dir = r"{alt_dir}".format(split_name=split_name)')
    print("    if not os.path.exists(images_dir):")
    print(f'        images_dir = r"{correct_image_dir}"  # Fallback to main directory')
    print()
    print('print(f"Using images directory: {images_dir}")')

if __name__ == "__main__":
    # Run the debug
    base_dir = "C:\\Users\\kelly\\Documents\\GitHub\\Pawnder"
    best_dir = debug_image_locations(base_dir)
    
    if best_dir:
        fix_training_script_paths(base_dir, best_dir)
    else:
        print("\n❌ Could not find image directory!")
        print("Please check if images were properly copied during data processing.")
