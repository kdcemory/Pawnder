"""
Unicode Fix Patch for training_script.py

Apply these changes to your training_script.py to fix Unicode issues.
"""

# 1. Add these imports at the top of your training_script.py file
import unicodedata
import re

# 2. Add these Unicode-safe utility functions
def safe_imread(image_path, flags=cv2.IMREAD_COLOR):
    """
    Safely read an image file that may contain Unicode characters in the path.
    """
    try:
        # First try normal imread
        img = cv2.imread(image_path, flags)
        if img is not None:
            return img
    except Exception:
        pass
    
    # If normal imread fails, try Unicode-safe method
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        if len(img_array) == 0:
            return None
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def sanitize_filename(filename):
    """
    Sanitize a filename by removing or replacing problematic Unicode characters.
    """
    # Normalize Unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Replace problematic characters
    replacements = {
        'ΓÇ»': '_',  # The specific character causing your issue
        'ΓÇ': '_',   # Related character
        'Γ': '_',    # Base character
        '»': '_',    # Angle quote
        '«': '_',    # Angle quote
        '"': '_',    # Smart quotes
        '"': '_',
        ''': '_',    # Smart quotes
        ''': '_',
        '…': '_',    # Ellipsis
        '–': '-',    # En dash
        '—': '-',    # Em dash
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    # Remove any remaining non-ASCII characters that might cause issues
    filename = re.sub(r'[^\x00-\x7F]+', '_', filename)
    
    # Clean up multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    return filename

def find_unicode_safe_path(original_path):
    """
    Find a file even if it has Unicode issues in the filename.
    """
    if os.path.exists(original_path):
        return original_path
    
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    if os.path.exists(dir_path):
        try:
            for file in os.listdir(dir_path):
                # Try exact match first
                if file == filename:
                    return os.path.join(dir_path, file)
                
                # Try sanitized match
                sanitized_filename = sanitize_filename(filename)
                sanitized_file = sanitize_filename(file)
                if sanitized_file == sanitized_filename:
                    return os.path.join(dir_path, file)
                
                # Try partial match for Unicode corruption
                base_name = os.path.splitext(filename)[0]
                file_base = os.path.splitext(file)[0]
                if len(base_name) > 10:  # Only for longer names
                    # Check if the first part matches (before Unicode corruption)
                    clean_base = re.sub(r'[^\x00-\x7F]+', '', base_name)
                    clean_file = re.sub(r'[^\x00-\x7F]+', '', file_base)
                    if clean_base and clean_file and clean_base in clean_file:
                        return os.path.join(dir_path, file)
                        
        except Exception as e:
            print(f"Error searching directory {dir_path}: {e}")
    
    return original_path

# 3. REPLACE the fix_image_path_resolution function with this enhanced version:
def fix_image_path_resolution(image_path, base_dir):
    """
    Enhanced version with Unicode handling.
    """
    import os
    import re
    
    # Video name to folder mapping (your existing mapping)
    VIDEO_FOLDER_MAP = {
        "1": "1", "3": "3", "4": "4", "5": "5", "50": "50", "51": "51", "52": "52", 
        "53": "53", "54": "54", "55": "55", "56": "56", "57": "57", "58": "58", 
        "59": "59", "60": "60", "61": "61", "62": "62", "63": "63", "64": "64", 
        "65": "65", "66": "66", "67": "67", "68": "68", "69": "69", "70": "70", 
        "71": "71", "72": "72", "73": "73", "74": "74", "80": "80", "81": "81",
        "m2-res_854p-7": "17", "excited": "19", "whale eye-2": "20",
        "shibu grin": "21", "shaking": "22", "playbow": "23",
        "piloerection and stiff tail": "24", "look away": "25", "lip licking": "26",
        "relaxedrottie": "27", "stresspain": "28", "happywelcome": "29",
        "bodylanguagepits": "30", "aggressive pit": "31",
        "Screen Recording 2025-03-06 at 9.33.52 AM": "32", "alertdog5": "33",
        "alertdog1": "34", "canine distemper": "35", "canine distemper2": "36",
        "fearanaussiestress": "37", "fearandanxiety": "38", "pancreatitis": "39",
        "relaxed dog": "40", "relaxed dog2": "41", "relaxed dog3": "42",
        "stressed kennel": "43", "stressed vet": "44", "stressedlab": "45",
        "stressedpit": "46", "Curious": "Curious", "head_tilt": "head_tilt",
        "m2-res_360p": "m2-res_360p", "m2-res_480p": "m2-res_480p",
        "m2-res_480p-2": "m2-res_480p-2", "m2-res_532p": "m2-res_532p",
        "m2-res_720p": "m2-res_720p", "m2-res_848p": "m2-res_848p",
        "m2-res_854p-2": "m2-res_854p-2", "m2-res_854p-3": "m2-res_854p-3",
        "m2-res_854p-4": "m2-res_854p-4", "m2-res_854p-5": "m2-res_854p-5",
        "m2-res_854p-6": "m2-res_854p-6", "play bow": "playbow",
        "relaxed": "relaxed", "resource guarding": "resource_guarding_1",
        "resource guarding 2": "resource_guarding_2", "alert dog": "34",
        "alertdog": "34", "head tilt": "head_tilt", "play_bow": "playbow",
        "resource_guarding": "resource_guarding_1", "relaxed_dog": "40",
        "relaxed_dog2": "41", "relaxed_dog3": "42", "stressed_kennel": "43",
        "stressed_vet": "44", "stressed_lab": "45", "stressed_pit": "46",
        "bodylanguage pits": "30", "body language pits": "30",
        "whale_eye-2": "20", "whale-eye-2": "20", "whale_eye": "20", "whale eye": "20"
    }
    
    # First, try Unicode-safe path finding
    safe_path = find_unicode_safe_path(image_path)
    if os.path.exists(safe_path):
        return safe_path
    
    # If the path already exists, return it
    if os.path.exists(image_path):
        return image_path
    
    # Special case for .png vs .PNG extension
    if isinstance(image_path, str) and image_path.lower().endswith('.png'):
        upper_path = image_path[:-4] + '.PNG'
        if os.path.exists(upper_path):
            return upper_path
    
    # Get filename if it's a path
    if isinstance(image_path, str):
        filename = os.path.basename(image_path)
    else:
        return image_path
    
    # Extract video name and frame number
    video_name = None
    frame_number = None
    
    # Try to extract using common patterns
    if "_frame_" in filename:
        parts = filename.split("_frame_")
        video_name = parts[0]
        frame_number = parts[1]
        if "." in frame_number:
            frame_number = frame_number.split(".")[0]
    
    # If we successfully extracted video name and frame number
    if video_name is not None and frame_number is not None:
        folder = VIDEO_FOLDER_MAP.get(video_name)
        
        # If not found directly, try case-insensitive lookup
        if folder is None:
            for k, v in VIDEO_FOLDER_MAP.items():
                if k.lower() == video_name.lower():
                    folder = v
                    break
        
        # If still not found, try partial matching
        if folder is None:
            for k, v in VIDEO_FOLDER_MAP.items():
                if k.lower() in video_name.lower() or video_name.lower() in k.lower():
                    folder = v
                    break
        
        # If we found a folder mapping
        if folder is not None:
            for videos_root in [
                os.path.join(base_dir, "Data", "raw", "Videos"),
                os.path.join(base_dir, "Data", "Raw", "Videos"),
                os.path.join(base_dir, "Data", "processed"),
                os.path.join(base_dir, "Data", "processed", "all_frames")
            ]:
                folder_path = os.path.join(videos_root, folder, "images")
                if os.path.exists(folder_path):
                    # Try exact frame with extensions
                    for ext in ['.PNG', '.png', '.jpg', '.jpeg']:
                        frame_path = os.path.join(folder_path, f"frame_{frame_number}{ext}")
                        safe_frame_path = find_unicode_safe_path(frame_path)
                        if os.path.exists(safe_frame_path):
                            return safe_frame_path
                    
                    # If not found, use modulo mapping
                    try:
                        frame_files = [f for f in os.listdir(folder_path) if f.startswith("frame_")]
                        if frame_files:
                            frame_num = int(frame_number)
                            frame_idx = frame_num % len(frame_files)
                            sorted_frames = sorted(frame_files)
                            return os.path.join(folder_path, sorted_frames[frame_idx])
                    except (ValueError, IndexError):
                        if frame_files:
                            return os.path.join(folder_path, sorted(frame_files)[0])
    
    # Last resort: try Unicode-safe path resolution
    return find_unicode_safe_path(image_path)

# 4. REPLACE the load_image method in DogEmotionWithBehaviors class:
def load_image(self, image_path, img_size=(224, 224)):
    """Load and preprocess an image for inference with Unicode support"""
    try:
        # Use the enhanced path resolution
        resolved_path = fix_image_path_resolution(image_path, self.base_dir)

        # Use Unicode-safe image loading
        img = safe_imread(resolved_path)
        
        if img is None:
            print(f"Image not found or could not be loaded: {resolved_path} (original: {image_path})")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

# 5. UPDATE the DataGenerator class __getitem__ method:
# In the __getitem__ method of your DataGenerator classes, 
# replace this line:
#     img = cv2.imread(resolved_path)
# 
# with this:
#     img = safe_imread(resolved_path)

# Example of the updated __getitem__ method section:
def example_updated_getitem_section():
    """
    Example of how to update the image loading section in __getitem__
    """
    # OLD CODE:
    # img = cv2.imread(resolved_path)
    # if img is None:
    #     raise ValueError(f"Failed to load image: {resolved_path}")
    
    # NEW CODE:
    img = safe_imread(resolved_path)
    if img is None:
        print(f"Failed to load image: {resolved_path}")
        continue  # Skip this sample and continue with the next one
    
    # Rest of the processing remains the same
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0

# 6. OPTIONAL: Clean existing problematic filenames
def clean_problematic_filenames(data_dir):
    """
    Clean existing filenames that have Unicode corruption.
    Run this once to fix your existing files.
    """
    all_frames_dir = os.path.join(data_dir, "Data", "processed", "all_frames")
    
    if os.path.exists(all_frames_dir):
        print(f"Cleaning filenames in {all_frames_dir}")
        renamed_count = 0
        
        for filename in os.listdir(all_frames_dir):
            if any(char in filename for char in ['ΓÇ»', 'ΓÇ', 'Γ']):
                old_path = os.path.join(all_frames_dir, filename)
                new_filename = sanitize_filename(filename)
                new_path = os.path.join(all_frames_dir, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
        
        print(f"Renamed {renamed_count} files with Unicode issues")
    
    # Also clean other directories
    for split in ['train', 'validation', 'test']:
        split_images_dir = os.path.join(data_dir, "Data", "processed", split, "images")
        if os.path.exists(split_images_dir):
            print(f"Cleaning filenames in {split_images_dir}")
            # Similar cleaning logic for split directories

# 7. HOW TO APPLY THE PATCHES:

"""
STEP-BY-STEP INSTRUCTIONS:

1. Add the imports at the top of training_script.py:
   import unicodedata
   import re

2. Add the safe_imread, sanitize_filename, and find_unicode_safe_path functions 
   to your training_script.py file (before the DogEmotionWithBehaviors class).

3. Replace your existing fix_image_path_resolution function with the enhanced version above.

4. Replace the load_image method in your DogEmotionWithBehaviors class with the new version.

5. In your DataGenerator classes (both regular and enhanced), in the __getitem__ method,
   replace this line:
       img = cv2.imread(resolved_path)
   with:
       img = safe_imread(resolved_path)

6. OPTIONAL: Run the cleaning function once to fix existing files:
   clean_problematic_filenames("C:\\Users\\kelly\\Documents\\GitHub\\Pawnder")

7. Test with a problematic filename to ensure it works.
"""
