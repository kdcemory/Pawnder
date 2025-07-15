import os
import zipfile
import shutil
from pathlib import Path

def extract_colab_dataset():
    """
    Run this on your LOCAL machine to extract downloaded dataset
    """
    
    # Paths
    download_folder = Path.home() / "Downloads"  # Adjust if your downloads go elsewhere
    target_path = Path(r"C:\Users\kelly\Documents\GitHub\pawnder\Data\raw")
    
    print(f"Looking for dataset zip files in: {download_folder}")
    print(f"Target directory: {target_path}")
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Look for dataset zip files
    zip_files = list(download_folder.glob("pawnder_dataset*.zip"))
    
    if not zip_files:
        print("No pawnder dataset zip files found in Downloads folder!")
        print("Available zip files:")
        for zip_file in download_folder.glob("*.zip"):
            print(f"  - {zip_file.name}")
        return
    
    # Extract each zip file
    for zip_file in zip_files:
        print(f"\nExtracting: {zip_file.name}")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract to target directory
            zip_ref.extractall(target_path)
        
        print(f"Extracted to: {target_path}")
        
        # Optional: Remove zip file after extraction
        # zip_file.unlink()  # Uncomment to delete zip after extraction
    
    print(f"\nDataset extraction complete!")
    print(f"Dataset location: {target_path}")
    
    # Show directory structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk(target_path):
        level = root.replace(str(target_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{sub_indent}{file}")
        if len(files) > 5:
            print(f"{sub_indent}... and {len(files) - 5} more files")

def extract_from_custom_location():
    """
    If your zip file is not in Downloads, specify the path here
    """
    
    # Specify the exact path to your zip file
    zip_file_path = input("Enter full path to your dataset zip file: ").strip().strip('"')
    target_path = Path(r"C:\Users\kelly\Documents\GitHub\pawnder\Data\raw")
    
    if not os.path.exists(zip_file_path):
        print(f"File not found: {zip_file_path}")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting: {zip_file_path}")
    print(f"To: {target_path}")
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    
    print("Extraction complete!")

if __name__ == "__main__":
    print("Pawnder Dataset Extraction Tool")
    print("1. Extract from Downloads folder (automatic)")
    print("2. Extract from custom location")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        extract_colab_dataset()
    elif choice == "2":
        extract_from_custom_location()
    else:
        print("Invalid choice. Running automatic extraction...")
        extract_colab_dataset()