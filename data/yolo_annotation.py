import os
import glob
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import collections
import numpy as np

# --- Configuration ---
# Path to the directory containing original images AND original .txt files
SOURCE_DIR = 'CVPDL_hw2/CVPDL_hw2/train/' 
# Root directory where the new YOLO dataset structure will be created
OUTPUT_YOLO_DIR = 'data_yolo/' 
TEST_SPLIT_RATIO = 0.2  # 20% of data will be used for validation
RANDOM_STATE = 42       # For reproducible splits

# Class names in the correct order (ID 0 to N-1)
CLASSES = ['car', 'hov', 'person', 'motorcycle']

# IMPORTANT: Define class rarity for stratification (RAREST to COMMONEST ID)
# Based on your chart: hov (1), person (2), motorcycle (3), car (0)
CLASS_RARITY_ORDER = [1, 2, 3, 0] 
# ---------------------

# --- Define target subdirectories ---
IMG_TRAIN_DIR = os.path.join(OUTPUT_YOLO_DIR, 'images', 'train')
IMG_VAL_DIR = os.path.join(OUTPUT_YOLO_DIR, 'images', 'val')
LABEL_TRAIN_DIR = os.path.join(OUTPUT_YOLO_DIR, 'labels', 'train')
LABEL_VAL_DIR = os.path.join(OUTPUT_YOLO_DIR, 'labels', 'val')
YAML_PATH = os.path.join(OUTPUT_YOLO_DIR, 'data.yaml')
# ------------------------------------

def convert_to_yolo(img_width, img_height, class_id, x_tl, y_tl, w, h):
    """Converts a single bounding box to YOLO format."""
    x_center = x_tl + w / 2
    y_center = y_tl + h / 2

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    # Clamp values to be within [0.0, 1.0] just in case
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    # Ensure class_id is an integer
    class_id = int(class_id)
    
    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def process_and_save_split(image_data_list, img_dest_dir, label_dest_dir):
    """Processes a list of image data, copies images, converts labels, and saves labels."""
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    print(f"Processing split for {img_dest_dir} and {label_dest_dir}...")
    
    for img_filename, width, height, annotations in tqdm(image_data_list, desc=f"Writing to {os.path.basename(img_dest_dir)}"):
        # 1. Copy image file
        source_img_path = os.path.join(SOURCE_DIR, img_filename)
        dest_img_path = os.path.join(img_dest_dir, img_filename)
        if os.path.exists(source_img_path):
             shutil.copy2(source_img_path, dest_img_path)
        else:
            print(f"Warning: Source image not found, cannot copy: {source_img_path}")
            continue # Skip if image missing

        # 2. Convert annotations to YOLO format
        yolo_lines = []
        for class_id, x, y, w, h in annotations:
             try:
                 yolo_line = convert_to_yolo(width, height, class_id, x, y, w, h)
                 yolo_lines.append(yolo_line)
             except Exception as e:
                  print(f"Error converting annotation in {img_filename}: {e}")
                  
        # 3. Write YOLO annotation file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        dest_label_path = os.path.join(label_dest_dir, label_filename)
        with open(dest_label_path, 'w') as f_out:
            f_out.write("\n".join(yolo_lines))

def create_yolo_structure_and_yaml():
    """
    Main function: Reads source data, splits, converts, creates structure, and writes YAML.
    """
    
    # Find all image files (.png and .jpg)
    image_files_png = glob.glob(os.path.join(SOURCE_DIR, '*.png'))
    image_files_jpg = glob.glob(os.path.join(SOURCE_DIR, '*.jpg'))
    all_image_files = sorted(image_files_png + image_files_jpg)
    
    if not all_image_files:
        print(f"Error: No .png or .jpg images found in {SOURCE_DIR}")
        return

    all_image_data = []
    all_strata_keys = []
    
    print("Reading source files and building stratification keys...")
    
    for img_path in tqdm(all_image_files, desc="Reading source data"):
        img_filename = os.path.basename(img_path)
        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(SOURCE_DIR, txt_filename) # Original txt file location

        # Get image dimensions
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Image could not be read")
            height, width, _ = img.shape
        except Exception as e:
            print(f"Could not read image {img_path}, skipping. Error: {e}")
            continue

        annotations = []
        present_classes = set()
        
        # Read original annotations
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split(',')
                        if len(parts) == 5:
                             class_id, x, y, w, h = map(float, parts)
                             annotations.append((int(class_id), x, y, w, h))
                             present_classes.add(int(class_id))
                        else:
                             print(f"Warning: Skipping malformed line in {txt_filename}: {line.strip()}")
                    except ValueError:
                         print(f"Warning: Skipping malformed line in {txt_filename}: {line.strip()}")
                         continue
        
        # Determine the stratification key for this image
        strata_key = -1 # Default key for images with no objects or annotations
        if present_classes:
            for class_label in CLASS_RARITY_ORDER: # Iterate from rarest to commonest
                if class_label in present_classes:
                    strata_key = class_label # Stratify by the rarest class found
                    break
        
        # Store image filename (not path), dimensions, and original annotations
        all_image_data.append((img_filename, width, height, annotations)) 
        all_strata_keys.append(strata_key)

    print("\nSource data reading complete.")
    print(f"Total images processed: {len(all_image_data)}")
    print("Stratification key counts (rarest class per image, -1 = no objects/annotations):")
    print(collections.Counter(all_strata_keys))

    if not all_image_data:
        print("No valid image/annotation pairs found to split.")
        return

    # Perform the stratified split
    print(f"\nPerforming {TEST_SPLIT_RATIO*100}% stratified split...")
    try:
        train_data, val_data = train_test_split(
            all_image_data,
            test_size=TEST_SPLIT_RATIO,
            stratify=all_strata_keys, # Use the calculated keys
            random_state=RANDOM_STATE
        )
    except ValueError as e:
         print(f"\nError during train_test_split (potentially too few samples for stratification): {e}")
         print("Consider reducing TEST_SPLIT_RATIO or checking data distribution.")
         # Fallback to non-stratified split if stratification fails
         print("Falling back to a non-stratified split...")
         train_data, val_data = train_test_split(
            all_image_data,
            test_size=TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE
        )

    
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")

    # Create directories, copy images, convert and save labels for both splits
    process_and_save_split(train_data, IMG_TRAIN_DIR, LABEL_TRAIN_DIR)
    process_and_save_split(val_data, IMG_VAL_DIR, LABEL_VAL_DIR)

    # Create data.yaml file
    print(f"\nCreating {YAML_PATH}...")
    yaml_data = {
        # Paths relative to the YAML file location (OUTPUT_YOLO_DIR)
        'train': 'images/train', 
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    try:
        with open(YAML_PATH, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully created {YAML_PATH}")
    except Exception as e:
        print(f"Error writing {YAML_PATH}: {e}")
    
    print("\nDataset structure creation complete.")

if __name__ == "__main__":
    # Add check for source directory existence
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: Source directory not found at {SOURCE_DIR}")
    else:
        create_yolo_structure_and_yaml()
