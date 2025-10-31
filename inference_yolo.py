import os
import glob
import csv
# Use Ultralytics YOLO instead of MMDetection
from ultralytics import YOLO 
from tqdm import tqdm
import numpy as np
import cv2 # Keep cv2 for saving visualizations if needed

# --- Configuration ---
MODEL_NAME = 'yolov8m_1920_scratch_141e_lr1e-4' # A name for this run/model
# No config path needed for YOLOv8 inference, just the weights
CHECKPOINT_PATH = '/home/cvlab123/Kyle_Having_Fun/NTU_CV/HW2_longtail_object_detection/Long-tailed_Object_Detection/data/runs/detect/yolov8m_scratch_imgsz1920_adamw_lr0.0001_wd3/weights/best.pt'
TEST_IMAGES_DIR = 'data/CVPDL_hw2/CVPDL_hw2/test/' # Directory containing 【test images
OUTPUT_CSV_PATH = f'{MODEL_NAME}_best_ckpt_submission.csv' # Output CSV file name
VISUALIZATION_DIR = f'visualization/{MODEL_NAME}' # Directory for saving visualized images
CONFIDENCE_THRESHOLD = 0.001 # Threshold for filtering detections
DEVICE = 'cuda:1' # Or 'cpu', or device index like 0, 1 etc.
# ---------------------

def format_predictions_for_kaggle_yolov8(results):
    """
    Formats the predictions from Ultralytics YOLO results object into the Kaggle submission string.

    Args:
        results (list): A list containing results for a single image (output of model.predict).

    Returns:
        str: The formatted PredictionString for the image.
    """
    if not results or not results[0]:
        return ""

    result = results[0] # Get the results object for the first (only) image
    boxes = result.boxes # Access the Boxes object containing detections

    pred_strings = []

    # Filter detections by confidence score
    # boxes.conf gives confidence scores, boxes.cls gives class indices
    # boxes.xyxy gives bounding boxes in [xmin, ymin, xmax, ymax] format (pixel coordinates)
    
    # Check if there are any detections before proceeding
    if boxes.conf is None or len(boxes.conf) == 0:
        return ""
        
    keep_indices = np.where(boxes.conf.cpu().numpy() >= CONFIDENCE_THRESHOLD)[0]

    if len(keep_indices) == 0:
        return "" # Return empty string if no detections meet threshold

    bboxes_xyxy = boxes.xyxy.cpu().numpy()[keep_indices]
    scores = boxes.conf.cpu().numpy()[keep_indices]
    labels = boxes.cls.cpu().numpy()[keep_indices]

    for bbox, score, label in zip(bboxes_xyxy, scores, labels):
        # YOLOv8 bbox format is [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = bbox

        # Convert to Kaggle format: [Top-left X, Top-left Y, width, height]
        # Ensure values are integers
        x = int(round(xmin))
        y = int(round(ymin))
        w = int(round(xmax - xmin))
        h = int(round(ymax - ymin))
        
        # Ensure width and height are at least 1 pixel
        w = max(1, w)
        h = max(1, h)

        # Class label should be integer
        label = int(label)

        # Format: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class_label>
        pred_strings.append(f"{score:.6f} {x} {y} {w} {h} {label}")

    # Join all predictions for the image with spaces
    return " ".join(pred_strings)

def main():
    # Initialize the YOLO model
    print(f"Loading YOLOv8 model from: {CHECKPOINT_PATH}")
    model = YOLO(CHECKPOINT_PATH)
    # Move model to the specified device (Ultralytics handles device placement nicely)
    model.to(DEVICE) 
    print("YOLOv8 model loaded.")

    # Get sorted list of test images
    test_image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*.png')))
    if not test_image_paths:
         test_image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg')))

    if not test_image_paths:
        print(f"Error: No images found in {TEST_IMAGES_DIR}")
        return

    print(f"Found {len(test_image_paths)} test images.")
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Open the CSV file for writing
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write header
        csvwriter.writerow(['Image_ID', 'PredictionString'])
        print(f"Writing results to {OUTPUT_CSV_PATH}...")

        # Process each image
        for i, img_path in enumerate(tqdm(test_image_paths, desc="Processing images")):
            image_id = i + 1 # Kaggle Image_ID starts from 1
            img_filename = os.path.basename(img_path)
            
            # --- YOLOv8 Inference ---
            # Use model.predict() for inference
            # Set save=True to save visualized images automatically
            # Set project and name to control output directory structure
            # Set conf for filtering (though we filter again manually for precision)
            results = model.predict(
                source=img_path,
                conf=CONFIDENCE_THRESHOLD, # Apply initial thresholding
                device=DEVICE,
                save=True, # Save images with boxes
                # augment=True,
                project=VISUALIZATION_DIR, # Base directory for saving
                name=f"image_{image_id:04d}", # Sub-directory for this image's results
                exist_ok=True, # Allows overwriting if run multiple times
                verbose=False # Suppress individual image prediction logs
            )
            # results is a list (usually with one element for one image)

            # Format predictions for CSV
            prediction_string = format_predictions_for_kaggle_yolov8(results)

            # Write row to CSV
            csvwriter.writerow([image_id, prediction_string])

    print("Inference complete. Submission file and visualizations saved.")

if __name__ == '__main__':
    main()
