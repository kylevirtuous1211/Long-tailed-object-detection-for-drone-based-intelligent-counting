import os
import glob
import csv
import numpy as np
from ultralytics import YOLO 
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import cv2 # <-- Uncommented this

# --- Configuration ---

# TODO: Fill in the paths to your two finetuned models
MODEL_1_PATH = 'data/runs/detect/yolov8m_finetune_lr1e-5/weights/best.pt'
MODEL_2_PATH = 'data/runs/detect/yolov8s_finetune_lr1e-5/weights/best.pt'

# --- Model Weights (Set how much you trust each model. 1.0 = equal trust)
MODEL_1_WEIGHT = 1.0
MODEL_2_WEIGHT = 1.0

# --- Prediction Settings (These run *before* WBF to reduce NMS lag)
MODEL_CONF_THRESH = 0.01  # Initial confidence threshold for models (avoids NMS timeout)
MODEL_IOU_THRESH = 0.5    # Initial IOU threshold for models

# --- WBF Fusion Settings
WBF_IOU_THRESH = 0.55     # IoU threshold for WBF (boxes > this will be fused)
WBF_SKIP_BOX_THRESH = 0.001 # Final confidence threshold (same as your old CONFIDENCE_THRESHOLD)

MODEL_NAME = 'yolov8_wbf_m-finetune_s-finetune' # A name for this fusion run

# --- NEW: Visualization Settings ---
VISUALIZE_N_IMAGES = 10 # Number of test images to save with boxes drawn
VISUALIZATION_DIR = f'visualization/{MODEL_NAME}' # Directory for saving visualized images

# --- Script Paths ---
OUTPUT_CSV_PATH = f'{MODEL_NAME}_submission.csv' # Output CSV file name
TEST_IMAGES_DIR = 'data/CVPDL_hw2/CVPDL_hw2/test/'
DEVICE = 'cuda:1' # Device for models to run on
# ---------------------

def format_wbf_for_kaggle(fused_boxes, fused_scores, fused_labels, orig_shape):
    """
    Formats the WBF (normalized) predictions into the Kaggle submission string.

    Args:
        fused_boxes (np.array): Array of [xmin, ymin, xmax, ymax] in normalized (0-1) format.
        fused_scores (np.array): Array of confidence scores.
        fused_labels (np.array): Array of class labels.
        orig_shape (tuple): The (height, width) of the original image.

    Returns:
        str: The formatted PredictionString for the image.
    """
    if len(fused_boxes) == 0:
        return ""
    
    pred_strings = []
    h, w = orig_shape

    for bbox, score, label in zip(fused_boxes, fused_scores, fused_labels):
        # WBF boxes are [xmin, ymin, xmax, ymax] normalized
        # Un-normalize them back to pixel coordinates
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox
        
        xmin = xmin_norm * w
        ymin = ymin_norm * h
        xmax = xmax_norm * w
        ymax = ymax_norm * h

        # Convert to Kaggle format: [Top-left X, Top-left Y, width, height]
        # Ensure values are integers
        x = int(round(xmin))
        y = int(round(ymin))
        box_w = int(round(xmax - xmin))
        box_h = int(round(ymax - ymin))
        
        # Ensure width and height are at least 1 pixel
        box_w = max(1, box_w)
        box_h = max(1, box_h)

        # Class label should be integer
        label = int(label)

        # Format: <conf> <bb_left> <bb_top> <bb_width> <bb_height> <class_label>
        pred_strings.append(f"{score:.6f} {x} {y} {box_w} {box_h} {label}")

    # Join all predictions for the image with spaces
    return " ".join(pred_strings)

# +++ NEW FUNCTION +++
def visualize_wbf_results(img_path, fused_boxes, fused_scores, fused_labels, orig_shape, save_path):
    """
    Draws WBF results on an image and saves it.

    Args:
        img_path (str): Path to the original image.
        fused_boxes (np.array): Array of [xmin, ymin, xmax, ymax] in normalized (0-1) format.
        fused_scores (np.array): Array of confidence scores.
        fused_labels (np.array): Array of class labels.
        orig_shape (tuple): The (height, width) of the original image.
        save_path (str): Path to save the visualized image.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            return
            
        h, w = orig_shape

        for bbox, score, label in zip(fused_boxes, fused_scores, fused_labels):
            # Un-normalize boxes
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox
            xmin = int(xmin_norm * w)
            ymin = int(ymin_norm * h)
            xmax = int(xmax_norm * w)
            ymax = int(ymax_norm * h)
            
            label = int(label)
            score_str = f"{score:.2f}"
            
            # Create a consistent color for each class
            np.random.seed(label) # Use label to seed the random color
            color = np.random.randint(0, 255, size=3).tolist()

            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Create text for label and score
            text = f"Class {label}: {score_str}"
            
            # Calculate text size to draw a background box
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw a filled box for the text background
            cv2.rectangle(img, (xmin, ymin - text_h - baseline - 2), (xmin + text_w, ymin), color, -1)
            
            # Put text on the image (using black text for better contrast on light colors)
            cv2.putText(img, text, (xmin, ymin - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(save_path, img)
    except Exception as e:
        print(f"Error during visualization for {img_path}: {e}")
# +++ END OF NEW FUNCTION +++

def main():
    # Initialize the YOLO models
    print(f"Loading Model 1 from: {MODEL_1_PATH}")
    model1 = YOLO(MODEL_1_PATH)
    model1.to(DEVICE)
    print(f"Loading Model 2 from: {MODEL_2_PATH}")
    model2 = YOLO(MODEL_2_PATH)
    model2.to(DEVICE)
    print("Models loaded.")

    # Get sorted list of test images
    test_image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*.png')))
    if not test_image_paths:
         test_image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg')))

    if not test_image_paths:
        print(f"Error: No images found in {TEST_IMAGES_DIR}")
        return

    print(f"Found {len(test_image_paths)} test images.")
    
    # --- Create visualization directory ---
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    print(f"Visualization images will be saved to: {VISUALIZATION_DIR}")
    
    # Open the CSV file for writing
    with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write header
        csvwriter.writerow(['Image_ID', 'PredictionString'])
        print(f"Writing results to {OUTPUT_CSV_PATH}...")

        # Process each image
        for i, img_path in enumerate(tqdm(test_image_paths, desc="Processing images")):
            image_id = i + 1 # Kaggle Image_ID starts from 1
            
            # --- Run Inference ---
            # We run predict on both models separately
            # We set save=False and verbose=False to speed things up
            results1 = model1.predict(
                source=img_path,
                conf=MODEL_CONF_THRESH,
                iou=MODEL_IOU_THRESH,
                device=DEVICE,
                save=False,
                verbose=False
            )
            
            results2 = model2.predict(
                source=img_path,
                conf=MODEL_CONF_THRESH,
                iou=MODEL_IOU_THRESH,
                device=DEVICE,
                save=False,
                verbose=False
            )

            # --- Extract Data for WBF ---
            # WBF requires boxes to be NORMALIZED (0-1)
            # We use .xyxyn to get normalized coordinates
            
            # Model 1 results
            if results1 and results1[0].boxes.conf is not None:
                boxes1 = results1[0].boxes.xyxyn.cpu().numpy()
                scores1 = results1[0].boxes.conf.cpu().numpy()
                labels1 = results1[0].boxes.cls.cpu().numpy()
                orig_shape = results1[0].orig_shape # Get image shape
            else:
                boxes1, scores1, labels1 = np.array([]), np.array([]), np.array([])
                orig_shape = (1920, 1920) # Fallback, should be overwritten by a valid result
            
            # Model 2 results
            if results2 and results2[0].boxes.conf is not None:
                boxes2 = results2[0].boxes.xyxyn.cpu().numpy()
                scores2 = results2[0].boxes.conf.cpu().numpy()
                labels2 = results2[0].boxes.cls.cpu().numpy()
                orig_shape = results2[0].orig_shape # Get image shape (will be same as model 1)
            else:
                boxes2, scores2, labels2 = np.array([]), np.array([]), np.array([])

            # --- Prepare for WBF ---
            boxes_list = [boxes1, boxes2]
            scores_list = [scores1, scores2]
            labels_list = [labels1, labels2]
            weights_list = [MODEL_1_WEIGHT, MODEL_2_WEIGHT]

            # --- Run WBF ---
            # This function fuses all boxes into one set
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights_list,
                iou_thr=WBF_IOU_THRESH,
                skip_box_thr=WBF_SKIP_BOX_THRESH
            )
            
            # --- Format for Kaggle ---
            # We use our new function to convert WBF output to Kaggle string
            prediction_string = format_wbf_for_kaggle(fused_boxes, fused_scores, fused_labels, orig_shape)

            # Write row to CSV
            csvwriter.writerow([image_id, prediction_string])
            
            # --- NEW: Visualize Results ---
            if i < VISUALIZE_N_IMAGES:
                img_filename = os.path.basename(img_path)
                save_path = os.path.join(VISUALIZATION_DIR, f"wbf_{img_filename}")
                
                # Call the new visualization function
                visualize_wbf_results(
                    img_path,
                    fused_boxes,
                    fused_scores,
                    fused_labels,
                    orig_shape,
                    save_path
                )

    print("WBF Inference complete. Submission file saved.")
    if VISUALIZE_N_IMAGES > 0:
        print(f"Saved {min(VISUALIZE_N_IMAGES, len(test_image_paths))} visualization images to {VISUALIZATION_DIR}")

if __name__ == '__main__':
    main()

