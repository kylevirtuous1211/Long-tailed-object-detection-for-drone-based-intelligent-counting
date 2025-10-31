# Train the model using YOLO
## Directory structure:
```
your_project/
├── data/
|   └──data_yolo (created by script)
│       ├── images/
│       │   ├── train/     # Put your 760 training images here (e.g., img0001.png)
│       │   └── val/       # Put your 190 validation images here
│       ├── labels/
│       │   ├── train/     # Put your 760 converted YOLO .txt files here (e.g., img0001.txt)
│       │   └── val/       # Put your 190 converted YOLO .txt files here
│       └── data.yaml      # The dataset configuration file
├── convert_to_yolo.py # Your conversion script
└── ... (your other files) 
```
## Unzip the data 
1. Put `taica-cvpdl-2025-hw-2.zip` inside `data/`
```
mkdir data
cd data/
unzip taica-cvpdl-2025-hw-2.zip
```
## Create yolo annotation directory format 
```
python yolo_annotation.py
```
It should create `data_yolo/` with `data.yaml` after the script.

## Train using a small YOLO
### Make sure you are inside `data/`
This command trains the yolov8s from scratch.
* imgsz=1920 is the key to high accuracy
```
 yolo detect train data=data_yolo/data.yaml model=yolov8s.yaml \ epochs=200 imgsz=1920 batch=4 device=1 \ name=yolov8s_scratch_imgsz1920_adamw_lr0.0001_wd optimizer=AdamW \ lr0=0.0001 weight_decay=0.0008 \ pretrained=False 
```

## Train using a median YOLO
### Make sure you are inside `data/`
This command trains the yolov8s from scratch.
* imgsz=1920 is the key to high accuracy
* I only trained till 141 epochs
```
 yolo detect train data=data_yolo/data.yaml model=yolov8m.yaml \ epochs=200 imgsz=1920 batch=2 device=0 \ name=yolov8m_scratch_imgsz1920_adamw_lr0.0001_wd optimizer=AdamW \ lr0=0.0001 weight_decay=0.0008 \ pretrained=False
```

# Finetune step: yolov8s using tiny learning rate
### Make sure you are inside `data/`
This command finetunes the yolov8s with smaller lr=1e-5
```
nohup yolo detect train \
    data=data_yolo/data.yaml \
    model=runs/detect/yolov8s_scratch_imgsz1920_adamw_lr0.0001_wd5/weights/best.pt \
    epochs=50 \
    imgsz=1920 \
    batch=2 \
    device=0 \
    name=yolov8m_finetune_lr1e-5 \
    optimizer=AdamW \
    lr0=1e-5 \
    lrf=0.1 \
    weight_decay=0.0008 > finetune_s.log 2>&1 &
```

# Finetune step: yolov8m using tiny learning rate
### Make sure you are inside `data/`
This command finetunes the yolov8s with smaller lr=1e-5
```
nohup yolo detect train \
    data=data_yolo/data.yaml \
    model=runs/detect/yolov8m_scratch_imgsz1920_adamw_lr0.0001_wd3/weights/best.pt \
    epochs=30 \
    imgsz=1920 \
    batch=2 \
    device=1 \
    name=yolov8m_finetune_lr1e-5 \
    optimizer=AdamW \
    lr0=1e-5 \
    lrf=0.1 \
    weight_decay=0.0008 > finetune_m.log 2>&1 & 
```

# Inference 
Go back to the root directory

update the checkpoint `CHECKPOINT_PATH`, `MODEL_NAME` in `inferece_yolo.py`:
```
# --- Configuration ---
MODEL_NAME = 'yolov8m_1920_scratch_141e_lr1e-4' # A name for this run/model
# No config path needed for YOLOv8 inference, just the weights
CHECKPOINT_PATH = '/home/cvlab123/Kyle_Having_Fun/NTU_CV/HW2_longtail_object_detection/Long-tailed_Object_Detection/data/runs/detect/yolov8m_scratch_imgsz1920_adamw_lr0.0001_wd3/weights/best.pt'
```
## WBF inference
If you want to use use WBF, need to update the path to 2 checkpoints

update the checkpoint path `MODEL_1_PATH` and `MODEL_2_PATH` in `WBF.py`
```
# TODO: Fill in the paths to your two finetuned models
MODEL_1_PATH = 'data/runs/detect/yolov8m_finetune_lr1e-5/weights/best.pt'
MODEL_2_PATH = 'data/runs/detect/yolov8s_finetune_lr1e-5/weights/best.pt'
```



