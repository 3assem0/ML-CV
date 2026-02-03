from ultralytics import YOLO
import os
import shutil
import glob
import random
import yaml

def prepare_dataset(raw_dir="data/raw", split_ratio=0.8):
    """
    Organizes raw images into YOLO structure (train/val).
    Note: Real object detection training requires LABELING (drawing boxes).
    This script assumes you will use a tool specifically to label images 
    OR we can train a Classifier (YOLOv8-cls) if the goal is just recognizing the object type in a cropped image.
    
    If the user wants Object Detection (finding WHERE the object is), they MUST label the data.
    For simplicity in this demo, we will assume the user has labeled data or this is a classification task.
    
    HOWEVER, given the prompt 'identify if ... artifact or stone', simple Classification might be easier 
    if we are cropping the object first.
    
    YOLOv8 Classify mode:
    data/
      train/
        artifact/
        stone/
      val/
        artifact/
        stone/
    """
    
    # We will set up for CLASSIFICATION for simplest custom training workflow 
    # unless the user wants to draw boxes manually.
    # Let's pivot to Classification for the "Custom Mode" as it's easier for end-users 
    # than bounding box annotation tools.
    
    base_dir = "data"
    for split in ["train", "val"]:
        for cls in ["artifact", "stone", "glass", "plastic"]:
            os.makedirs(f"{base_dir}/{split}/{cls}", exist_ok=True)
            
    # Move files
    for cls in ["artifact", "stone", "glass", "plastic"]:
        src_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(src_path):
            continue
            
        files = glob.glob(os.path.join(src_path, "*.*"))
        random.shuffle(files)
        
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for f in train_files:
            shutil.copy(f, f"{base_dir}/train/{cls}/{os.path.basename(f)}")
        for f in val_files:
            shutil.copy(f, f"{base_dir}/val/{cls}/{os.path.basename(f)}")

def train_custom_model(epochs=5):
    # Prepare Data
    prepare_dataset()
    
    # Load Model (Classification)
    model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

    # Train
    model.train(data="data", epochs=epochs, imgsz=224)
    
    # Save Best
    # YOLO automatically saves to runs/classify/train/weights/best.pt
    # We copy it to models/
    os.makedirs("models", exist_ok=True)
    try:
        # Pind the latest run directory
        runs = sorted(glob.glob("runs/classify/train*"), key=os.path.getmtime)
        if runs:
            latest_run = runs[-1]
            src = f"{latest_run}/weights/best.pt"
            dst = "models/custom_archaeology.pt"
            shutil.copy(src, dst)
            print(f"Model saved to {dst}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_custom_model()
