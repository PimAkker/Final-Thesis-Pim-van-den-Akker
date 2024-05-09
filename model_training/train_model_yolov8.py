import torch
import yaml
from ultralytics import YOLO

# --- Step 1:  Dataset and Model Configuration ---

data_yaml = 'path/to/your/data.yaml'  
model_name = 'yolov8n-seg.pt'  


yolo = YOLO(model_name)  
results = yolo.train(data=data_yaml, epochs=100)  # Adjust epochs as needed

# --- Step 3: Save Trained Model ---

results.save(save_dir='runs/train/exp') 
