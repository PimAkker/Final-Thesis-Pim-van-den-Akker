import os
import torch
from ultralytics import YOLO

# Ensure you're using a compatible device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths
data_path = 'path/to/your/dataset'  # Path to your dataset
model_path = 'path/to/your/pretrained/model'  # Path to your pretrained model if any
results_path = 'path/to/save/results'  # Path to save the training results

# Load the model
# If you don't have a pretrained model, you can start with a YOLOv8 base model
model = YOLO(model_path if os.path.exists(model_path) else 'yolov8s-seg.pt').to(device)

# Define hyperparameters
hyperparameters = {
    'epochs': 50,            # Number of epochs to train for
    'batch_size': 16,        # Batch size
    'img_size': 640,         # Input image size
    'learning_rate': 0.001,  # Initial learning rate
    'momentum': 0.937,       # Momentum for SGD
    'weight_decay': 0.0005,  # Weight decay
    'warmup_epochs': 3.0,    # Warmup epochs for learning rate scheduler
    'warmup_momentum': 0.8,  # Warmup initial momentum
    'warmup_bias_lr': 0.1,   # Warmup initial bias learning rate
    'box': 0.05,             # Box loss gain
    'cls': 0.5,              # Class loss gain
    'cls_pw': 1.0,           # Class positive weight
    'obj': 1.0,              # Object loss gain (scale)
    'obj_pw': 1.0,           # Object positive weight
    'iou_t': 0.20,           # IoU training threshold
    'anchor_t': 4.0,         # Anchor-multiple threshold
    'fl_gamma': 0.0,         # Focal Loss gamma (efficientDet default is gamma=1.5)
    'hsv_h': 0.015,          # Image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,            # Image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,            # Image HSV-Value augmentation (fraction)
    'degrees': 0.0,          # Image rotation (+/- deg)
    'translate': 0.1,        # Image translation (+/- fraction)
    'scale': 0.5,            # Image scale (+/- gain)
    'shear': 0.0,            # Image shear (+/- deg)
    'perspective': 0.0,      # Image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,           # Image flip up-down (probability)
    'fliplr': 0.5,           # Image flip left-right (probability)
    'mosaic': 1.0,           # Mosaic augmentation probability
    'mixup': 0.0,            # Mixup augmentation probability
    'copy_paste': 0.0        # Copy-Paste augmentation probability
}

# Set up training configuration
train_cfg = {
    'data': data_path,       # Path to dataset
    'project': results_path, # Where to save results
    'name': 'exp',           # Experiment name
    'batch': hyperparameters['batch_size'],
    'epochs': hyperparameters['epochs'],
    'imgsz': hyperparameters['img_size'],
    'hyp': hyperparameters,  # Hyperparameters
    'device': device,
    'workers': 8,            # Number of data loading workers (threads)
    'freeze': [0],           # Freeze layers (optional)
    'save_period': 1,        # Save checkpoint every X epochs
    'exist_ok': True,        # Overwrite existing experiment with the same name
    'verbose': True,         # Verbose output
}

# Train the model
model.train(**train_cfg)

# Save the trained model
model_path = os.path.join(results_path, 'yolov8s-seg-train.pt')
model.save(model_path)
print(f'Model saved to {model_path}')
