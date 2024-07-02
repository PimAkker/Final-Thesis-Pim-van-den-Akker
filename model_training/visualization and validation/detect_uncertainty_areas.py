#%%
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split(os.sep)[:-1])
os.chdir(path)
# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split(os.sep)
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split(os.sep)
        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
sys.path.append(os.getcwd())
from category_information import category_information
import torch
from  model_training.utilities.dataloader import get_transform, LoadDataset, get_model_instance_segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_data(data_path, percentage=1):
    
    dataset = LoadDataset(data_path, get_transform(train=False), ignore_indexes=2)    
    percentage_of_dataset_to_use = 1
    dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * percentage_of_dataset_to_use)))
    return dataset

def load_model(weights_load_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(weights_load_path))
    model.to(device)
    return model

def run_model(model, data, image_number, device, mask_threshold, box_threshold):
    """runs the model and returns number_of_images images"""
    boxes = []
    labels = []
    masks = []
    scores = []
    
    img, _ = data[image_number]
    model.eval()
    
    with torch.no_grad():
        prediction = model([img.to(device)])
        pass_threshold_indices_labels = torch.where(prediction[0]['scores'] > box_threshold)
        boxes_thresholded = prediction[0]['boxes'][pass_threshold_indices_labels]
        labels_thresholded = prediction[0]['labels'][pass_threshold_indices_labels]
        
        
        boxes.append(boxes_thresholded)
        labels.append(labels_thresholded)
        masks.append(prediction[0]['masks'])
        scores.append(prediction[0]['scores'])
        
    return prediction, boxes, labels, masks, img, scores

def show_boxes(img, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
    for box in boxes[0]:
        box = box.cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
def show_masks(img, masks):
    fig, ax = plt.subplots(1)
    ax.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
    for mask in masks[0]:
        mask = mask[0].cpu().numpy()
        ax.imshow(mask, alpha=0.5)
        # show some info 
        plt.title(f"mask shape: {mask.shape}")
    plt.show()
#%%

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_to_test_on = r'real_world_data\Real_world_data_V2'
    # data_to_test_on = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\varying_heights\[]"
    
    box_threshold = 0.5
    mask_threshold = 0.5
    
    image_nr = 7
    
    num_classes = len(category_information)  
    weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_v3\weights.pth"
    model = load_model(weights_load_path, num_classes, device)
    data = load_data(data_to_test_on)
    
    prediction_dict, boxes,labels,masks,img, scores  = run_model(model, data, image_nr, device, mask_threshold, box_threshold)
    print(f"number of boxes: {len(boxes)}")
    print(f"number of labels: {len(labels)}")
    print(f"number of masks: {len(masks)}")
    
    
    show_boxes(img, boxes)
    show_masks(img, masks)

    
    
# %%
