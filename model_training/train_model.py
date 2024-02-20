#%%
import torch
import os
import sys



# sys.path.append(os.path.join(os.curdir, r"model_training/utilities"))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split("\\")[:-1])

os.chdir(path)


# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")
        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
sys.path.append(os.getcwd())
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utilities.coco_eval import *
from utilities.engine import *
from utilities.utils import *
from utilities.transforms import *
from utilities.dataloader import *
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import sys

import os
from utilities.engine import train_one_epoch, evaluate
import utilities.utils
from category_information import category_information, class_factor

# force reload the module
import importlib
importlib.reload(utilities.utils)

#%%
if __name__ == '__main__':


    data_root = r'data'
    num_classes = len(category_information)
    
    continue_from_checkpoint = False
    save_model = True
    num_epochs = 0
    train_percentage = 0.8
    batch_size = 8
    learning_rate = 0.005
    momentum=0.9
    weight_decay=0.0005
    
    weights_save_path = r"data\Models"
    weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\model_2024-02-15_13-45-08.pth"
    
    save_info_path= r"data\Models\info"
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # device = torch.device('cpu')

    dataset = LoadDataset(data_root, get_transform(train=True))
    dataset_test = LoadDataset(data_root, get_transform(train=True))
   
    total_samples = len(dataset)
    train_samples = int(train_percentage * total_samples)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:train_samples])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_samples:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )


    # get the model
    model = get_model_instance_segmentation(num_classes)
    if continue_from_checkpoint:
        model.load_state_dict(torch.load(weights_load_path))
    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    
    metrics =[]
    IoU_info = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metrics.append(train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        IoU_info.append(evaluate(model, data_loader_test, device=device))

    print("That's it!")
#%% printing the metrics
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    lr= []
    loss = []
    loss_box_reg = []
    loss_mask = []
    loss_objectness = []
    
    loss_rpn_box_reg = []
    loss_classifier = []

    for i, metric in enumerate(metrics):
        lr.append(metric.meters['lr'].median)
        loss.append(metric.meters['loss'].median)
        loss_box_reg.append(metric.meters['loss_box_reg'].median)
        loss_mask.append(metric.meters['loss_mask'].median)
        loss_objectness.append(metric.meters['loss_objectness'].median)
        loss_rpn_box_reg.append(metric.meters['loss_rpn_box_reg'].median)
        loss_classifier.append(metric.meters['loss_classifier'].median)

        
    plt.plot(lr, label='learning rate')
    plt.plot(loss, label='loss')
    plt.plot(loss_box_reg, label='loss_box_reg')
    plt.plot(loss_mask, label='loss_mask')
    plt.plot(loss_objectness, label='loss_objectness')
    plt.plot(loss_rpn_box_reg, label='loss_rpn_box_reg')
    plt.plot(loss_classifier, label='loss_classifier')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    

    
    
    



#%%
if __name__ == '__main__':
    import datetime
    if save_model:
        datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save the model
        torch.save(model.state_dict(), os.path.join(weights_save_path, f"model_{datetime}_epochs_{num_epochs}.pth"))
        
        # save metrics the metrics list 
        with open(os.path.join(save_info_path, f"metrics_{datetime}.txt"), 'w') as f:
            for metric in metrics:
                f.write(f"{metric}\n")
        # save the evaluator list
        with open(os.path.join(save_info_path, f"IoU_info_{datetime}.txt"), 'w') as f:
            for i, info in enumerate(IoU_info):
                f.write(f"IoU for epoch {i+1}: {info.coco_eval['bbox'].stats}\n")
        

        
    
#%% Get the intersection over union for each class
if __name__ == '__main__':
    model.eval()
    num_classes = len(category_information)
    image_path = r"data\test_Images"
    mask_path = r"data\test_Masks"
    
    # get num of files in path
    nr_of_files_in_path = len(os.listdir(image_path))
    label_confidence_threshold = 0.5
    mask_confidence_threshold = 0.1
    
    
    for file_nr in range(0,nr_of_files_in_path):
        nr_of_files_in_path = len(os.listdir(image_path))
        # file_nr  = np.random.randint(0,nr_of_files_in_path)
        image_path_temp = os.path.join(image_path , f"input-{file_nr}-.jpg")
        true_mask_path_temp = os.path.join(mask_path, f"inst-mask-{file_nr}-.npy")

        image_orig = read_image(image_path_temp).to(device)
        mask_true = torch.from_numpy(np.load(true_mask_path_temp)).to(device)
        eval_transform = get_transform(train=False).to(device)
        
        true_labels =(torch.unique(mask_true)//class_factor)[1:]
        mask_true= mask_true.to(device)
        masks_true =((mask_true == true_labels[:, None, None]).to(dtype=torch.uint8)).to(device)
        
        num_classes_mask = len(torch.unique(true_labels))
        
        with torch.no_grad():
            x = eval_transform(image_orig)
            # convert RGBA -> RGB and move to device
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]
        
        pred_labels, pred_boxes, pred_masks = pred['labels'].to(device), pred['boxes'], pred['masks'].to(device)
        pred_masks = pred_masks > mask_confidence_threshold


        # get the masks for each class
        pred_masks_per_class = torch.zeros((num_classes_mask, *masks_true.shape[1:]), dtype=torch.bool) 
        
        
        
        for i, label in enumerate(torch.unique(true_labels)):
            pred_masks_per_class[i] = torch.any(pred_masks[pred_labels == label], dim=0)

        
        
        
        # calculate the IoU
        IoU = []
        class_names = [list(category_information.keys())[list(category_information.values()).index(x)] for x in true_labels]
        for i in range(len(pred_masks_per_class)):
            IoU.append(calculate_IoU(true_masks_per_class[i] ,pred_masks_per_class[i]))
            print(f"IoU for class {class_names[i]}: {IoU[i]}")


        
        
        
# %%
