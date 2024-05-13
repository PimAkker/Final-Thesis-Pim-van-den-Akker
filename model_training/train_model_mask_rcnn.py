#%%
import torch
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(path))

os.chdir(path)

# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

import torch
import numpy as np
from  model_training.utilities.coco_eval import *
from  model_training.utilities.engine import *
from  model_training.utilities.utils import *
from  model_training.utilities.transforms import *
from  model_training.utilities.dataloader import *
import matplotlib.pyplot as plt

from model_training.utilities.engine import train_one_epoch, evaluate
import model_training.utilities.utils
from category_information import category_information

# force reload the module
import importlib
importlib.reload(model_training.utilities.utils)
import time

#%%
if __name__ == '__main__':
    start_time = time.time()

    data_root = r'data'
    num_classes = len(category_information)
    
    continue_from_checkpoint = True
    save_model = True
    num_epochs = 1
    
    train_percentage = 0.8
    test_percentage = 0.2
    percentage_of_data_to_use = 1 #for debugging purposes only option to only use a percentage of the data
    
    batch_size = 4
    learning_rate = 0.0005
    momentum=0.9
    weight_decay=0.0005
    
    weights_save_path = r"data\Models"
    weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\2024-05-13_14-31-24\weights.pth"
    
    save_info_path= r"data\Models\info"
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = LoadDataset(data_root, get_transform(train=True))
    dataset_test = LoadDataset(data_root, get_transform(train=False))
   
    total_samples = len(dataset)
    train_samples = int(train_percentage * total_samples*percentage_of_data_to_use)


    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indices = indices[:int(percentage_of_data_to_use * len(indices))]
    dataset = torch.utils.data.Subset(dataset, indices[:train_samples])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_samples:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
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
    end_time = time.time()
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
    import pandas as pd
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    run_folder = os.path.join(save_info_path, datetime)
    os.makedirs(run_folder, exist_ok=True)
    print(f'saving model info to {run_folder}...')
    if save_model:
        model_info_path = os.path.join(run_folder, "model_info.txt")
        num_images_trained = len(dataset)
        test_set_size = len(dataset_test)
        total_time = end_time-start_time
        
        with open(model_info_path, 'w') as f:
            f.write(f"Number of images trained: {num_images_trained}\n")
            f.write(f"Test set size: {test_set_size}\n")
            f.write(f"Number of epochs: {num_epochs}\n")
            f.write(f"Date: {datetime}\n")
            f.write(f"Continue from checkpoint: {continue_from_checkpoint}\n")
            if continue_from_checkpoint:
                f.write(f"Path to loaded checkpoint: {weights_load_path}\n")
            f.write(f"Total training time: {int(total_time)} seconds ({round(total_time/(60*60), 2)} hours) with time per epoch of {round(total_time / num_epochs)} seconds\n")
            f.write(f"Learning rate: {learning_rate}\n")	
            f.write(f"Momentum: {momentum}\n")
            f.write(f"Weight decay: {weight_decay}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Percentage of data used: {percentage_of_data_to_use*100}%\n")
            f.write(f"Device: {device}\n")
            f.write(f"Optimizer: {optimizer}\n")
            f.write(f"Learning rate scheduler: {lr_scheduler}\n")
            f.write(f"Model: {model}\n")
            
            
       
            with open(os.path.join(run_folder, f"metrics.txt"), 'w') as f:
                for metric in metrics:
                    f.write(f"{metric}\n")

        # save the model
        model_path = os.path.join(run_folder, f"weights.pth")
        torch.save(model.state_dict(), model_path)

        # save IoU info to a file
        
        
        
        IoU_info_path = os.path.join(run_folder, f"IoU_info_test_set.csv")
        coco_eval_bbox = np.empty((len(IoU_info), 5))
        mean_average_precision_box, mean_average_recall_box, mean_average_precision_segm, mean_average_recall_segm = [], [], [], []
        for i, info in enumerate(IoU_info):
            bbox_IoU_array = info.coco_eval['bbox'].stats 
            mean_average_precision_box.append(round(np.mean(bbox_IoU_array[:6]), 6))
            mean_average_recall_box.append(round(np.mean(bbox_IoU_array[6:]), 6))
            segm_IoU_array = info.coco_eval['segm'].stats
            mean_average_precision_segm.append(round(np.mean(segm_IoU_array[:6]), 6))
            mean_average_recall_segm.append(round(np.mean(segm_IoU_array[6:]), 6))

        data = {"epoch": range(num_epochs),
                "mean average precision (box)": mean_average_precision_box,
                "mean average recall (box)": mean_average_recall_box,
                "mean average precision (segm)": mean_average_precision_segm,
                "mean average recall (segm)": mean_average_recall_segm}

        
        df = pd.DataFrame(data)
        df.to_csv(IoU_info_path, index=False)
    print(f'model info saved to {run_folder}')
                 
            
              
        
# %%
