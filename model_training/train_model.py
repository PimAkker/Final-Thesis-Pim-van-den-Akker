#%%
import torch
import os
import sys
sys.path.append(os.path.join(os.curdir, "utilities"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
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

#%%
if __name__ == '__main__':
    import os
    from utilities.engine import train_one_epoch, evaluate
    import utilities.utils
    from category_information import catogory_information
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    data_root = r'../data'
    num_classes = len(catogory_information)
    
    num_epochs = 5
    train_percentage = 0.8
    batch_size = 16
    learning_rate = 0.005
    momentum=0.9
    weight_decay=0.0005
    
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


    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

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

    

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
#%%
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save the model
    torch.save(model.state_dict(), os.path.join(r"..\data\Models\\", f"model_{datetime}.pth"))

    
    #%%

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    image = read_image(r"..\data\Images\input-0-.jpg")
    mask_true = np.load(r"..\data\Masks\inst-Mask0-0-.npy")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    confidence_threshold = 0.9
    

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    # get the labels from category_information and link them to the pred_labels
    pred_labels = pred["labels"]
    pred_labels = pred_labels[pred["scores"] > confidence_threshold]
    pred_labels = [list(catogory_information.keys())[list(catogory_information.values()).index(x)] for x in pred_labels]
    # change chairs removed to REMCHAIR
    pred_labels = [x.replace("Chairs removed", " REMCHAIR") for x in pred_labels]
   
    pred_boxes = pred["boxes"].long()
    pred_boxes = pred_boxes[pred["scores"] > confidence_threshold]
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
    # output_image = image
    # output_image = torch.zeros_like(image)
    
    mask_true = torch.from_numpy(mask_true)
    # instances are encoded as different colors
    obj_ids = torch.unique(mask_true)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)
    # set images to device
    
    # split the color-encoded mask into a set
    # of binary masks
    masks_true = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8)
    # convert masks too booleon
    masks_true = masks_true.bool()

    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks_true, alpha=0.5, colors="purple")


    masks = (pred["masks"] > confidence_threshold).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()