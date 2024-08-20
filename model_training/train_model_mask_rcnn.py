import torch
import os
import sys

root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())


import time
import datetime
import pandas as pd
import numpy as np
from model_training.utilities.coco_eval import *
from model_training.utilities.engine import *
from model_training.utilities.utils import *
from model_training.utilities.transforms import *
from model_training.utilities.dataloader import *
from category_information import category_information
from re import search as regexsearch
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data_root="data",
                 num_classes=4, 
                 continue_from_checkpoint=False, 
                 save_model=True, 
                 num_epochs=1, 
                 train_percentage=0.8, 
                 test_percentage=0.2, 
                 percentage_of_data_to_use=1, 
                 batch_size=4, 
                 learning_rate=0.0005, 
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0005, 
                 weights_save_path="", 
                 weights_load_path="",
                 device=torch.device('cpu'),
                 output_folder=None,
                 plot_metrics_bool=True,
                 seed = None,
                 
                 ):

        self.data_root = data_root
        self.num_classes = num_classes
        self.continue_from_checkpoint = continue_from_checkpoint
        self.save_model = save_model
        self.num_epochs = num_epochs
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.percentage_of_data_to_use = percentage_of_data_to_use
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        # self.momentum = momentum
        self.weight_decay = weight_decay
        self.weights_save_path = weights_save_path
        self.weights_load_path = weights_load_path
        self.device = device
        self.output_folder = output_folder
        self.plot_metrics_bool = plot_metrics_bool
        
        
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        
        self.train_loss_metric = []
        self.val_loss_metric = []
        self.IoU_info = []
        
        self.start_time = None
        self.end_time = None
        
        self.seed = seed

    def setup(self):
        # set the seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        
        path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(path))
        os.chdir(path)

        root_dir_name = 'Blender'
        root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
        os.chdir(root_dir_path)
        sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

        self.model = get_model_instance_segmentation(self.num_classes)
        if self.continue_from_checkpoint:
            self.model.load_state_dict(torch.load(self.weights_load_path))
        self.model.to(self.device)

        # params = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay, eps=self.eps, betas=self.betas)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        

    def train(self):
        self.dataset = LoadDataset(self.data_root, get_transform(train=True))
        self.dataset_test = LoadDataset(self.data_root, get_transform(train=False))

        total_samples = len(self.dataset)
        train_samples = int(self.train_percentage * total_samples * self.percentage_of_data_to_use)

        indices = torch.randperm(len(self.dataset)).tolist()
        indices = indices[:int(self.percentage_of_data_to_use * len(indices))]
        self.dataset = torch.utils.data.Subset(self.dataset, indices[:train_samples])
        self.dataset_test = torch.utils.data.Subset(self.dataset_test, indices[train_samples:])

        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, collate_fn=utils.collate_fn)
        data_loader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=6, collate_fn=utils.collate_fn)

        for epoch in range(self.num_epochs):
            self.train_loss_metric.append(run_one_epoch(self.model, self.optimizer, data_loader, self.device, epoch, print_freq=10, train=True))
            self.val_loss_metric.append(run_one_epoch(self.model, self.optimizer, data_loader_test, self.device, epoch, print_freq=10, train=False))
            self.lr_scheduler.step()
        self.IoU_info.append(evaluate(self.model, data_loader_test, device=self.device))

    def save_model_info(self):
        
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
               
        save_folder = self.output_folder
        os.makedirs(save_folder, exist_ok=True)
        
        print(f'saving model info to {save_folder}...')
        
        if self.save_model:
            
            with open(os.path.join(save_folder, f"metrics.txt"), 'w') as f:
                for metric in self.train_loss_metric:
                    f.write(f"{metric}\n")
                    
                for metric in self.val_loss_metric:
                    f.write(f"Val {metric}\n")
            
            model_info_path = os.path.join(save_folder, "model_info.txt")
            num_images_trained = len(self.dataset)
            test_set_size = len(self.dataset_test)
            total_time = self.end_time - self.start_time

            with open(model_info_path, 'w') as f:
                f.write(f"Number of images trained: {num_images_trained}\n")
                f.write(f"Test set size: {test_set_size}\n")
                f.write(f"Number of epochs: {self.num_epochs}\n")
                f.write(f"Date: {datetime_str}\n")
                f.write(f"Continue from checkpoint: {self.continue_from_checkpoint}\n")
                if self.continue_from_checkpoint:
                    f.write(f"Path to loaded checkpoint: {self.weights_load_path}\n")
                f.write(f"Total training time: {int(total_time)} seconds ({round(total_time/(60*60), 2)} hours) with time per epoch of {round(total_time / self.num_epochs)} seconds\n")
                f.write(f"Learning rate: {self.learning_rate}\n")
                # f.write(f"Momentum: {self.momentum}\n")
                f.write("betas: " + str(self.betas) + "\n")
                f.write(f"eps: {self.eps}\n")
                f.write(f"Weight decay: {self.weight_decay}\n")
                f.write(f"Batch size: {self.batch_size}\n")
                f.write(f"Percentage of data used: {self.percentage_of_data_to_use*100}%\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Optimizer: {self.optimizer}\n")
                f.write(f"Learning rate scheduler: {self.lr_scheduler}\n")
                f.write(f"Model: {self.model}\n")

            model_path = os.path.join(save_folder, f"weights.pth")
            torch.save(self.model.state_dict(), model_path)


            # This code is outdated, as the data is only saved once after running, however it should also work for saving during more runs
            IoU_info_path = os.path.join(save_folder, f"IoU_info_test_set.csv")
            mean_average_precision_box, mean_average_recall_box, mean_average_precision_segm, mean_average_recall_segm = [], [], [], []
            for i, info in enumerate(self.IoU_info):
                bbox_IoU_array = info.coco_eval['bbox'].stats
                mean_average_precision_box.append(round(np.mean(bbox_IoU_array[:6]), 6))
                mean_average_recall_box.append(round(np.mean(bbox_IoU_array[6:]), 6))
                segm_IoU_array = info.coco_eval['segm'].stats
                mean_average_precision_segm.append(round(np.mean(segm_IoU_array[:6]), 6))
                mean_average_recall_segm.append(round(np.mean(segm_IoU_array[6:]), 6))


            data = {
                    # "epoch": range(self.num_epochs), # this can be used if you want to save the IoU for each epoch
                    "mean average precision (box)": mean_average_precision_box,
                    "mean average recall (box)": mean_average_recall_box,
                    "mean average precision (segm)": mean_average_precision_segm,
                    "mean average recall (segm)": mean_average_recall_segm}

            df = pd.DataFrame(data)
            df.to_csv(IoU_info_path, index=False)

        print(f'model info saved to {save_folder}')

    def plot_metrics(self):
        lr = []
        loss = []
        loss_box_reg = []
        loss_mask = []
        loss_objectness = []
        loss_rpn_box_reg = []
        loss_classifier = []
        
        loss_val = []

        for i, metric in enumerate(self.train_loss_metric):
            lr.append(metric.meters['lr'].median)
            loss.append(metric.meters['loss'].median)
            loss_box_reg.append(metric.meters['loss_box_reg'].median)
            loss_mask.append(metric.meters['loss_mask'].median)
            loss_objectness.append(metric.meters['loss_objectness'].median)
            loss_rpn_box_reg.append(metric.meters['loss_rpn_box_reg'].median)
            loss_classifier.append(metric.meters['loss_classifier'].median)
        for i, metric in enumerate(self.val_loss_metric):
            loss_val.append(metric.meters['loss'].median)

        

        plt.plot(lr, label='learning rate')
        plt.plot(loss, label='loss')
        plt.plot(loss_box_reg, label='loss_box_reg')
        plt.plot(loss_mask, label='loss_mask')
        plt.plot(loss_objectness, label='loss_objectness')
        plt.plot(loss_rpn_box_reg, label='loss_rpn_box_reg')
        plt.plot(loss_classifier, label='loss_classifier')
        plt.plot(loss_val, label='val_loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def run(self):
        self.start_time = time.time()
        self.setup()
        self.train()
        self.end_time = time.time()
        if self.plot_metrics_bool:
            self.plot_metrics()
        self.save_model_info()
        print("That's it!")
        
        train_loss =float(regexsearch( r'loss: ([0-9.]+) \([0-9.]+\)',str(self.train_loss_metric[-1])).group(1))
        val_loss = float(regexsearch( r'loss: ([0-9.]+) \([0-9.]+\)',str(self.val_loss_metric[-1])).group(1))
        
        return train_loss, val_loss

if __name__ == '__main__':
    
    trainer = ModelTrainer(data_root= r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\varying_height_no_walls_no_big_varying_model_WITH_object_shift_v3\[]",
                            num_classes=len(category_information),
                            continue_from_checkpoint=False,
                            save_model=True,
                            num_epochs=7,
                            train_percentage=0.8,
                            test_percentage=0.2,
                            percentage_of_data_to_use=1,
                            batch_size=8,
                            learning_rate=1e-4,
                            betas=(0.9, 0.999),
                            eps=1e-5,
                            weight_decay=5e-4,
                            # weights_save_path="",
                            weights_load_path="",
                            device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                            output_folder=r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\varying_height_no_walls_no_big_varying_model_WITH_object_shift_v3_model",
                            plot_metrics_bool=True,
                            seed = 42	
                            
    )
    
    trainer.run()
