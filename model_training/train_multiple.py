import train_model_mask_rcnn
import os 
from category_information import category_information
import torch
if __name__=="__main__":
        multiple_dataset_folder_root = r"data/ablation/"
        output_parent_folder = r"data/ablation/models"

        data_paths = os.listdir(multiple_dataset_folder_root)

        for data_path in data_paths:
                data_path = os.path.join(multiple_dataset_folder_root, data_path)
                trainer = train_model_mask_rcnn.ModelTrainer(data_root= data_path,
                                num_classes=len(category_information),
                                continue_from_checkpoint=False,
                                save_model=True,
                                num_epochs=1,
                                train_percentage=0.8,
                                test_percentage=0.2,
                                percentage_of_data_to_use=1,
                                batch_size=4,
                                learning_rate=0.0005,
                                momentum=0.9,
                                weight_decay=0.0005,
                                weights_save_path="",
                                weights_load_path="",
                                device= torch.device('cpu'),
                                outputs_folder=data_path,
                                plot_metrics_bool=False)
                trainer.run()
