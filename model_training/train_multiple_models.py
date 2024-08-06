import train_model_mask_rcnn
import os 
from category_information import category_information
import torch
if __name__=="__main__":
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        multiple_dataset_folder_root = None
        dataset_root = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_height_no_walls_v4\[]"
        output_parent_folder = r"data/test/percentage_models"

        # list dirs in dataset_root
        dataset_root_dirs = os.listdir(dataset_root)
        
        
        

        for i in range(len(dataset_root_dirs)):
                
                trainer = train_model_mask_rcnn.ModelTrainer(data_root= dataset_root,
                                num_classes=len(category_information),
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
                                device= device ,
                                outputs_folder=os.path.join(output_parent_folder,f"{dataset_root_dirs[i]}"),
                                plot_metrics_bool=False)
                trainer.run()
