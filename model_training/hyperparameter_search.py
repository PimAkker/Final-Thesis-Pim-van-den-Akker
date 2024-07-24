import os
import torch
import random
import itertools
import train_model_mask_rcnn
import os 
from category_information import category_information
import torch
import pandas as pd

def random_grid_search(trainer_class, param_grid, num_trials, fixed_params):

    
        param_combinations = list(itertools.product(*param_grid.values()))
        if num_trials == -1:
            num_trials = len(list(itertools.product(*param_grid.values())))
        random.shuffle(param_combinations)

        trials = param_combinations[:num_trials]
        best_val_loss = float('inf')
        best_params = None
        results = None

        for trial_params in trials:
                params = dict(zip(param_grid.keys(), trial_params))
                params.update(fixed_params)

                print(f"Running trial with params: {params}")

                trainer = trainer_class(**params)
                train_loss, val_loss = trainer.run()
                result = params.copy()
                result['train_loss'] = train_loss
                result['val_loss'] = val_loss 
                       
                if results is None:
                      results = pd.DataFrame([result])
                else:
                      results = pd.concat((results,pd.DataFrame([result])))
                      
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = pd.DataFrame([params])

        
        # save the dataframe to a csv file
        results.to_csv(os.path.join(fixed_params['output_folder'], 'complete_results.csv'))
        best_params.to_csv(os.path.join(fixed_params['output_folder'], f'best_params_loss:{best_val_loss}.csv'))
        print(results)
        print(f"Best validation loss: {best_val_loss}, with params: {best_params}")
        return results

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    multiple_dataset_folder_root = None
    dataset_root = r"/home/student/Pim/code/Blender/data/same_height_no_walls_v4/[]"
    output_parent_folder = r"/home/student/Pim/code/Blender/data/test/parameter_search_output"
    
    percentage_of_data_to_use_list = [1]  # Define your list of percentages
    
    # Hyperparameter search space
    param_grid = {
        # 'num_epochs': [],
        'eps': [1e-4, 1e-5, 1e-6],
        'betas': [(0.9, 0.999), (0.95, 0.999), (0.99, 0.999)],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'weight_decay': [0.0001, 0.0005, 0.001],
    }

    num_trials = 5  # Number of random trials to run

    for i in range(len(percentage_of_data_to_use_list)):
        fixed_params = {
            'num_epochs': 7,
            'data_root': dataset_root,
            'num_classes': len(category_information),
            'batch_size': 8,
            'continue_from_checkpoint': False,
            'save_model': True,
            'train_percentage': 0.8,
            'test_percentage': 0.2,
            'percentage_of_data_to_use': .005,
            'weights_save_path': "",
            'weights_load_path': "",
            'device': device,
            'output_folder': output_parent_folder,
            'plot_metrics_bool': False
        }

        results = random_grid_search(train_model_mask_rcnn.ModelTrainer, param_grid, num_trials, fixed_params)
