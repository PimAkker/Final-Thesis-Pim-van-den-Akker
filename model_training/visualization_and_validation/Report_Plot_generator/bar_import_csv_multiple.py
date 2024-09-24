#%%
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

csv_paths = {
    "Reference": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model\IoU_results_real_world_with_shift_v2.csv',
    "LiDAR High Freq.": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\lidar_high_model\IoU_info_rwv3.csv',
    "LiDAR Low Freq": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\lidar_low_model\IoU_info_rwv3.csv',
    "LiDAR Block Size": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\LiDAR_block_size_model\IoU_info_real_world_v3_lidar_block_size.csv',
    "Pillar Width": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\pillar_width_model\IoU_info_rwv3.csv',
    "Chair Size": r"C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\chair_size_model\IoU_info_rwv3_chair size.csv",
    'Pillar Chair Block Size':r"C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\chair_pillar_lidar_block_model\IoU_info_real_world_v3_pillar_chair_lidar.csv"
}

# csv_paths = {
#     "Reference":r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\Custom_vs_General_synthetic\IoU_results_customized_on_synthetic.csv",
#     "LiDAR High Freq.": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\LiDAR_high_freq_model_on_synthetic.csv',
#     "LiDAR Low Freq": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\LiDAR_low_freq_model_on_synthetic.csv',
#     "LiDAR Block Size": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\LiDAR_block_size_model_on_synthetic.csv',
#     "Pillar Width": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\pillar_width_model_on_synthetic.csv',
#     "Chair Size": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\chair_size_model_on_synthetic.csv",
#     'Pillar Chair Block Size':r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\syn_ablation_results\chair_pillar_lidar_block_model_on_synthetic.csv"
# }






# csv_paths = {
#     "Customized": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_no_object_shift_big_v5_model\IoU_info_rwv3_bigshiftv5.csv",
#     "General": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\varying_height_no_walls_no_big_varying_model_WITH_object_shift_v3_model\IoU_results_with_object_shift_v3.csv"
# }

# csv_paths = { 
        
#         "Customized": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\Custom_vs_General_real\IoU_results_customized_model_on_real.csv",
#         "General": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\Custom_vs_General_real\IoU_results_generalized_model_on_syn.csv",
# }    
# csv_paths = {
#     "Customized":r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\Custom_vs_General_synthetic\IoU_results_customized_on_synthetic.csv",
#     "General": r"C:\Users\pimde\OneDrive\thesis\Blender\data\Results\Custom_vs_General_synthetic\IoU_results_general_on_synthetic.csv",
# }    


# save_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model'
name = 'IoU_results_mAP_box_syn_vs_real_with_shift_v2.pdf'

save_row = 0

# save_path = os.path.join(save_path, name)

dfs = []  # List to store the dataframes

# Read and store the dataframes for each CSV file
for label, csv_path in csv_paths.items():
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 0'])
    df.columns = df.columns.str.capitalize()
    df = df.rename(columns={'Chairs': 'Chair Correct',
                            'Pillars': 'Pillar Correct',
                            'Pillars removed': 'Pillars Removed',
                            'Chairs removed': 'Chairs Removed',
                            'Chairs new': 'Chairs New',
                            'Pillars new': 'Pillars New'})
    df.iloc[1:] = df.iloc[1:].astype(float)
    dfs.append((label, df)) # Store the label along with the dataframe

# Create the bar plot
indices = np.arange(len(dfs[0][1].columns))
width = np.min(np.diff(indices)) / (len(dfs) + 1)

plt.rcParams["font.family"] = "cmr10"  # Set font to Computer Modern Roman
plt.rcParams["axes.grid"] = True  # Enable grid
plt.rcParams["font.size"] = 20  # Set default font size

# plt.figure(figsize=(11, 9))
plt.figure(figsize=(28.1, 10.5))

# plt.figure()
# 
# Plot each dataframe
# Create a new dataframe to store the values
new_df = pd.DataFrame(columns=dfs[0][1].columns)
combined_df = pd.DataFrame(columns=dfs[0][1].columns)
# Plot each dataframe and save the values
for i, (label, df) in enumerate(dfs):
    plt.bar(indices + i * width, np.array(df.iloc[save_row+1]), label=label, width=width, align='edge')
    new_df.loc[label] = df.iloc[save_row+1]
    #save just the "Combined" row values
    combined_df[label] = df.iloc[save_row+1].values
combined_df = combined_df.T
combined_df.columns = dfs[0][1].columns
combined_df.index.name = 'Model'
combined_df.reset_index(inplace=True)


# Set the x-axis ticks to be the column names
plt.xticks(indices+0.5, dfs[0][1].columns)

# Add labels, title, and legend
plt.ylabel('mAP')
plt.xlabel('Categories')
# plt.legend(loc='lower left', bbox_to_anchor=(.846, 0.55), title='Fixed Parameters:')
plt.legend()
# Ensure y-axis numbers are displayed
plt.yticks(np.arange(0.1, 1.1, step=0.1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig(r'C:\Users\pimde\OneDrive\thesis\graphs\synthetic_mAP_box_ablation.pdf')
# plt.savefig(save_path)
plt.show()


# %%
