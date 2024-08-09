#%%
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

csv_paths = {
    "Reference": r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model\IoU_results_real_world_with_shift_v2.csv',
    "LiDAR High Freq.": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\csv_results\IoU_info_rwv3_lidar_high.csv',
    "LiDAR Block Size": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\csv_results\IoU_info_rwv3_block_size_model.csv',
    "LiDAR Low Freq": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\csv_results\IoU_info_rwv3_lidar_low.csv',
    "Pillar Width": r'C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\csv_results\IoU_info_rwv3_pillar_width.csv',
    "Chair Size": r"C:\Users\pimde\OneDrive\thesis\Blender\data\ablation\csv_results\IoU_info_rwv3_chair size.csv"
}

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

plt.figure(figsize=(25.1, 8.9))

# Plot each dataframe
for i, (label, df) in enumerate(dfs):
    plt.bar(indices + i * width, np.array(df.iloc[save_row+1]), label=label, width=width, align='edge')

# Set the x-axis ticks to be the column names
plt.xticks(indices+0.5, dfs[0][1].columns)

# Add labels, title, and legend
plt.ylabel('mAP')
plt.xlabel('Categories')
plt.legend(loc='lower left', bbox_to_anchor=(.846, 0.55))
# plt.legend()
# Ensure y-axis numbers are displayed
plt.yticks(np.arange(0.1, 1.1, step=0.1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig(r'C:\Users\pimde\OneDrive\thesis\graphs\mAP_box_ablation.pdf')
# plt.savefig(save_path)
plt.show()

# %%
