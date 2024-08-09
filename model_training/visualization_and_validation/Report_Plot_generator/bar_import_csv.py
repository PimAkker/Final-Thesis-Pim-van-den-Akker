# import two csv files and show them in one plot side by side
#%%




import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
csv_real_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model\IoU_results_real_world_with_shift_v2.csv'
csv_synthetic_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model\IoU_results_synthetic_with_shift_v2.csv' 

save_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v2_model'
name = 'IoU_results_mAP_box_syn_vs_real_with_shift_v2.pdf'

save_row = 0

save_path = os.path.join(save_path, name)

real_df = pd.read_csv(csv_real_path)
synthetic_df = pd.read_csv(csv_synthetic_path)



# Delete the first row
# real_df = real_df.iloc[]
# synthetic_df = synthetic_df.iloc[1:]
real_df = real_df.drop(columns=['Unnamed: 0'])
synthetic_df = synthetic_df.drop(columns=['Unnamed: 0'])

real_df.columns = real_df.columns.str.capitalize()
synthetic_df.columns = synthetic_df.columns.str.capitalize()
# rename the columns chair to Chair Correct

real_df = real_df.rename(columns={'Chairs': 'Chair Correct'})
real_df = real_df.rename(columns={'Pillars': 'Pillar Correct'})
real_df = real_df.rename(columns={'Pillars removed': 'Pillars Removed'})
real_df = real_df.rename(columns={'Chairs removed': 'Chairs Removed'})
real_df = real_df.rename(columns={'Chairs new': 'Chairs New'})
real_df = real_df.rename(columns={'Pillars new': 'Pillars New'})

synthetic_df = synthetic_df.rename(columns={'Chairs': 'Chair Correct'})
synthetic_df = synthetic_df.rename(columns={'Pillars': 'Pillar Correct'})
synthetic_df = synthetic_df.rename(columns={'Pillars removed': 'Pillars Removed'})
synthetic_df = synthetic_df.rename(columns={'Chairs removed': 'Chairs Removed'})
synthetic_df = synthetic_df.rename(columns={'Chairs new': 'Chairs New'})
synthetic_df = synthetic_df.rename(columns={'Pillars new': 'Pillars New'})

# Create the bar plot
indices = np.arange(len(real_df.columns))
width = np.min(np.diff(indices)) / 3

real_df.iloc[1:] = real_df.iloc[1:].astype(float)
synthetic_df.iloc[1:] = synthetic_df.iloc[1:].astype(float)
plt.rcParams["font.family"] = "cmr10"  # Set font to Computer Modern Roman
plt.rcParams["axes.grid"] = True  # Enable grid
plt.rcParams["font.size"] = 18  # Set default font size

plt.figure(figsize=(12, 8))
plt.bar(real_df.columns, np.array(real_df.iloc[save_row+1]), label='Real world data', width=-width, align='edge')
plt.bar(synthetic_df.columns, np.array(synthetic_df.iloc[save_row+1]), label='Synthetic data', width=width, align='edge')

# Add labels, title, and legend
plt.ylabel('IoU Score')
plt.xlabel('Categories')

plt.legend()

# Ensure y-axis numbers are displayed
plt.yticks(np.arange(0, 1.1, step=0.1))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

#set the whitespace between the bars
# plt.subplots_adjust(left=0.1, right=0.2, top=0.3, bottom=0.1)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(save_path)
plt.show()




# %%
