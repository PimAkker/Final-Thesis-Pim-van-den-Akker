#%% 
import pandas as pd
import numpy as np 
import os
import sys

import matplotlib.pyplot as plt 
# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")

#%%

metadata_df = pd.read_csv(r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_height_no_walls_v2\[]\Metadata\object_count_metadata.csv")

metadata_sum = metadata_df.sum()


plt.bar(metadata_df.columns, metadata_sum)
plt.xlabel('Labels')
plt.ylabel('Metadata Sum')
plt.title('Metadata Insight')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees and align them to the right
plt.tight_layout()  # Increase space for x-axis labels
plt.show()

plt.show()


# %%
