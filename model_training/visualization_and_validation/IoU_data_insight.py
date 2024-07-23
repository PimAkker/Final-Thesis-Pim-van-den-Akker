#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Percentage_results\IoU_results.csv'  # Adjust the path to where the CSV file is located
dataframe = pd.read_csv(data_path)

# Rename 'Unnamed: 0' column to 'datatype'
dataframe = dataframe.rename(columns={'Unnamed: 0': 'datatype'})
mean_average_precision_box = dataframe.copy()

# Filter for the relevant data
mean_avg_precision_box = mean_average_precision_box[mean_average_precision_box['datatype'] == "mean_avg_precision_box"]
mean_avg_recall_box = dataframe[dataframe['datatype'] == "mean_avg_recall_box"]
mean_avg_precision_segm = dataframe[dataframe['datatype'] == "mean_avg_precision_segm"]
mean_avg_recall_segm = dataframe[dataframe['datatype'] == "mean_avg_recall_segm"]
model = dataframe[dataframe['datatype'] == "model_name"]['Combined']

# Set model row as the first row in the dataframes
mean_avg_precision_box.insert(1, 'model', model.values)
mean_avg_recall_box.insert(1, 'model', model.values)
mean_avg_precision_segm.insert(1, 'model', model.values)
mean_avg_recall_segm.insert(1, 'model', model.values)

# Sort the data per the model column
mean_avg_precision_box = mean_avg_precision_box.sort_values(by='model')
mean_avg_recall_box = mean_avg_recall_box.sort_values(by='model')
mean_avg_precision_segm = mean_avg_precision_segm.sort_values(by='model')
mean_avg_recall_segm = mean_avg_recall_segm.sort_values(by='model')

# Remove the model column
# mean_avg_precision_box = mean_avg_precision_box.drop(columns=['model'])
# mean_avg_recall_box = mean_avg_recall_box.drop(columns=['model'])
# mean_avg_precision_segm = mean_avg_precision_segm.drop(columns=['model'])
# mean_avg_recall_segm = mean_avg_recall_segm.drop(columns=['model'])

# Extract model names
model_names = model.values

# Plot the data with the model as the x-axis and the IoU as the y-axis
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'teal', 'lime', 'olive', 'navy', 'maroon', 'aqua', 'silver', 'fuchsia', 'indigo']  # Add more colors if needed
def plot_data(data, title):
    for i, column in enumerate(data.columns.drop('model')[1:]):
        sorted_data = data[['model', column]].sort_values(by='model')
        
        # Fit a curve to the data
        x = np.arange(len(sorted_data['model']))
        y = sorted_data[column]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)

        # Plot the curve
        plt.plot(sorted_data['model'], p(x), color=colors[i])

        # Plot the data points
        plt.plot(sorted_data['model'], sorted_data[column],  label=column,marker='o', linestyle='None', color=colors[i ])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend next to the plot
    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('IoU')
    plt.title(title)
    plt.show()
    
plot_data(mean_avg_precision_box, 'Mean Average Precision Box')
plot_data(mean_avg_recall_box, 'Mean Average Recall Box')
plot_data(mean_avg_precision_segm, 'Mean Average Precision Segmentation')
plot_data(mean_avg_recall_segm, 'Mean Average Recall Segmentation')

# %%
