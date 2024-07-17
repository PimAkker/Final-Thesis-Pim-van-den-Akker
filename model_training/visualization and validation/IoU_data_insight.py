#%%
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Percentage_results\IoU_results.csv'
dataframe = pd.read_csv(data_path)

#rename 'Unnamed: 0' column to datatype
dataframe = dataframe.rename(columns={'Unnamed: 0': 'datatype'})
mean_average_precision_box = dataframe.copy()
# make a mask for every row that conotains mean_avg_precision_box
mean_avg_precision_box = mean_average_precision_box[mean_average_precision_box['datatype']=="mean_avg_precision_box"]
mean_avg_recall_box = dataframe[dataframe['datatype']=="mean_avg_recall_box"]
mean_avg_precision_segm = dataframe[dataframe['datatype']=="mean_avg_precision_segm"]
mean_avg_recall_segm = dataframe[dataframe['datatype']=="mean_avg_recall_segm"]
model = dataframe[dataframe['datatype']=="model_name"]['Combined']

# set model row as first row in the dataframes
mean_avg_precision_box.insert(1, 'model', model.values)
mean_avg_recall_box.insert(1, 'model', model.values)
mean_avg_precision_segm.insert(1, 'model', model.values)
mean_avg_recall_segm.insert(1, 'model', model.values)

# sort the data per the model column
mean_avg_precision_box = mean_avg_precision_box.sort_values(by='model')
mean_avg_recall_box = mean_avg_recall_box.sort_values(by='model')
mean_avg_precision_segm = mean_avg_precision_segm.sort_values(by='model')
mean_avg_recall_segm = mean_avg_recall_segm.sort_values(by='model')

# remove the model column
mean_avg_precision_box = mean_avg_precision_box.drop(columns=['model'])
mean_avg_recall_box = mean_avg_recall_box.drop(columns=['model'])
mean_avg_precision_segm = mean_avg_precision_segm.drop(columns=['model'])
mean_avg_recall_segm = mean_avg_recall_segm.drop(columns=['model'])




# %%
# plot the data with the model as the legend and the IoU as the y-axis

plt.figure(figsize=(10, 6))
plt.plot(model.values,mean_avg_precision_box.iloc[:, 1:], marker='o')
plt.legend(mean_avg_precision_box.columns[1:].values)
plt.xticks(rotation=45)

# %%
