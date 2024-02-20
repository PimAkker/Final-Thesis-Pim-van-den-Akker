#%%
import csv
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_csv(file_path):
    try:
        data = np.genfromtxt(file_path, delimiter=',')
        return data
    except FileNotFoundError:
        print("File not found.")
        sys.exit()
    except Exception as e:
        print("An error occurred while reading the CSV file:", str(e))
        sys.exit()

def get_scans(data):
    filtered_cloud_array = []
    for i in range(int(data[1:, 0].max())):
        filtered_cloud = data[data[:, 0] == (i + 1)]
        if filtered_cloud.shape[0] > 0:
            filtered_cloud_array.append(filtered_cloud)
    print("Total number of scans:", len(filtered_cloud_array))
    return filtered_cloud_array

def plot_scan(cloud, file_name=None):
    plt.figure()
    plt.scatter(cloud[:, 1], cloud[:, 2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'file: {file_name}')
    plt.show()
def animate_scan(scans):
    # popout a window with the plot
    plt.ion()
    
    plt.figure()  # Create a single figure outside the loop

    for scan in scans:
        plot_scan(scan)
        plt.pause(0.1)
        plt.clf()  # Clear the figure
def get_all_csv_file_paths_from_folder(folder_path):
    import os
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_paths.append(os.path.join(root, file))
    file_name = [file.split("\\")[-1] for file in file_paths]
    return file_paths, file_name
if __name__ == "__main__":
    csvFile = r"real_world_data\csv_files\rosbag1_box.csv"
    data = read_csv(csvFile)
    scans = get_scans(data)
    # animate_scan(scans)
    file_paths, file_name = get_all_csv_file_paths_from_folder(r"real_world_data\csv_files")
    
    for i, file_path in enumerate(file_paths):
        data = read_csv(file_path)
        scans = get_scans(data)
        plot_scan(scans[0], file_name[i])

# %%
