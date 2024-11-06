import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataObject:
    def __init__(self, file_name, data):
        self.file_name = file_name
        self.data = data
        self.theta, self.r, self.intensity = self.get_data()
    
    def get_data(self):
        theta = []
        r = []
        intensity = []
        for i in range(0, len(data), 16):
            thetai = np.linspace(90-45/2, 90+45/2, 16)
            thetai = np.radians(theta)
            ri = data[i:i+16, 0]
            intensityi = data[i:i+16, 1]

            theta.append(thetai)
            r.append(ri)
            intensity.append(intensity)
        return theta, r, intensity
    
    def plot_data(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        #plot each section of 16 rows as a separate line
        
        ax.set_title(self.file_name)
        ax.legend()
        plt.show()

def find_lvm_files(directory, testing=False):
    lvm_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.lvm'):
                lvm_files.append(os.path.join(root, file))
    if testing:
        print(f"Found {len(lvm_files)} .lvm files")  # Debug print
    return lvm_files

def read_lvm_file(file_path, testing=False):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Filter out lines that are empty or contain non-numeric data
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and all(char.isdigit() or char in '.- \t' for char in stripped_line):
                filtered_lines.append(stripped_line)
        
        # Convert the filtered lines to a single string and load the data
        if filtered_lines:
            filtered_data = '\n'.join(filtered_lines)
            data = np.loadtxt(filtered_data.splitlines(), delimiter='\t')
        else:
            data = np.array([])
    except ValueError as e:
        print(f"Error reading {file_path}: {e}")
        data = np.genfromtxt(file_path, delimiter='\t', skip_header=0, invalid_raise=False)
    return data

def main(directory, testing=False):
    lvm_files = find_lvm_files(directory)
    data_arrays = []
    for lvm_file in lvm_files:
        if testing:
            print(f"Processing file: {lvm_file}")  # Debug print
        data = read_lvm_file(lvm_file)
        if data.size > 0:  # Only append non-empty data arrays
            data_arrays.append((os.path.basename(lvm_file), data))
    return data_arrays

def plot_data(data, file_name):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #plot each section of 16 rows as a separate line
    for i in range(0, len(data), 16):
        theta = np.linspace(90-45/2, 90+45/2, 16)
        theta = np.radians(theta)
        y = data[i:i+16, 0]
        ax.plot(theta, y, label=f"Section {i//16}")
    ax.set_title(file_name)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    testing = False
    directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
    if testing:
        print(f"Exploring directory: {directory}")  # Debug print
    data_arrays = main(directory)
    df = pd.DataFrame()
    for i, (file_name, data) in enumerate(data_arrays):
        #print(f"Data from file {i+1} ({file_name}):\n", data)
        data_object = DataObject(file_name, data)
        t, r, intensity = data_object.get_data()
        print(f"Data from file {i+1} ({file_name}):\n", t, r, intensity)

