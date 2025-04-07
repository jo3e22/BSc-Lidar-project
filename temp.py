import os
import numpy as np

def find_lvm_files(directory):
    lvm_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.lvm'):
                lvm_files.append(os.path.join(root, file))
    print(f"Found {len(lvm_files)} .lvm files")  # Debug print
    return lvm_files

def read_lvm_file(file_path):
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

def main(directory):
    lvm_files = find_lvm_files(directory)
    data_arrays = []
    for lvm_file in lvm_files:
        print(f"Processing file: {lvm_file}")  # Debug print
        data = read_lvm_file(lvm_file)
        if data.size > 0:  # Only append non-empty data arrays
            data_arrays.append((os.path.basename(lvm_file), data))
    return data_arrays



if __name__ == "__main__":
    directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
    print(f"Exploring directory: {directory}")  # Debug print
    data_arrays = main(directory)
    for i, (file_name, data) in enumerate(data_arrays):
        print(f"Data from file {i+1} ({file_name}):\n", data)
