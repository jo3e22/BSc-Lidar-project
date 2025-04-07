import os
import numpy as np
import pandas as pd

def find_lvm_files(directory, testing=False):
    '''
    This function searches the given directory for .lvm files and returns a list of their paths.
    Inputs:
        directory (str): The directory to search for .lvm files.
        testing (bool): If True, debug print statements will be enabled.
    Outputs:
        lvm_files (list): A list of the paths to the .lvm files found in the directory.
    '''
    lvm_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.lvm'):
                lvm_files.append(os.path.join(root, file))
    if testing:
        print(f"Found {len(lvm_files)} .lvm files")  # Debug print
    return lvm_files

def read_lvm_file(file_path, testing=False):
    '''
    This function reads the data from an .lvm file and returns it as a pandas DataFrame.
    Inputs:
        file_path (str): The path to the .lvm file to read.
        testing (bool): If True, debug print statements will be enabled.
    Outputs:
        df (pd.DataFrame): A pandas DataFrame containing the data from the .lvm file. The DataFrame will have an attribute 'filename' containing the name of the file it was read from.
    '''
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
    
    # Convert the data to a pandas DataFrame
    if data.size > 0:
        df = pd.DataFrame(data, columns=['Distance (cm)', 'Intensity (relative)'])
        df.attrs['filename'] = os.path.basename(file_path)
    else:
        df = pd.DataFrame()
    
    return df

def main(directory, string = None, testing=False):
    '''
    This function is the main logic for the program. It will search the given directory for .lvm files and read them into pandas DataFrames.
    Inputs:
        directory (str): The directory to search for .lvm files.
        testing (bool): If True, debug print statements will be enabled.
    Outputs:
        data_frames (list): A list of pandas DataFrames containing the data from the .lvm files. Each DataFrame will have an attribute 'filename' containing the name of the file it was read from.
    '''
    lvm_files = find_lvm_files(directory)
    data_frames = []
    for lvm_file in lvm_files:
        if testing:
            print(f"Processing file: {lvm_file}")  # Debug print
        df = read_lvm_file(lvm_file)
        if not df.empty:  # Only append non-empty DataFrames
            if string is not None:
                if string in df.attrs['filename']:
                    data_frames.append(df)
            else:
                data_frames.append(df)
    return data_frames

def get_dataframes(directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder", string=None, testing=False):
    '''
    This function is the entry point for the program. It will search the given directory for .lvm files and read them into pandas DataFrames.
    Inputs:
        directory (str): The directory to search for .lvm files.
        testing (bool): If True, debug print statements will be enabled.
    Outputs:
        data_frames (list): A list of pandas DataFrames containing the data from the .lvm files. Each DataFrame will have an attribute 'filename' containing the name of the file it was read from.
    '''
    return main(directory, string, testing)

if __name__ == "__main__":
    testing = False
    directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
    if testing:
        print(f"Exploring directory: {directory}")  # Debug print
    data_frames = main(directory)
    for df in data_frames:
        print(f"DataFrame for {df.attrs['filename']}:\n{df}")
