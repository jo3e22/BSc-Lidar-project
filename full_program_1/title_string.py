import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np
import re


directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
data_frames = init.get_dataframes(directory, 'cube', False)

# Define the pattern for splitting
pattern = r"(\w+)\.(\d+)\.(\d+)\.(\w+)X(\d+)Y(\d+)_leddar(\d+)\.lvm"

for i in range(len(data_frames)):
    df = data_frames[i]
    file_name, theta_arr, r_arr, i_arr = dp.get_values(df, +3)

    # Use re.match to find the pattern
    match = re.match(pattern, file_name)

    if match:
        shape, day, month, grid_identifier, grid_x, grid_y, leddar = match.groups()
        #print(f"Shape: {shape}")
        #print(f"day: {day}")
        #print(f"month: {month}")
        print(f"Identifier: {grid_identifier}")
        print(f"Object X: {grid_x}")
        print(f"Object Y: {grid_y}")
        print(f"Leddar: {leddar}")
    else:
        print("No match found")
    
    print('\n')
