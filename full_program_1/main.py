import program_init as init
import data_processing as dp
import matplotlib.pyplot as plt
import simulated_data_v2 as sim
import numpy as np

directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder"
directory = "C:/Users/james/OneDrive - University of Southampton/PHYS part 3/BSc Project/data_folder/get_data_06.11"
data_frames = init.get_dataframes(directory, '', False)
#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#fig, ax = plt.subplots(figsize=(10, 10))

def main(df, ax):
    file_name, theta_arr, r_arr, i_arr = dp.get_values(df, +3)
    #dp.plot_polar(ax, file_name, theta_arr, r_arr, i_arr)
    x, y = dp.polar2cartesian(theta_arr, r_arr)
    #dp.plot_cartesian(ax, file_name, x, y)
    max_intensity = max(i_arr)
    scaled_intensities = scale_to_fixed_range(i_arr, 22000)
    print(max(r_arr))
    colour_scale = plt.cm.viridis(scaled_intensities)


    for i in range(0, 16):
        distance = r_arr[i]
        intensity = i_arr[i]
        start_angle = (90 - (48/2)) + i*3 +3
        end_angle = start_angle + 3
        mid_angle = (start_angle + end_angle) / 2
        mid_angle_rad = np.deg2rad(mid_angle)

        #create colour map for intensity

        # Plot the dotted line in the middle of each detector's view
        ax.plot([0, 0 + distance * np.cos(mid_angle_rad)], 
                [0, 0 + distance * np.sin(mid_angle_rad)], 
                 linestyle='dotted', color='gray')
        
        ax.scatter(distance * np.cos(mid_angle_rad), distance * np.sin(mid_angle_rad), color=colour_scale[i], s=30)


        #ax.plot([distance * np.cos(mid_angle_rad-np.deg2rad(1.5)), distance * np.cos(mid_angle_rad+np.deg2rad(1.5))], 
                #[distance * np.sin(mid_angle_rad-np.deg2rad(1.5)), distance * np.sin(mid_angle_rad+np.deg2rad(1.5))], 
                #linestyle='dotted', color='red')

def scale_to_fixed_range(arr, fixed_max=30000):
    scaled_arr = (arr / fixed_max)
    return scaled_arr


data_frames = data_frames[-3::]
titles = ['Control', 'Control Dot', 'Reflector']
fig, axs = plt.subplots(1, 3, figsize=(5, 10))
for i in range(len(data_frames)):
    ax = axs[i]
    df = data_frames[i]
    file_name, theta_arr, r_arr, i_arr = dp.get_values(df, +3)
    #fig, ax = plt.subplots()
    print(file_name)


    main(df, ax)

    ax.set_ylim(0, 1300)
    ax.set_xlim(-150, 150)
    ax.set_xticks([-150, 0, 150])
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(file_name)
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.set_title(titles[i])

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
fig.colorbar(sm, ax=axs, orientation='horizontal')
fig.suptitle('Long corridor Experiment (2)')

plt.show()
