'''
When setting up the lidar configuration to imitate a car, the average width of a new car has been taken to be ~180cm, therfore the lidars should be placed within this distance of each other.
displacment from centre = 80-90cm
'''
#%%
import numpy as np
from matplotlib import pyplot as plt
import simulated_data_v2 as sim
import pandas as pd

# %%
map, leftLidar, rightLidar = sim.init([500, 500], [50], [100, 400], [250-80, 0, 15], [250+80, 0, -15])
#map, leftLidar, rightLidar = sim.init([1000], [50], [350, 400], [500-80, 0, 15], [500+80, 0, -15])
left_data = leftLidar.scan(map)
right_data = rightLidar.scan(map)

# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].clear()
ax[1].clear()

ax[0].imshow(map, cmap='gray', interpolation='none', origin='lower')
leftLidar.plot_detector(ax[0], left_data)
rightLidar.plot_detector(ax[0], right_data)
ax[0].axis('off')
ax[0].set_title('Environment Map')

blind_plot = leftLidar.blind_plot() + rightLidar.blind_plot()
binary_plot = np.where(blind_plot > 0, 1, 0)
ax[1].imshow(binary_plot, cmap='gray', interpolation='none', origin='lower')
#ax[1].axis('off')
ax[1].set_title('Blind Map')
plt.show()


#%%
left_points = np.array(leftLidar.get_points())
right_points = np.array(rightLidar.get_points())

plt.plot(left_points[:, 0], left_points[:, 1], 'ro')
plt.plot(right_points[:, 0], right_points[:, 1], 'bo')
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.show()















# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Normalize the intensities to the highest value and create a DataFrame from the left_data with values rounded to 2 significant figures
left_data[1] /= np.max(left_data[1])
left_data_df = pd.DataFrame(np.round(left_data.T, 2), columns=['Distance (cm)', 'Intensity (relative)'])

# Plot the table
ax.axis('off')
table = ax.table(cellText=left_data_df.values, colLabels=left_data_df.columns, cellLoc='center', loc='center')
table.scale(1, 2.4)  # Set the width and height of the table

plt.show()
# %%
