#%%
import numpy as np
from matplotlib import pyplot as plt
import simulated_data_v2 as sim

# %%
Env = sim.Environment(500, 500)
Object = sim.Object(shape=20, x_start=(100+Env.border_width), y_start=(400+Env.border_width))

map = sim.add_small_mask_to_large_mask(Env.map, Object.mask, Object.x_start, Object.y_start)

leftLidar = sim.Sensor(0+Env.border_width, 0+Env.border_width, 48/2, map)
rightLidar = sim.Sensor(500-Env.border_width, 0+Env.border_width, -48/2, map)
left_data = leftLidar.scan(map)
right_data = rightLidar.scan(map)

# %%
fig, ax = plt.subplots()
ax.clear()
ax.imshow(map, cmap='gray', interpolation='none', origin='lower')
leftLidar.plot_detector(ax, left_data)
rightLidar.plot_detector(ax, right_data)
ax.axis('off')
ax.set_title('Environment Map')
# %%
