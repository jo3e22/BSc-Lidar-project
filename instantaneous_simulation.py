#%%
import numpy as np
from matplotlib import pyplot as plt
import simulated_data_v2 as sim

# %%
Env = sim.Environment(30, 30)
Object = sim.Object(shape=1, x_start=1, y_start=4)

map = sim.add_small_mask_to_large_mask(Env.map, Object.mask, Object.y_start, Object.x_start)

#leftLidar = sim.Sensor(0+Env.border_width, 0+Env.border_width, -45, map)

# %%
plt.imshow(map, cmap = 'gray')
plt.show()
# %%
