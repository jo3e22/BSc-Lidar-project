import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation

# Simulation parameters
fov = 48  # Field of view in degrees
num_detectors = 16  # Number of detectors
detector_angle = fov / num_detectors
room_size = 10  # Room size in meters
object_speed = 0.5  # meters per second
object_distance = 8.0  # meters from the sensors
object_width = 1.0  # meters
sensor_distance = 5.0  # distance between the two sensors

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-room_size / 2, room_size / 2)
ax.set_ylim(0, room_size)
ax.set_aspect('equal')

# Initialize wedges for detectors of the first sensor angled inward
angle_offset = 20  # degrees to angle inward
detectors1 = [Wedge((-sensor_distance / 2, 0), 100, 90 - fov/2 + i*detector_angle - angle_offset, 90 - fov/2 + (i+1)*detector_angle - angle_offset, color='gray', alpha=0.5) for i in range(num_detectors)]
highlighted1 = [Wedge((-sensor_distance / 2, 0), 0, 90 - fov/2 + i*detector_angle - angle_offset, 90 - fov/2 + (i+1)*detector_angle - angle_offset, color='red', alpha=0.5) for i in range(num_detectors)]

# Initialize wedges for detectors of the second sensor angled inward
detectors2 = [Wedge((sensor_distance / 2, 0), 100, 90 - fov/2 + i*detector_angle + angle_offset, 90 - fov/2 + (i+1)*detector_angle + angle_offset, color='gray', alpha=0.5) for i in range(num_detectors)]
highlighted2 = [Wedge((sensor_distance / 2, 0), 0, 90 - fov/2 + i*detector_angle + angle_offset, 90 - fov/2 + (i+1)*detector_angle + angle_offset, color='red', alpha=0.5) for i in range(num_detectors)]

for wedge in detectors1 + highlighted1 + detectors2 + highlighted2:
    ax.add_patch(wedge)

obj_sc = ax.scatter([], [], c='red', s=100, label='Object')

def init():
    obj_sc.set_offsets(np.c_[[], []])
    for wedge in detectors1 + highlighted1 + detectors2 + highlighted2:
        wedge.set_color('gray')
    return detectors1 + highlighted1 + detectors2 + highlighted2 + [obj_sc]

def update(frame):
    global object_position
    object_position = (frame * object_speed) % (room_size * 2) - room_size

    # Object's position (moving side to side)
    object_positions = [[object_position, object_distance]]

    # Determine if detectors of the first sensor see the object or the wall
    for i, (detector, highlight) in enumerate(zip(detectors1, highlighted1)):
        start_angle = 90 - fov/2 + i * detector_angle - angle_offset
        end_angle = start_angle + detector_angle

        angle_to_object = np.degrees(np.arctan2(object_distance, object_position + sensor_distance / 2))
        if start_angle <= angle_to_object <= end_angle:
            highlight.set_radius(np.sqrt(object_distance**2 + (object_position + sensor_distance / 2)**2))
            highlight.set_color('red')
        else:
            highlight.set_radius(0)
            highlight.set_color('gray')

    # Determine if detectors of the second sensor see the object or the wall
    for i, (detector, highlight) in enumerate(zip(detectors2, highlighted2)):
        start_angle = 90 - fov/2 + i * detector_angle + angle_offset
        end_angle = start_angle + detector_angle

        angle_to_object = np.degrees(np.arctan2(object_distance, object_position - sensor_distance / 2))
        if start_angle <= angle_to_object <= end_angle:
            highlight.set_radius(np.sqrt(object_distance**2 + (object_position - sensor_distance / 2)**2))
            highlight.set_color('blue')
        else:
            highlight.set_radius(0)
            highlight.set_color('gray')

    # Update plot
    obj_sc.set_offsets(object_positions)
    return detectors1 + highlighted1 + detectors2 + highlighted2 + [obj_sc]

# Keep a reference to the animation object
anim = FuncAnimation(fig, update, init_func=init, frames=300, interval=100)
plt.show()
