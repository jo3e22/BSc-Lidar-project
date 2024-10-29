import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation

# Simulation parameters
fov = 48  # Field of view in degrees
num_detectors = 16  # Number of detectors
detector_angle = fov / num_detectors
room_size = 15  # Room size in meters
object_speed = 0.5  # meters per second
object_distance = 10.0  # meters from the sensors
object_width = 1.0  # meters

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-room_size / 2, room_size / 2)
ax.set_ylim(0, room_size)
ax.set_aspect('equal')

# Initialize wedges for detectors
detectors = [Wedge((0, 0), room_size, 90 - fov/2 + i*detector_angle, 90 - fov/2 + (i+1)*detector_angle, color='gray', alpha=0.5) for i in range(num_detectors)]
highlighted = [Wedge((0, 0), 0, 90 - fov/2 + i*detector_angle, 90 - fov/2 + (i+1)*detector_angle, color='red', alpha=0.5) for i in range(num_detectors)]
for wedge in detectors + highlighted:
    ax.add_patch(wedge)

obj_sc = ax.scatter([], [], c='red', s=100, label='Object')

def init():
    obj_sc.set_offsets(np.c_[[], []])
    for wedge in detectors + highlighted:
        wedge.set_color('gray')
    return detectors + highlighted + [obj_sc]

def update(frame):
    global object_position
    object_position = (frame * object_speed) % (room_size * 2) - room_size

    # Object's position (moving side to side)
    object_positions = [[object_position, object_distance]]

    # Determine if detectors see the object or the wall
    for i, (detector, highlight) in enumerate(zip(detectors, highlighted)):
        start_angle = 90 - fov/2 + i * detector_angle
        end_angle = start_angle + detector_angle

        angle_to_object = np.degrees(np.arctan2(object_distance, object_position))
        if start_angle <= angle_to_object <= end_angle:
            highlight.set_radius(  object_distance / (np.sin(np.deg2rad(angle_to_object)))  )
            highlight.set_color('red')
        else:
            highlight.set_radius(0)
            highlight.set_color('gray')

    # Update plot
    obj_sc.set_offsets(object_positions)
    return detectors + highlighted + [obj_sc]

# Keep a reference to the animation object
anim = FuncAnimation(fig, update, init_func=init, frames=300, interval=100)
plt.show()

