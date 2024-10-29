import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the serial port
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Set up the polar plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
sc = ax.scatter([], [], c=[], cmap='viridis', s=5)  # Use colormap for intensity

def init():
    ax.set_xlim(0, np.pi)  # Adjust the limit based on your sensor's FOV
    ax.set_ylim(0, 10)  # Adjust the range based on your sensor's range
    return sc,

def update(frame):
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting).decode('ascii').split()
        angles = np.linspace(-np.pi/6, np.pi/6, len(data)//2)  # Assuming half data points are distances
        distances = np.array([float(data[i]) for i in range(0, len(data), 2)])
        intensities = np.array([float(data[i]) for i in range(1, len(data), 2)])
        
        sc.set_offsets(np.c_[angles, distances])
        sc.set_array(intensities)
    return sc,

ani = FuncAnimation(fig, update, init_func=init, blit=True)
plt.colorbar(sc)
plt.show()
