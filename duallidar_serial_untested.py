import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the serial ports for both sensors
ser1 = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
ser2 = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)

# Set up the Cartesian plot
fig, ax = plt.subplots()
scatter1 = ax.scatter([], [], c='blue', label='Sensor 1')
scatter2 = ax.scatter([], [], c='red', label='Sensor 2')

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.legend()
    return scatter1, scatter2

def update(frame):
    if ser1.in_waiting > 0:
        data1 = ser1.read(ser1.in_waiting).decode('ascii').split()
        distances1 = np.array([float(data1[i]) for i in range(0, len(data1), 2)])
        angles1 = np.linspace(-np.pi/6, np.pi/6, len(distances1))  # Sensor 1 FOV

        x1 = distances1 * np.cos(angles1)
        y1 = distances1 * np.sin(angles1)
        scatter1.set_offsets(np.c_[x1, y1])

    if ser2.in_waiting > 0:
        data2 = ser2.read(ser2.in_waiting).decode('ascii').split()
        distances2 = np.array([float(data2[i]) for i in range(0, len(data2), 2)])
        angles2 = np.linspace(-np.pi/6, np.pi/6, len(distances2))  # Sensor 2 FOV

        # Adjust the position for the second sensor
        x2 = distances2 * np.cos(angles2)
        y2 = distances2 * np.sin(angles2)
        scatter2.set_offsets(np.c_[x2, y2])

    return scatter1, scatter2

ani = FuncAnimation(fig, update, init_func=init, blit=True)
plt.show()
