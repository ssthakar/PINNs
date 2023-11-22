import matplotlib.pyplot as plt
import numpy as np

# Read data from the first file
file1_data = np.loadtxt('testXY.txt')
x1_values, y1_values = file1_data[:, 0], file1_data[:, 1]

# Read data from the second file
file2_data = np.loadtxt('test.txt')
y2_values = file2_data
sin_values = np.sin(np.pi * x1_values) * np.sin(np.pi * y1_values)

# Plot both on the same graph
plt.figure(figsize=(10, 5))

# Plotting from the second file
plt.plot(y2_values, linestyle='--', color='r', label='PINNs prediction',linewidth=5)

# Plotting sin(pi * x) * sin(pi * y) from the first file
plt.plot(sin_values, linestyle='-', color='g', label='sin(pi * x) * sin(pi * y)')

# Customize the plot
plt.title('PINNs prediction vs. sin(pi * x) * sin(pi * y) at x = 0.25')
plt.xlabel('y (x 1e-2 m)')
plt.ylabel('T')
plt.legend()
plt.show()

