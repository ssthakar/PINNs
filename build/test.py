import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# Read coordinates from the file
coordinates = np.loadtxt('total.txt')
x, y = coordinates[:, 0], coordinates[:, 1]

# Read temperatures from the file
data = np.loadtxt('initial.txt')
temperatures = data[:, 3]
# Reshape the temperatures into a 2D grid
triang = tri.Triangulation(x, y)
# Z = temperatures.reshape(X.shape)

# Create a contour plot
plt.tricontourf(triang, temperatures, cmap='viridis',levels=np.linspace(-1.00001,1.00001,10))
plt.colorbar(label='Temperature')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Temperature Contour Plot')
plt.gca().set_aspect('equal')
plt.show()
