import numpy as np
import matplotlib.pyplot as plt

# Read data from the file
data = np.loadtxt('intial.txt')  # Replace 'your_file.txt' with your file path

# Separate x, y, and temperature values
x = data[:, 0]
y = data[:, 1]
temp = data[:, 2]

# Create a meshgrid for contour plot
x_unique = np.unique(x)
y_unique = np.unique(y)
X, Y = np.meshgrid(x_unique, y_unique)
Z = np.zeros_like(X)

# Fill Z with temperature values at corresponding (x, y) points
for i in range(len(x)):
    xi = np.where(x_unique == x[i])[0][0]
    yi = np.where(y_unique == y[i])[0][0]
    Z[yi, xi] = temp[i]

# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, cmap='viridis')  # Adjust colormap as needed
plt.colorbar(contour)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of Temperature')
plt.show()
