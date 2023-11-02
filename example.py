import numpy as np
from idw import idw
import matplotlib.pyplot as plt

# Create sample data points for interpolation
np.random.seed(0)
x_points = np.random.randint(0, 100, 2)
y_points = np.random.randint(0, 100, 2)
values = np.random.randint(1, 100, 2)

# Define the grid for interpolation
x_grid, y_grid = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
interpolated_values = np.zeros_like(x_grid)

interpolated_values = idw(x_points, y_points, values, x_grid, y_grid, 2, 1e-6)

plt.figure(figsize=(10, 8))
plt.scatter(x_points, y_points, c=values, cmap='viridis', s=100, edgecolor='k', linewidth=1, alpha=0.8)
plt.contourf(x_grid, y_grid, interpolated_values, levels=50, cmap='viridis', alpha=0.8)
plt.colorbar(label='Interpolated Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Inverse Distance Weighted Interpolation no Numba')
plt.show()