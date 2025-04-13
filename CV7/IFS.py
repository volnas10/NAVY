import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

first_model = [
    [ 0.00,  0.00,  0.01,  0.00,  0.26,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.00],
    [ 0.20, -0.26, -0.01,  0.23,  0.22, -0.07,  0.07,  0.00,  0.24,  0.00,  0.80,  0.00],
    [-0.25,  0.28,  0.01,  0.26,  0.24, -0.07,  0.07,  0.00,  0.24,  0.00,  0.22,  0.00],
    [ 0.85,  0.04, -0.01, -0.04,  0.85,  0.09,  0.00,  0.08,  0.84,  0.00,  0.80,  0.00]
]

second_model = [
    [ 0.05,  0.00,  0.00,  0.00,  0.60,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00,  0.00],
    [ 0.45, -0.22,  0.22,  0.22,  0.45,  0.22, -0.22, 0.22, -0.45,  0.00,  1.00,  0.00],
    [-0.45,  0.22, -0.22,  0.22,  0.45,  0.22,  0.22, -0.22,  0.45,  0.00,  1.25,  0.00],
    [ 0.49, -0.08,  0.08,  0.08,  0.49,  0.08,  0.08, -0.08,  0.49,  0.00,  2.00,  0.00]
]

def apply_transformation(point, transformation):
    # Create the matrix
    matrix = np.array(transformation[:9]).reshape(3, 3)
    # Create the vector
    vector = np.array(transformation[9:])

    new_point = np.dot(matrix, point) + vector
    return new_point

point = np.array([0, 0, 0])
iterations = 10000
points = [point]
for i in range(iterations):
    # Randomly choose a row
    row = random.randint(0, 3)
    transformation = first_model[row] # or second_model[row]

    # Apply the transformation to the point and save it
    point = apply_transformation(point, transformation)
    points.append(point)

xs, ys, zs = zip(*points)

# Draw points using scatterplot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, s=2, color="black")
#ax.view_init(elev=30, azim=45) # Optional: set different view angle to see second model better
ax.set_title("IFS Fractal")
plt.show()