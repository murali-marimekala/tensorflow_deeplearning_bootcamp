import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Print the TensorFlow version
print(tf.__version__)

# Create a scalar constant tensor with the value 7
scalar = tf.constant(7)
print(scalar)
# Print the number of dimensions (rank) of the scalar tensor
print(f"scalar number of dimensions are", scalar.ndim)
# Print the shape of the scalar tensor
print(f"scalar shape is", scalar.shape)

# Create a vector constant tensor with values [10, 10]
vector = tf.constant([10, 10])
print(vector)
# Print the number of dimensions (rank) of the vector tensor
print(f"vector number of dimensions are", vector.ndim)
# Print the shape of the vector tensor
print(f"vector shape is", vector.shape)

# Create a matrix constant tensor with values [[10, 7], [7, 10]]
matrix = tf.constant([[10, 7], [7, 10]])
print(matrix)
# Print the number of dimensions (rank) of the matrix tensor
print(f"matrix number of dimensions are", matrix.ndim)
# Print the shape of the matrix tensor
print(f"matrix shape is", matrix.shape)

# Create another matrix constant tensor with specific data type (float16)
another_matrix = tf.constant([[10., 7.], [3., 2.], [8., 9.]], dtype=tf.float16)
print(another_matrix)
# Print the number of dimensions (rank) of the another_matrix tensor
print(f"another_matrix number of dimensions are", another_matrix.ndim)
# Print the shape of the another_matrix tensor
print(f"another_matrix shape is", another_matrix.shape)

# Plotting the tensors
fig = plt.figure(figsize=(12, 8))

# Plot scalar
ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title("Scalar")
ax1.text2D(0.5, 0.5, f"Value: {scalar.numpy()}\nRank: {scalar.ndim}\nShape: {scalar.shape}", fontsize=12, ha='center', va='center', transform=ax1.transAxes)
ax1.set_axis_off()

# Plot vector
ax2 = fig.add_subplot(222, projection='3d')
ax2.set_title("Vector")
ax2.bar3d([0, 1], [0, 0], [0, 0], [0.1, 0.1], [0.1, 0.1], vector.numpy(), shade=True)
ax2.set_xlabel(f"Rank: {vector.ndim}, Shape: {vector.shape}")

# Plot matrix
ax3 = fig.add_subplot(223, projection='3d')
ax3.set_title("Matrix")
x, y = np.meshgrid(range(matrix.shape[0]), range(matrix.shape[1]))
ax3.plot_surface(x, y, matrix.numpy().T, cmap='viridis')
ax3.set_xlabel(f"Rank: {matrix.ndim}, Shape: {matrix.shape}")

# Plot another matrix
ax4 = fig.add_subplot(224, projection='3d')
ax4.set_title("Another Matrix")
x, y = np.meshgrid(range(another_matrix.shape[0]), range(another_matrix.shape[1]))
ax4.plot_surface(x, y, another_matrix.numpy().T, cmap='viridis')
ax4.set_xlabel(f"Rank: {another_matrix.ndim}, Shape: {another_matrix.shape}")

plt.tight_layout()
plt.show()