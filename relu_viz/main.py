import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.maximum(0,x)

def surface_plot_piecewise(w1, b1, v1, w2, b2, v2, w3, b3, v3):
    x = np.linspace(-5, 5, 500)
    y = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(x, y)

    # Calculate individual neuron outputs
    H1 = v1 * relu(w1[0] * X + w1[1] * Y + b1)
    H2 = v2 * relu(w2[0] * X + w2[1] * Y  + b2)
    H3 = v3 * relu(w3[0] * X + w3[1] * Y + b3)

    # Total network output
    Z = H1 + H2 + H3

    # 4. Plot the surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.7)

    # Add labels and grid
    ax.set_title("Surface Plot of Piecewise Functions (Polytopes)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Output (Z)")
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Output Value")
    plt.show()


# Define weights, biases, and scaling factors
w1 = [1, 2]  # Weights for Neuron 1
b1 = 1  # Bias for Neuron 1
v1 = 1  # Scaling factor for Neuron 1

w2 = [3, -1]  # Weights for Neuron 2
b2 = -1  # Bias for Neuron 2
v2 = 0.5  # Scaling factor for Neuron 2

w3 = [-2, 1]  # Weights for Neuron 3
b3 = 0.5  # Bias for Neuron 3
v3 = 1.5  # Scaling factor for Neuron 3

# Generate and plot the surface
surface_plot_piecewise(w1, b1, v1, w2, b2, v2, w3, b3, v3)
