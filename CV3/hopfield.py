import numpy as np
import matplotlib.pyplot as plt

patterns = [
    np.array([
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1]
    ]).astype(np.float64),

    np.array([
        [1, 1, 1, 1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, 1, 1, 1, 1]
    ]).astype(np.float64),

    np.array([
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1]
    ]).astype(np.float64)
]


def add_noise(vec, noise_level=0.3):
    if noise_level > 1:
        noise_level = 1
    noise_level *= 2

    # Create vector of random noise up to noise_level
    noise = np.random.uniform(0, noise_level, vec.shape)

    # Create mask for negative and positive values
    mask_neg = vec < 0
    mask_pos = vec > 0
    vec_copy = vec.copy()

    # Add noise to the vector (by adding or substracting it)
    vec_copy[mask_neg] += noise[mask_neg]
    vec_copy[mask_pos] -= noise[mask_pos]

    return vec_copy

# All cells at the same time
def run_synchronous(initial_state, W):
    state = initial_state.copy()

    # For each cell contributions of all other cells are summed up to get the most likely color
    net_input = np.dot(W, state) # Multiply the weight matrix W with the state vector
    state = np.where(net_input > 0, 1, -1) # Apply the rule

    return state

# One cell at a time
def run_async(initial_state, W):
    state = initial_state.copy()

    indices = np.random.permutation(len(state)) # Randomize the order of indices
    for i in indices:
        net_input = W[i, :].dot(state) # Only the i-th row of W is multiplied by the state vector
        new_val = 1 if (net_input > 0) else -1 # Apply the rule

        state[i] = new_val

    return state

# Display all the patterns
def plot_patterns(original, noisy, synchronous, asynchronous):
    num_patterns = len(original)
    fig, axes = plt.subplots(4, num_patterns, figsize=(num_patterns * 2, 8))
    fig.patch.set_facecolor("lightgray")

    row_labels = ["Original", "Noisy", "Sync. Recall", "Async. Recall"]

    for i in range(num_patterns):
        for row, patterns, label in zip(range(4), [original, noisy, synchronous, asynchronous], row_labels):
            ax = axes[row, i]
            ax.imshow(patterns[i], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold", labelpad=10)

        axes[0, i].set_title(f"Pattern {i + 1}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


# Flatten the patterns
p_vectors = np.array([pat.flatten() for pat in patterns])
n_patterns, size = p_vectors.shape

# Each weight represents the likelihood that 2 cells have the same color
# When reconstructing the patters, the most likely color is calculated by checking the sum of all other cells that should have the same color
# @ is matrix multiplication
W = (p_vectors.T @ p_vectors) / n_patterns # Compute weight matrix using Hebbian rule
np.fill_diagonal(W, 0) # Set diagonal to 0 to avoid self-connections

# Add noise to patterns
p_noisy = add_noise(p_vectors, 0.4)

# Run synchronous recall
sync_result = np.array([run_synchronous(p, W) for p in p_noisy])
sync_result = sync_result.reshape(n_patterns, 5, 5)

# Run asynchronous recall
async_result = np.array([run_async(p, W) for p in p_noisy])
async_result = async_result.reshape(n_patterns, 5, 5)

# Show results
plot_patterns(patterns, p_noisy.reshape(n_patterns, 5, 5), sync_result, async_result)