import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import random

GRID_SIZE = 9

TRAP = -2
WALL = -1
EMPTY = 0
CHEESE = 1
AGENT = 2

colors = {
    EMPTY: "white",
    CHEESE: "yellow",
    AGENT: "blue",
    WALL: "black",
    TRAP: "red"
}

cmap = mcolors.ListedColormap([colors[TRAP], colors[WALL], colors[EMPTY], colors[CHEESE], colors[AGENT]])

# Function to plot the maze using matplotlib
def plot_maze(maze, agent_pos, name, Q=None):
    display_maze = maze.copy()
    display_maze[agent_pos[0], agent_pos[1]] = AGENT

    plt.clf()
    plt.imshow(display_maze, cmap=cmap, vmin=-2, vmax=2)

    ticks = []
    for i in range(GRID_SIZE):
        ticks.append(0.5 + i)

    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(True, color="black", linewidth=0.5)
    plt.title(name)

    if Q is not None:
        absolute_max_q = float('-inf')
        smallest_max_q = float('inf')

        for key in Q.keys():
            max_q_value = max(Q[key])
            absolute_max_q = max(absolute_max_q, max_q_value)
            smallest_max_q = min(smallest_max_q, max_q_value)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if maze[i, j] == EMPTY:
                    q_values = Q.get((i, j), [0.0, 0.0, 0.0, 0.0])
                    max_q_value = max(q_values)
                    color_intensity = (max_q_value - smallest_max_q) / (absolute_max_q - smallest_max_q)
                    color = plt.cm.plasma(color_intensity)
                    plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color, alpha=0.3))

                    best_action = np.argmax(q_values)
                    if best_action == 0:  # Up
                        plt.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif best_action == 1:  # Down
                        plt.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif best_action == 2:  # Left
                        plt.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                    elif best_action == 3:  # Right
                        plt.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

    plt.show()
    plt.pause(0.01)

# Initialize maze
maze = [
    [EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, TRAP, WALL, EMPTY, EMPTY, WALL, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, WALL, TRAP, WALL, EMPTY, EMPTY, EMPTY],
    [WALL, WALL, EMPTY, EMPTY, EMPTY, WALL, EMPTY, TRAP, WALL],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, WALL, WALL, WALL, EMPTY, WALL, WALL, EMPTY],
    [EMPTY, EMPTY, WALL, WALL, WALL, EMPTY, WALL, EMPTY, EMPTY],
    [WALL, EMPTY, TRAP, EMPTY, EMPTY, EMPTY, WALL, EMPTY, TRAP],
    [EMPTY, EMPTY, EMPTY, EMPTY, TRAP, EMPTY, WALL, EMPTY, CHEESE],
]
maze = np.array(maze)

# Collect possible starting positions (all empty cells)
start_positions = []
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if maze[i, j] == EMPTY:
            start_positions.append((i, j))

action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

# Initialize Q-table with all zeros except walls
Q = {}
for i in range(9):
    for j in range(9):
        if maze[i, j] == WALL:
            continue
        Q[(i, j)] = [0.0, 0.0, 0.0, 0.0]  # Initialize each action's value to zero


learning_rate = 0.2
discount_factor = 0.95
exploration_prob = 0.3 # When training 30% of the time random action will be chosen instead of the best one
epochs = 1000
plot_frequency = 200

for epoch in range(epochs):
    start_pos = random.choice(start_positions) # For each epoch choose a random starting position
    current_i, current_j = start_pos[0], start_pos[1]
    done = False

    # Draw initial maze
    if epoch % plot_frequency == 0:
        plot_maze(maze, (current_i, current_j), f"Epoch {epoch}")
        time.sleep(0.5)

    while not done:
        # Either choose to explore or exploit
        if np.random.random() < exploration_prob:
            a = np.random.randint(4)  # Explore
        else:
            q_values = Q[(current_i, current_j)]
            a = np.argmax(q_values) # Exploit

        # Perform action
        delta_i, delta_j = action_deltas[a]
        new_i = current_i + delta_i
        new_j = current_j + delta_j

        # Handle walls and boundaries
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE and maze[new_i, new_j] != WALL:
            next_i, next_j = new_i, new_j
        else:
            next_i, next_j = current_i, current_j

        # Determine reward based on next cell type
        cell_type = maze[next_i, next_j]

        if cell_type == CHEESE:
            reward = 10.0 # Cheese reached, max reward
            done = True
        elif cell_type == TRAP:
            reward = -20.0 # Trap reached, max penalty
            done = True
        else:
            reward = -1.0 # Small penalty for each step to force shorter paths and not bumping into walls
            done = False

        # Update Q-value for current state and action
        current_q = Q[(current_i, current_j)][a]
        if not done:
            max_next_q = np.max(Q.get((next_i, next_j), [0.0, 0.0, 0.0, 0.0])) # Get max Q-value of the next state
        else:
            max_next_q = 0.0

        # Q-values are updated based on current reward and the maximum Q-value of the next state
        # This way, reward from the cheese is gradually propagated through the entire maze
        new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
        Q[(current_i, current_j)][a] = new_q

        # Move to next state
        current_i, current_j = next_i, next_j

        # Draw maze after each step
        if (epoch + 1) % plot_frequency == 0:
            plot_maze(maze, (current_i, current_j), f"Epoch {epoch + 1}")
            time.sleep(0.05)


# Function to test the trained agent
def test_run(maze, start, show=False):
    done = False
    current_i, current_j = start[0], start[1]

    if show:
        plot_maze(maze, (current_i, current_j), f"Test run from {start}")
        time.sleep(0.5)
    while not done:
        q_values = Q[(current_i, current_j)]
        a = np.argmax(q_values)

        delta_i, delta_j = action_deltas[a]
        new_i = current_i + delta_i
        new_j = current_j + delta_j

        # Compute next position considering walls and boundaries
        if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE and maze[new_i, new_j] != WALL:
            next_i, next_j = new_i, new_j
        else:
            next_i, next_j = current_i, current_j

        cell_type = maze[next_i, next_j]

        if cell_type == CHEESE or cell_type == TRAP:
            done = True

        current_i, current_j = next_i, next_j

        if show:
            plot_maze(maze, (current_i, current_j), f"Test run from {start}")
            time.sleep(0.2)

    return maze[current_i, current_j] == CHEESE

# Show results from 3 positions after training
testing_starts = [(0, 0), (1, 4), (8, 0)]
for start in testing_starts:
    test_run(maze, start, show = True)

# Test on all possible positions
successfull_runs = 0
for start in start_positions:
    if test_run(maze, start):
        successfull_runs += 1

print(f"Success rate: {successfull_runs / len(start_positions) * 100:.2f}%")

#Show the trained Q-table
plot_maze(maze, (0, 0), "Trained Q-table", Q)
