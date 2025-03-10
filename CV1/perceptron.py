import numpy as np
import matplotlib.pyplot as plt
import math

class Perceptron:
    def __init__(self, lr=0.01):
        self.weights = None
        self.bias = None
        self.lr = lr

    def fit(self, X, y, epochs=100):
        # Initialize random weights and bias
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand()

        for epoch in range(epochs):
            average_error = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction

                # Update weights and bias
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

                average_error += math.fabs(error)

            if epoch % 100 == 0:
                average_error = average_error / epochs
                print(f"Epoch {epoch}, Loss: {average_error}")


    def predict(self, x):
        return self.sigmoid(np.dot(self.weights, x) + self.bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Generate 100 points for training
x = np.random.uniform(-6, 6, 100)
y = np.random.uniform(-18, 18, 100)
X_train = np.array([x, y]).T

# Classify points
y_line = 3 * x + 2
above = y > y_line
below = y < y_line
on_line = np.isclose(y, y_line, atol=0.5)

y_train = np.zeros(100)
y_train[above] = 1
y_train[on_line] = 0.5 # Points close to 0.5 will be on the line
y_train[below] = 0

# Fit perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train, epochs=800)

# Generate 100 points for testing
x = np.random.uniform(-6, 6, 100)
y = np.random.uniform(-18, 18, 100)
X_test = np.array([x, y]).T

# Predict
y_pred = np.zeros(100)
for i, xi in enumerate(X_test):
    y_pred[i] = perceptron.predict(xi)

above = y_pred > 0.5
below = y_pred <= 0.5
on_line = np.isclose(y_pred, 0.5, atol=0.1)

# Show results
plt.figure(figsize=(8, 6))

x_line = np.linspace(-6, 6, 100)
y_line = 3 * x_line + 2

plt.plot(x_line, y_line, 'r-', label="y = 3x + 2")  # Line
plt.scatter(x[above], y[above], color='blue', label="Above", alpha=0.7)
plt.scatter(x[below], y[below], color='green', label="Below", alpha=0.7)
plt.scatter(x[on_line], y[on_line], color='orange', label="On line", marker='x', s=100)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot of points and line")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()