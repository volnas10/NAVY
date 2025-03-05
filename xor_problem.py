import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Initialize weights and biases for the hidden layer
        self.weights_input_to_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)

        # Initialize weights and biases for the output layer
        self.weights_hidden_to_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_pass(self, X):
        # Calculate hidden layer activations
        self.hidden_input = np.dot(X, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)

        # Calculate output layer activations
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        self.output_output = self.relu(self.output_input)

        return self.output_output

    def backward_pass(self, X, y, predicted):
        # Calculate error in the output layer
        output_error = y - predicted
        output_delta = output_error * self.relu_derivative(predicted)

        # Calculate error in the hidden layer
        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)

        # Update weights and biases for the output layer
        self.weights_hidden_to_output += self.lr * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += self.lr * np.sum(output_delta, axis=0)

        # Update weights and biases for the hidden layer
        self.weights_input_to_hidden += self.lr * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.lr * np.sum(hidden_delta, axis=0)

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            predicted = self.forward_pass(X)

            # Backward pass
            self.backward_pass(X, y, predicted)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - predicted))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward_pass(X)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("Predictions:")
print(predictions)