import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Weights and biases for (input > hidden) layer
        self.weights_ih = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)

        # Weights and biases for (hidden > output) layer
        self.weights_ho = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    # Relu activation, sigmoid didn't work
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_pass(self, X):
        # Calculate hidden layer activations
        self.hidden_input = np.dot(X, self.weights_ih) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)

        # Calculate output layer activations
        self.output_input = np.dot(self.hidden_output, self.weights_ho) + self.bias_output
        self.output_output = self.relu(self.output_input)

        return self.output_output

    def backward_pass(self, X, y, predicted):
        # Calculate error in the output layer
        output_error = y - predicted
        output_delta = output_error * self.relu_derivative(predicted)

        # Calculate error in the hidden layer
        hidden_error = np.dot(output_delta, self.weights_ho.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)

        # Update weights and biases for the output layer
        self.weights_ho += self.lr * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += self.lr * np.sum(output_delta, axis=0)

        # Update weights and biases for the hidden layer
        self.weights_ih += self.lr * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.lr * np.sum(hidden_delta, axis=0)

    def fit(self, X, y, epochs=1000):
        last_loss = 1
        epoch = 0
        while epoch < epochs:
            # Forward pass
            predicted = self.forward_pass(X)

            # Backward pass
            self.backward_pass(X, y, predicted)

            # Since XOR is a simple problem, we can check the loss every epoch and try new weights if the loss is high
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - predicted))
                if last_loss - loss < 0.001:
                    if loss > 0.25:
                        # Reset weights and biases
                        self.weights_ih = np.random.rand(self.input_size, self.hidden_size)
                        self.bias_hidden = np.zeros(self.hidden_size)
                        self.weights_ho = np.random.rand(self.hidden_size, self.output_size)
                        self.bias_output = np.zeros(self.output_size)

                        epoch = 0
                        last_loss = 1

                        print("Training stuck, resetting")
                        continue
                    else:
                        print("Loss not decreasing, stopping training")
                        break

                print(f"Epoch {epoch}, Loss: {loss}")
                last_loss = loss
            epoch += 1

    def predict(self, X):
        return self.forward_pass(X)

# Only 4 possible inputs for XOR
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([[0], [0], [1], [1]])

mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("Predictions:")
print(predictions)