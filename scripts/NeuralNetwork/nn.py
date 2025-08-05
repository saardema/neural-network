import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Define the structure of the neural network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values
        self.weights_ih = np.random.randn(self.input_size, self.hidden_size)
        self.weights_ho = np.random.randn(self.hidden_size, self.output_size)
        self.bias_h = np.zeros((1, self.hidden_size))
        self.bias_o = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def forward(self, X):
        # Perform forward propagation
        self.hidden_layer_input = np.dot(X, self.weights_ih) + self.bias_h
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_ho) + self.bias_o
        predicted_output = self.sigmoid(self.output_layer_input)
        return predicted_output

    def backpropagation(self, X, y, learning_rate):
        # Perform backpropagation to adjust weights and biases
        predicted_output = self.forward(X)

        # Calculate output layer error
        output_error = y - predicted_output
        output_delta = output_error * self.sigmoid_derivative(predicted_output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_ho.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_ho += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_ih += X.T.dot(hidden_delta) * learning_rate
        self.bias_o += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_h += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Train the neural network for a specified number of epochs
        for epoch in range(epochs):
            self.backpropagation(X, y, learning_rate)
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


# Example usage:
if __name__ == "__main__":
    # Define the input data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create a neural network with 2 input nodes, 4 hidden nodes, and 1 output node
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the network for 10000 epochs with a learning rate of 0.1
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the trained network
    print("\nPredictions after training:")
    for i in range(len(X)):
        prediction = nn.forward(X[i])
        print(f"Input: {X[i]}, Predicted Output: {prediction}")
