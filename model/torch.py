import numpy as np

class NeuralNetwork:
    """
        A simple feedforward neural network class implementing a fully connected
        architecture from scratch using only NumPy. This neural network supports 
        one hidden layer and uses the sigmoid activation function.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
            Initializes the neural network with random weights and biases.

            Args:
            - input_size (int): Number of input features.
            - hidden_size (int): Number of neurons in the hidden layer.
            - output_size (int): Number of output neurons.
            - learning_rate (float): Learning rate for gradient descent updates.
        """
        # Initialize weights and biases for input -> hidden layer
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Initialize weights and biases for hidden -> output layer
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
        # Learning rate (how large the gradient descent step is)
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        """
            Computes the sigmoid activation function.
            
            Sigmoid squashes values between 0 and 1, and is useful for binary classification.

            Args:
                - z (numpy array): Linear combination of inputs and weights.

            Returns:
                - Sigmoid-activated output (numpy array).
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """
            Computes the derivative of the sigmoid function.
            
            This is required during backpropagation to calculate the gradients.

            Args :
                - a (numpy array): Output of the sigmoid function.

            Returns:
                - Derivative of the sigmoid function (numpy array).
        """
        return a * (1 - a)

    def forward(self, X):
        """
            Implements forward propagation through the neural network.
            
            Forward propagation computes the activations at each layer.

            Args :
                - X (numpy array): Input data of shape (number of samples, number of features).

            Returns:
                - A1 (numpy array): Activation output from the hidden layer.
                - A2 (numpy array): Activation output from the output layer (predicted output).
        """
        # Compute linear combination for hidden layer (input -> hidden)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)  # Apply activation (sigmoid)

        # Compute linear combination for output layer (hidden -> output)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)  # Apply activation (sigmoid)

        return self.A1, self.A2

    def compute_loss(self, Y, A2):
        """
            Computes the mean squared error (MSE) loss between predicted and actual labels.
            
            MSE loss is commonly used for regression tasks but can also work for binary classification.

            Args:
                - Y (numpy array): True labels (shape: number of samples, number of outputs).
                - A2 (numpy array): Predicted labels from the output layer (shape matches Y).

            Returns:
                - loss (float): Mean squared error between predicted and actual labels.
        """
        m = Y.shape[0]  # Number of samples
        loss = np.mean((Y - A2) ** 2)
        return loss

    def backward(self, X, Y, A1, A2):
        """
            Implements backpropagation to compute the gradients of weights and biases.
            
            Gradients are computed using the chain rule of calculus, moving backwards 
            through the network.

            Args:
                - X (numpy array): Input data (shape: number of samples, number of features).
                - Y (numpy array): True labels (shape: number of samples, number of outputs).
                - A1 (numpy array): Activation output from the hidden layer.
                - A2 (numpy array): Predicted labels from the output layer.

            Returns:
                - Gradients for weights and biases (dW1, db1, dW2, db2).
        """
        m = X.shape[0]  # Number of training examples

        # Output layer error (A2 - Y is the difference between prediction and truth)
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer error, using the derivative of the sigmoid function
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(A1)  # Element-wise multiplication
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        """
            Updates weights and biases using gradient descent.
            
            Parameters are updated by subtracting the gradient multiplied by the learning rate.

            Args:
            - dW1, db1, dW2, db2 (numpy arrays): Gradients for weights and biases.
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=10000):
        """
            Trains the neural network using the provided dataset.
            
            Training involves performing forward propagation, computing loss, 
            performing backpropagation, and updating parameters for each epoch.

            Args:
                - X (numpy array): Input data (shape: number of samples, number of features).
                - Y (numpy array): True labels (shape: number of samples, number of outputs).
                - epochs (int): Number of times to loop over the dataset.
        """
        for epoch in range(epochs):
            # Forward propagation
            A1, A2 = self.forward(X)

            # Compute loss
            loss = self.compute_loss(Y, A2)

            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, Y, A1, A2)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)

            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        """
            Makes predictions on new data by performing forward propagation.
            
            The output is thresholded at 0.5, returning 0 or 1 for classification.

            Args:
                - X (numpy array): New input data (shape: number of samples, number of features).

            Returns:
                - predictions (numpy array): Binary predictions (0 or 1) based on the output layer.
        """
        _, A2 = self.forward(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions

