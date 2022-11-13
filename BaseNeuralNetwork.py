import numpy as np

#layer superclass that will be le used later
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input): #forward propagation: input -> output (input of the next layer) -> output of the next layer -> output of the next layer
        pass

    def backward(self, output_error, learning_rate): #back propagation: output_error -> input_error -> input_error -> input_error
        # GRADIENT DESCENT:
        # finds the minimum of a function by taking the derivative of the function and moving in the direction of the negative gradient to minimize a cost or loss function
        # obviously this means only differentiable (and convex) functions can be used

        # W = W - learning_rate * dE/dW

        # CALCULATIONS FOR THE BACKWARD FUNCTION
        # output_error = (output - target) * derivative(output)
        # input_error = output_error * weights.T
        pass

#dense (fully connected) layer! not a CNN yet but we will get there
class Dense(Layer):
    def __init__(self, input_size, output_size): #input_size is the number of neurons in the previous layer, output_size is the number of neurons in the current layer
        super().__init__() #call the superclass constructor
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size) #initialize the weights with a random normal distribution
        self.biases = np.zeros(output_size) #initialize the biases to 0

    def forward(self, input): #input is the output of the previous layer
        self.input = input #save the input for backpropagation
        self.output = np.dot(input, self.weights) + self.biases #calculate the output of the current layer using the formula Y = XW + B

    def backward(self, output_error, learning_rate): #will be used later

