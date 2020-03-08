import numpy as np
import math
import time
import matplotlib.pyplot as plt

#sigmoid and derivative
def sigmoid(z):
    #math.exp could throw out of range error if z is too big or too small
    if z < -700:
        return 0
    if z > 700:
        return 1
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

#tanh and derivative
def tanh(z):
    if z < -700:
        return -1
    if z > 700:
        return 1
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def tanh_derivative(x):
    return 1-tanh(x)**2

#relu and derivative
def relu(z):
    return max(0,z)
def relu_derivative(x):
    return float(x>0)

#wrapper to quickly swap activation functions
def activation_function(z):
    return sigmoid(z)
def activation_function_derivative(x):
    return sigmoid_derivative(x)


vectorised_activation = np.vectorize(activation_function)

class Network:
    def __init__(self, learning_rate: float, topology: np.ndarray, activation_function: callable, activation_function_derivative: callable, batch_size: int = 0):
        self.vectorised_activation = np.vectorize(activation_function)
        self.vectorised_derivative = np.vectorize(activation_function_derivative)
        
        np.random.seed(int(time.time()))

        self.intermidiate_sums = list()
        for i in range(1, len(topology)):
            self.intermidiate_sums.append(None)

        #create all weigth matrices
        self.matrices = list()
        for layer in range(len(topology)-1):
            #self.matrices.append(np.random.rand((topology[layer]+1, topology[layer+1], batch_size)))
            self.matrices.append(np.random.rand(topology[layer+1], topology[layer]+1))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for i in range(len(self.matrices)):
            weigthed_inputs = np.dot(self.matrices[i], np.append(1, inputs))
            self.intermidiate_sums[i] = weigthed_inputs
            inputs = self.vectorised_activation(weigthed_inputs)
        return inputs

    def backprop(self, expected_outputs: np.ndarray) -> None:
        deltas = list()
        fault = np.subtract(self.intermidiate_sums[len(self.intermidiate_sums)-1], expected_outputs)
        print(self.intermidiate_sums)
        #print(self.vectorised_derivative(self.intermidiate_sums[len(self.intermidiate_sums)-1]))
        print(fault)
        deltas.append(np.multiply(self.vectorised_derivative(self.intermidiate_sums[len(self.intermidiate_sums)-1]), fault))

        for i in range(len(self.matrices)-1):
            print("yeet")

        print(deltas)
        print(self.matrices)



    def infer(self, inputs: np.ndarray) -> np.ndarray:
        for i in range(len(self.matrices)):
            inputs = self.vectorised_activation(np.dot(self.matrices[i], np.append(1, inputs)))
        return inputs

n = Network(1, np.array([4,2,2]), sigmoid, sigmoid_derivative)
result = n.forward(np.array([[0.5,0.5,0.5,1]], float))
print(result)
print("noooow the back propogationnnnn")
n.backprop(np.array((2,2)))