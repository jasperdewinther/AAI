import numpy as np
import math
import time
import matplotlib.pyplot as plt

#sigmoid and derivative
def sigmoid(z):
    #math.exp could throw out of range error if z is too big or too small
    if z < -700:
        z = -700
    if z > 700:
        z = 700
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
        self.learning_rate = learning_rate

        np.random.seed(int(time.time()))

        self.intermidiate_sums = list()
        for _ in range(len(topology)):
            self.intermidiate_sums.append(None)

        #create all weigth weigths_matrices
        self.weigths_matrices = list()
        for layer in range(len(topology)-1):
            self.weigths_matrices.append(np.random.rand(topology[layer+1], topology[layer]+1))
        print(self.weigths_matrices)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.intermidiate_sums[0] = inputs.copy()
        for i in range(len(self.weigths_matrices)):
            weigthed_inputs = np.dot(self.weigths_matrices[i], np.insert(inputs, 0, 1, axis=0))
            self.intermidiate_sums[i+1] = weigthed_inputs
            inputs = self.vectorised_activation(weigthed_inputs)
        return inputs

    def backprop(self, expected_outputs: np.ndarray) -> None:
        deltas = list()
        #calculate delta of all output nodes first
        faults = np.subtract(expected_outputs, self.vectorised_activation(self.intermidiate_sums[-1]))
        G_derived = self.vectorised_derivative(self.intermidiate_sums[-1])
        deltas_output_nodes = np.multiply(G_derived, faults)
        deltas.append(deltas_output_nodes)

        #calculate delta of all other nodes
        for i in reversed(range(1,len(self.weigths_matrices))):
            #take weight matrix without bias weigths
            weigths_to_the_right = self.weigths_matrices[i][:,1:]
            summed_deltas_to_the_right = np.dot(weigths_to_the_right.T, deltas[0])
            G_derived = self.vectorised_derivative(self.intermidiate_sums[i])
            new_deltas = np.multiply(summed_deltas_to_the_right, G_derived)
            deltas.insert(0, new_deltas)

        #calculate weigth adjustments using deltamatrix
        for i in range(len(deltas)):
            a = None
            if i == 0:
                a = self.intermidiate_sums[i]
            else:
                a = self.vectorised_activation(self.intermidiate_sums[i])
            a = np.insert(a, 0, 1, axis=0)
            aD = np.outer(deltas[i], a)
            naD = np.multiply(self.learning_rate, aD)
            self.weigths_matrices[i] = np.add(self.weigths_matrices[i], naD)
        return np.sum(faults)


    def infer(self, inputs: np.ndarray) -> np.ndarray:
        for i in range(len(self.weigths_matrices)):
            inputs = self.vectorised_activation(np.dot(self.weigths_matrices[i], np.append(1, inputs)))
        return inputs

n = Network(10, np.array([2,4,1]), sigmoid, sigmoid_derivative)

inputsNOR = np.array([[0,0], [0,1], [1,0], [1,1]])
validationNOR = np.array([0,1,1,0])


faults = list()
for i in range(1000):
    loopFaults = 0
    for j in range(4):
        result = n.forward(inputsNOR[j])
        loopFaults +=abs(n.backprop(validationNOR[j]))
    faults.append(loopFaults)

for j in range(4):
    result = n.forward(inputsNOR[j])
    n.backprop(validationNOR[j])

for i in range(len(inputsNOR)):
    desired = validationNOR[i]
    print("input:", inputsNOR[i], "result:" , n.forward(inputsNOR[i]), "desired:", desired)


#plot learning graph
plt.plot(faults)
plt.xlabel('iteration')
plt.ylabel('fault')
plt.show()