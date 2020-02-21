import math
import numpy as np
import time
import random

def sigmoid(z):
    if z < -300:
        return 0
    if z > 300:
        return 1
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class Perceptron:
    def __init__(self, weigths, threshold):
        self.weigths = weigths
        self.threshold = threshold
    def infer(self, inputs):
        if len(self.weigths) != len(inputs):
            raise Exception("not as many inputs as weigths")
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weigths[i]
        return int(sigmoid(sum) >= self.threshold)

p = Perceptron((-1,-1), 0.5)



class Neuron:
    def __init__(self, weigths, rate_of_change=0.1, bias=random.uniform(-1,1)):
        self.inputs = None
        self.summed_input = None
        self.output = None
        self.weigths = weigths
        self.rate_of_change = rate_of_change
        self.bias = bias
    def __str__(self):
        return "rate of change: " + str(self.rate_of_change) + " bias: " + str(self.bias) + " weigths:" + str(self.weigths)
    def infer(self, inputs):
        if len(self.weigths) != len(inputs):
            raise Exception("not as many inputs as weigths")
        self.inputs = inputs
        self.summed_input = np.dot(inputs, self.weigths)+self.bias
        #print(inputs, self.weigths)
        self.output = sigmoid(self.summed_input)
        return self.output
    def update(self, desired):
        delta = (self.output-desired) * sigmoid_derivative(self.output)
        #print("error:", (self.output-desired)**2, "out:", self.output, "desired:", desired, "sig:", sigmoid_derivative(self.output), "bias:", self.bias)
        for i in range(len(self.weigths)):
            #print(self.rate_of_change, delta, self.inputs[i])
            self.weigths[i] = self.weigths[i] - self.rate_of_change * delta * self.inputs[i]
        self.bias = self.bias - self.rate_of_change*delta
        

random.seed(time.time())
default_weights = np.full((2), 0, float)
default_weights[0] = random.uniform(-1,1)
default_weights[1] = random.uniform(-1,1)
n = Neuron(default_weights, 2)
print(n)

for _ in range(10000):
    for i in range(2):
        for j in range(2):
            desired = p.infer((i,j))
            #print("input:", [i,j], "desired:", desired, "result:" , n.infer((i,j)))
            n.infer((i,j))
            n.update(desired)
            #print(n)
print(n)
for i in range(2):
    for j in range(2):
        desired = p.infer((i,j))
        print("input:", [i,j], "result:" , n.infer((i,j)), "desired:", desired)