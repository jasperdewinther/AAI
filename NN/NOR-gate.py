import math
import numpy as np
import time
import random

def sigmoid(z):
    #math.exp could throw out of range error if z is too big or too small
    if z < -300:
        return 0
    if z > 300:
        return 1
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


class Neuron:
    def __init__(self, weigths, rate_of_change=0.1, bias=random.uniform(-1,1), batch_size=1):
        self.inputs = None
        self.summed_input = None
        self.output = None
        self.weigths = weigths
        self.rate_of_change = rate_of_change
        self.bias = bias
        self.batch_size = batch_size
        self.batch_counter = 0
        self.batch_bias = 0
        self.batch_weigths = np.full(weigths.shape, 0, float)
    def __str__(self):
        return "rate of change: " + str(self.rate_of_change) + " bias: " + str(self.bias) + " weigths:" + str(self.weigths)
    def infer(self, inputs):
        if len(self.weigths) != len(inputs):
            raise Exception("not as many inputs as weigths")
        self.inputs = inputs
        self.summed_input = np.dot(inputs, self.weigths)+self.bias
        self.output = sigmoid(self.summed_input)
        return self.output
    def update(self, desired):
        fault = self.output-desired
        delta = (fault) * sigmoid_derivative(self.output)
        if self.batch_size <= 1:
            #if bias is 1 always apply change immediately
            for i in range(len(self.weigths)):
                self.weigths[i] = self.weigths[i] - self.rate_of_change * delta * self.inputs[i]
            self.bias = self.bias - self.rate_of_change*delta
            return fault
        else:
            #if bias is not 1 accumulate the changes
            self.batch_counter+=1
            self.batch_bias += self.rate_of_change*delta
            for i in range(len(self.weigths)):
                self.batch_weigths[i] += self.rate_of_change * delta * self.inputs[i]
        if self.batch_counter == self.batch_size and not self.batch_size <= 1:
            #apply accumulated changes
            self.batch_counter = 0
            self.weigths = self.weigths - self.batch_weigths/self.batch_size
            self.bias = self.bias - self.batch_bias/self.batch_size
        return fault

def createNeuron(learning_rate, batch_size):
    #initialise the neuron with random weigths and a learning rate of 2
    random.seed(time.time())
    default_weights = np.full((2), 0, float)
    default_weights[0] = random.uniform(-1,1)
    default_weights[1] = random.uniform(-1,1)
    return Neuron(default_weights, learning_rate, batch_size=batch_size)

inputsNOR = np.array([[0,0], [0,1], [1,0], [1,1]])
validationNOR = np.array([1,0,0,0])

fault = 100
counter = 0
#the larger the learning rate the faster it will hit the fault threshold, but there is also a higher chance that fault will be 0 because all results are 0
#a learning rate of 200 is relatively stable and makes the neural network take less than 50 cycles (24 batches) to hit the fault threshold
n = createNeuron(20,2)
while fault > 0.00000001:
    #loop over possible inputs
    for i in range(len(inputsNOR)):
        desired = validationNOR[i]
        n.infer(inputsNOR[i])
        fault = n.update(desired)
        #print(fault)
        counter+=1
print(n)
print("took", counter, "updates and", counter/1, "batches")
for i in range(len(inputsNOR)):
    desired = validationNOR[i]
    print("input:", inputsNOR[i], "result:" , n.infer(inputsNOR[i]), "desired:", desired)