import math


def sigmoid(z):
  return 1 / (1 + math.exp(-z))

class Perceptron:
    def __init__(self, weigths, threshold):
        self.weigths = weigths
        self.threshold = threshold
    def infer(self, inputs):
        if len(self.weigths) != len(inputs):
            return -1 
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weigths[i]
        return int(sigmoid(sum) >= self.threshold)

p = Perceptron((-1,-1,-1), 0.5)


for i in range(2):
    for j in range(2):
        for k in range(2):
            print("input:", [i,j,k], "result:" , p.infer((i,j,k)))