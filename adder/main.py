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
        return int(sigmoid(sum) > self.threshold)

class Adder:
    def __init__(self):
        #p1 and p2 are used for the xor
        self.p1 = Perceptron((1,1), 0.6)
        self.p2 = Perceptron((-1,-1), 0.2)
        #p3 is an and gate
        self.p3 = Perceptron((1,1), 0.8)
    def infer(self, in1, in2, in3):
        result_first_xor = self.p3.infer((self.p1.infer((in1,in2)), self.p2.infer((in1,in2))))
        result_second_xor = self.p3.infer((self.p1.infer((result_first_xor,in3)), self.p2.infer((result_first_xor,in3))))
        result_first_and = self.p3.infer((in1, in2))
        result_second_and = self.p3.infer((in3, result_first_xor))
        result_first_or = self.p1.infer((result_first_and, result_second_and))
        return result_first_or, result_second_xor

adder = Adder()

for i in range(2):
    for j in range(2):
        for k in range(2):
            print("input:", [i,j,k], "result:" , adder.infer(i,j,k))