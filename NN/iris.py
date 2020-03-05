import math
import numpy as np
import time
import random
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
    return (x>0)*1

#wrapper to quickly swap activation functions
def activation_function(z):
    return sigmoid(z)
def activation_function_derivative(x):
    return sigmoid_derivative(x)

class Neuron:
    def __init__(self, weigths, rate_of_change=0.1, bias=random.uniform(-1,1), batch_size=1, output_node = True, node_number=0):
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
        self.output_node = output_node
        self.node_number = node_number
        self.delta = None

    #used for debugging
    def __str__(self):
        return "rate of change: " + str(self.rate_of_change) + " bias: " + str(self.bias) + " weigths:" + str(self.weigths) + " node number: " + str(self.node_number) + " output_node: " + str(self.output_node) + " current delta: " + str(self.delta)
    def __repr__(self):
        return str(self) + "\n"
    
    def infer(self, inputs):
        if len(self.weigths) != len(inputs):
            print(len(self.weigths), len(inputs))
            raise Exception("not as many inputs as weigths")
        #store values for use in update function
        self.inputs = inputs
        self.summed_input = np.dot(inputs, self.weigths)+self.bias
        self.output = activation_function(self.summed_input)
        return self.output

    def update(self, desired = None, nodes_to_the_right = None):
        self.delta = None
        fault = None
        if self.output_node == True:
            #calculate delta for output node
            fault = self.output-desired
            self.delta = fault * activation_function_derivative(self.summed_input)
        else:
            #calculate from nodes to the right and their weigths
            weigthedSum = 0
            for i in range(len(nodes_to_the_right)):
                weigthedSum += nodes_to_the_right[i].weigths[self.node_number] * nodes_to_the_right[i].delta
            self.delta = activation_function_derivative(self.summed_input)*weigthedSum
            fault = weigthedSum
        if self.batch_size <= 1:
            #if bias is 1 always apply change immediately
            for i in range(len(self.weigths)):
                self.weigths[i] = self.weigths[i] - self.rate_of_change * self.delta * self.inputs[i]
            self.bias = self.bias - self.rate_of_change*self.delta
            return fault
        else:
            #if bias is not 1 accumulate the changes
            self.batch_counter+=1
            self.batch_bias += self.rate_of_change*self.delta
            for i in range(len(self.weigths)):
                self.batch_weigths[i] += self.rate_of_change * self.delta * self.inputs[i]
        if self.batch_counter == self.batch_size:
            #apply accumulated changes
            self.batch_counter = 0
            self.weigths = self.weigths - self.batch_weigths/self.batch_size
            self.bias = self.bias - self.batch_bias/self.batch_size
            #reset back to start
            self.batch_weigths = np.full(self.weigths.shape, 0, float)
            self.batch_bias = 0
        return fault

class Network:
    def __init__(self, setup=np.array([1]), learning_rate=1, batch_size=1):
        self.neurons = list()
        for _ in range(len(setup)-1):
            self.neurons.append(list())
        for i in range(len(setup)):
            for j in range(setup[i]):
                #create nodes and layers
                if i==len(setup)-1:
                    #make sure the last layer is made out of output nodes
                    self.neurons[i-1].append(self.__createNeuron(setup[i-1], learning_rate, batch_size, j, True))
                elif i == 0:
                    continue
                else:
                    self.neurons[i-1].append(self.__createNeuron(setup[i-1], learning_rate, batch_size, j))

    def __createNeuron(self, inputs, learning_rate, batch_size, neuron_number=0, output_node=False):
        #initialise the neuron with random weigths and a learning rate of 2
        random.seed(time.time())
        default_weights = np.full((inputs), 0, float)
        for i in range(inputs):
            default_weights[i] = random.uniform(-1,1)
        return Neuron(default_weights, learning_rate, batch_size=batch_size, node_number=neuron_number, output_node=output_node)

    #used for debugging
    def __str__(self):
        s = ""
        for i in range(len(self.neurons)):
            if i == len(self.neurons)-1:
                s+= "output layer:\n"
            else:
                s+="hidden layer " + str(i) + ":\n"
            for j in range(len(self.neurons[i])):
                s+=str(self.neurons[i][j]) + "\n"
            s+="\n"
        return s
            

    def infer(self, inputs):
        layer_input = inputs.copy()
        #loop over all layers
        for layer_iterator in range(len(self.neurons)):
            layer_result = np.full((len(self.neurons[layer_iterator])), 0, float)
            #loop over all neurons in layer
            for neuron_iterator in range(len(self.neurons[layer_iterator])):
                #do infer and store outputs as inputs for the next layer
                layer_result[neuron_iterator] = self.neurons[layer_iterator][neuron_iterator].infer(layer_input)
            layer_input = layer_result.copy()
        return layer_input
    
    def update(self, desired):
        summedFault = 0
        for layer_iterator in reversed(range(len(self.neurons))):
            for neuron_iterator in range(len(self.neurons[layer_iterator])):
                #update every neuron from right to left
                if layer_iterator == len(self.neurons)-1:
                    #the output layer uses the desired output to calculate delta
                    summedFault += abs(self.neurons[layer_iterator][neuron_iterator].update(desired=desired[neuron_iterator]))
                else:
                    #the other layers use the layer to their right
                    self.neurons[layer_iterator][neuron_iterator].update(nodes_to_the_right=self.neurons[layer_iterator+1])
        return summedFault


def getIrisInAndOutputs():
    numberOfOutputs = 3
    #inputs and outputs
    linecount = 0
    inputCount = 0
    labels = []
    f = open("iris.data", "r")
    for x in f:
        if x.split(',')[-1].rstrip() not in labels:
            labels.append(x.split(',')[-1].rstrip())
        inputCount = len(x.split(','))-1
        linecount+=1
    f.close()

    inputs = np.full((linecount, inputCount), 0, float)
    outputs = np.full((linecount, numberOfOutputs), 0, float)
    f = open("iris.data", "r")
    line = 0
    for x in f:
        for value in range(len(x.split(','))-1):
            inputs[line][value] = x.split(',')[value]
        outputs[line][labels.index(x.split(',')[-1].rstrip())] = 1
        line+=1
    return inputs, outputs, labels

def createTestSet(inputs, outputs, partValidation):
    #split inputs and outputs into train and validate set
    #partValidation must be a number between 0 and 1
    length = len(inputs)
    modulus = int((1/(length*partValidation))*length)

    trainInputs = list()
    validationInputs = list()
    trainOutputs = list()
    validationOutputs = list()

    for i in range(len(inputs)):
        if i%modulus == 0:
            validationInputs.append(inputs[i])
            validationOutputs.append(outputs[i])
        else:
            trainInputs.append(inputs[i])
            trainOutputs.append(outputs[i])
    return trainInputs, validationInputs, trainOutputs, validationOutputs

def compareResults(array1, array2):
    #check if the highest value of every result has the same index of the highest result of the other array
    if len(array1) != len(array2):
        raise Exception("arrays are not the same size")
    good = 0
    for i in range(len(array1)):
        # Find index of maximum value from 2D numpy array
        maxIndex1 = np.where(array1[i] == np.amax(array1[i]))
        maxIndex2 = np.where(array2[i] == np.amax(array2[i]))
        if maxIndex1 == maxIndex2:
            good+=1
    return good/len(array1)


inputs, outputs, labels = getIrisInAndOutputs()
trainIn, validateIn, trainOut, validateOut = createTestSet(inputs, outputs, 0.5)

#default fault has to be more than the threshold in while loop
fault = 100
counter = 0
batch_size = len(trainIn)
learning_rate = 0.1
#4 inputs 2 layers of 4 hidden nodes and 3 outputs
network_topology = [4,4,4,3]

#network with layout, learning rate and batch size
network = Network(network_topology, learning_rate, batch_size)

print("STARTING NETWORK:")
print(network)

#fualts is used to graph the learning rate
faults = list()
#keep improving until control-c has been pressed or a lot of training has been done
try:
    for j in range(10000):
        fault = 0
        #loop over possible inputs
        for i in range(len(trainIn)):
            desired = trainOut[i]
            network.infer(trainIn[i])
            fault += network.update(desired)
            counter+=1
        #sometimes print progress
        if counter%100==0:    
            faults.append(fault)
            print("cycle:", str(j), "fault:", fault)
except:
    pass



#print results
print("END NETWORK:")
print(network)
print("took", counter, "updates and", counter/batch_size, "batches")

#list to compare with ground truth
answers = list()
for i in range(len(validateIn)):
    desired = validateOut[i]
    realResult = network.infer(validateIn[i])
    print("input:", validateIn[i], "result:" , realResult, "desired:", desired)
    answers.append(realResult)

precision = compareResults(answers, validateOut)
print("correctly guessed from validation set:", precision*100, "%")

#plot learning graph
plt.plot(faults)
plt.xlabel('iteration')
plt.ylabel('fault')
plt.show()