import random
import time
from functools import reduce
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import math


random.seed(time.time())


def print_encoded_individual(individual):
    arr = np.array(individual)
    encoded_parameters = np.split(arr, 4)
    A = np.packbits(np.insert(encoded_parameters[0], 0, [0,0], axis=0))
    B = np.packbits(np.insert(encoded_parameters[1], 0, [0,0], axis=0))
    C = np.packbits(np.insert(encoded_parameters[2], 0, [0,0], axis=0))
    D = np.packbits(np.insert(encoded_parameters[3], 0, [0,0], axis=0))
    print("A:", A, "B:", B, "C:", C, "D:", D)


def individual(length, min, max):
    return [random.randint(min, max) for x in range(length)]


def population(count, length, min, max):
    return [individual(length, min, max) for x in range(count)]


def fitness(individual, target):
    arr = np.array(individual)
    #parse int array into the 4 parameters
    encoded_parameters = np.split(arr, 4)
    A = int(np.packbits(np.insert(encoded_parameters[0], 0, [0,0], axis=0))[0])
    B = int(np.packbits(np.insert(encoded_parameters[1], 0, [0,0], axis=0))[0])
    C = int(np.packbits(np.insert(encoded_parameters[2], 0, [0,0], axis=0))[0])
    D = int(np.packbits(np.insert(encoded_parameters[3], 0, [0,0], axis=0))[0])
    result = (A - B)**2 + (C + D)**2 - (A - 30)**3 - (C - 40)**3
    return abs(target - result)

def grade(population, target):
    summed = reduce(add, (fitness(x, target) for x in population), 0)
    return summed / len(population)


def evolve(population, target, min_value, max_value, retain=0.2,
           random_select=0.05, mutate=0.01):
    graded = [(fitness(x, target), x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
    # crossover parents to create offspring
    desired_length = len(population) - len(parents)
    children = []
    while len(children) < desired_length:
        male = random.randint(0, len(parents) - 1)
        female = random.randint(0, len(parents) - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            #pick attributes from male and female randomly
            child = [male[i] if random.randint(0,1) else female[i] for i in range(len(male))]
            children.append(child)
    # mutate some individuals
    for individual in children:
        if mutate > random.random():
            for i in range(len(individual)):
                #allow for multiple mutations per individual
                #increase this percentage by a bit (by using sqrt), otherwise the percentage of mutation would be way too low
                if math.sqrt(mutate) > random.random():
                    individual[i] = random.randint(
                        min_value, max_value)
    parents.extend(children)
    return parents


#this problem is not a good fit for an evolutionary algorithm as it has a smooth gradient
#and the equation is so simple i could do it off the top of my head

target = 100000
p_count = 100
i_length = 24   # 4*6 bits, these bits will be splitted and transformed to integers for evaluation
i_min = 0       # binary min and max
i_max = 1
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target), ]
for step in range(15):  # we stop after 15 generations
    p = evolve(p, target, i_min, i_max, retain=0.2, mutate=0.01)
    score = grade(p, target)
    fitness_history.append(score)
    print("step", str(step) + " score:", score)
#print entire population
for i in p:
    print_encoded_individual(i)

#using the code below, the best parameters will always be found
#the best parameters are A:0, B:63, C:0, D:63
#bestA = None
#bestB = None
#bestC = None
#bestD = None
#bestResult = 0
#
#for A in range(64):
#    for B in range(64):
#        for C in range(64):
#            for D in range(64):
#                result = (A - B)**2 + (C + D)**2 - \
#                    (A - 30)**3 - (C - 40)**3
#                fitness_history.append(result)
#                if result > bestResult:
#                    bestResult = result
#                    bestA = A
#                    bestB = B
#                    bestC = C
#                    bestD = D
#print("bestA:", bestA, "bestB:", bestB, "bestC:", bestC, "bestD:", bestD, "result:", bestResult)

plt.plot(fitness_history)
plt.xlabel('iteration')
plt.ylabel('fault')
plt.show()