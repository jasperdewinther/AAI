import numpy as np
import math
import matplotlib.pyplot as plt


#get indexes of the lowest values in a np array
def lowest_indexes(array, k):
    lowest = np.full((k),0)
    minimum = 0
    index = 0
    for i in range(k):
        newMin = 8569248
        for j in range(len(array)):
            if array[j] > minimum and array[j] < newMin:
                newMin = array[j]
                index = j
        minimum = newMin
        lowest[i] = index
    return lowest

#get most frequent item in list
def most_frequent(List): 
    return max(set(List), key = List.count) 

#run nearest neighbor and return the season that is most likely
def k_nearestneighbor(k, thruthValues, entry, labels):
    distances = get_distances(thruthValues, entry)
    indexes = lowest_indexes(distances, k)
    seasons_found = list()
    for i in indexes:
        seasons_found.append(labels[i])
    return most_frequent(seasons_found)

#get distance between 2 np arrays
def distance(entry1, entry2):
    distance = 0
    for i in range(entry1.shape[0]):
        distance+=abs(entry1[i]-entry2[i])**2
    return distance

#get array with distances to all points of np array from a given entry
def get_distances(npArray, entry):
    distances = np.full((len(npArray)), 0, float)
    for i in range(len(npArray)):
        distances[i] = distance(entry, npArray[i])
    return distances

#get the highest and lowest values of every category
def maxMinValues(npArray):
    entriesAmount = len(npArray[0])
    minMax = np.full((entriesAmount, 2), math.inf)
    for i in range(len(data[0])):
        if minMax[i][0] == math.inf:
            minMax[i][0] = data[0][i]
        if minMax[i][1] == math.inf:
            minMax[i][1] = data[0][i]
    for label in data:
        for i in range(len(label)):

            if minMax[i][0] > label[i]:
                minMax[i][0] = label[i]
            if minMax[i][1] < label[i]:
                minMax[i][1] = label[i]
    return minMax

#scale data to a maximum of 1 and minimum of 0
def scaleData(npArray, minMax):
    multiplier = np.full((len(minMax)),0, float)
    addition = np.full((len(minMax)),0)
    for i in range(len(minMax)):
        addition[i] = -minMax[i][0]
        multiplier[i] = 1/(minMax[i][1] + addition[i])
    for i in npArray:
        for j in range(len(i)):
            i[j] = (i[j]+addition[j])*multiplier[j]
    return npArray


#load dataset labels
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
labels = []
for label in data:
    if label < 20000301:
        labels.append('winter')
    elif 20000301 <= label < 20000601:
        labels.append('lente')
    elif 20000601 <= label < 20000901:
        labels.append('zomer')
    elif 20000901 <= label < 20001201:
        labels.append('herfst')
    else:
        labels.append('winter')

#load dataset
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
#scale dataset
minMax = maxMinValues(data)
scaledArray = scaleData(data, minMax)

#load validation labels
validationData = np.genfromtxt('validation1.csv', delimiter=';', usecols=[0], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validationLabels = []
for label in validationData:
    if label < 20010301:
        validationLabels.append('winter')
    elif 20010301 <= label < 20010601:
        validationLabels.append('lente')
    elif 20010601 <= label < 20010901:
        validationLabels.append('zomer')
    elif 20010901 <= label < 20011201:
        validationLabels.append('herfst')
    else:
        validationLabels.append('winter')
#load validation dataset and scale using training dataset scaling
validationData = np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
scaledValidation = scaleData(validationData, minMax)

#get days
days = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
for j in days:
    #print the most probable seasons from days
    print(k_nearestneighbor(59, data, j, labels))




#data about the best k
highest_k = 0
highest_value = 0
#array used to plot data later
array = np.full((366), 0, float)
#check for every value up until a k of 366
for i in range(1, 366):
    good = 0
    for j in range(len(scaledValidation)):
        #do k-nearestneighbor for all validationdata and check if correct with labels 
        if k_nearestneighbor(i, data, scaledValidation[j], labels) == validationLabels[j]:
            good+=1
    #store data
    array[i] = good/len(scaledValidation)*100
    if good/len(scaledValidation)*100 > highest_value:
        highest_value = good/len(scaledValidation)*100
        highest_k = i
    #print result
    print(str(i) + " percentage correct: " +  str(good/len(scaledValidation)*100))

#plot graph of accuracy
plt.plot(array)
plt.ylabel('some numbers')
plt.show()

#print best result
print("best k: " + str(highest_k) + " and the value was: " + str(highest_value) + "%")