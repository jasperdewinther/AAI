import numpy as np
import math
import random
import matplotlib.pyplot as plt

#get indexes of the lowest values in a np array
def lowest_indexes(array, k):
    lowest = np.full((k),0, float)
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

def total_len_to_centroids(array, clusters, meanClusters):
    total_len = 0
    for i in range(len(meanClusters)):
        for j in range(len(clusters)):
            if clusters[j] == i:
                total_len += distance(meanClusters[i], array[j])
    return total_len

def get_unique_ints(amount_of_ints, maximum_exclusive):
    centroids = np.full([amount_of_ints], 0)
    while len(np.unique(centroids)) != amount_of_ints:
        for i in range(len(centroids)):
            centroids[i] = random.randint(0,maximum_exclusive)
    return centroids

def get_clusters(array, centroids):
    clusters = np.full([len(array)], -1)
    for i in range(len(array)):
        highest_value = math.inf
        highest_iterator = 0
        for j in range(len(centroids)):
            #print(i,j,array[i], centroids[j])
            distance_to_centroid = distance(array[i], centroids[j])
            if distance_to_centroid < highest_value:
                highest_value = distance_to_centroid
                highest_iterator = j
        clusters[i] = highest_iterator
    return clusters

def count_elements_in_np_array(array, element):
    counter = 0
    for i in array:
        if i == element:
            counter+=1
    return counter


def get_new_centroids(array, clusters, k):
    means = np.full([k, len(array[0])], 0, float)
    for i in range(len(array)):
        for j in range(len(array[i])):
            means[clusters[i]][j] += array[i][j]
    for i in range(len(means)):
        for j in range(len(means[0])):
            counted = count_elements_in_np_array(clusters, i)
            if counted == 0:
                continue
            means[i][j] /= counted
    return means


def get_centroids(k, array):
    centroid_indexes = get_unique_ints(k, len(array)-1)
    centroids = np.full([k, len(array[0])], 0, float)
    for i in range(k):
        for j in range(len(array[0])):
            centroids[i][j] = array[centroid_indexes[i]][j]
    return centroids

def plot_truthValues_and_centroids(truthValues, centroids, clusters):
    x = np.hsplit(truthValues,2)[0]
    y = np.hsplit(truthValues,2)[1]
    centroidx = np.hsplit(centroids,2)[0]
    centroidy = np.hsplit(centroids,2)[1]
    colormap = np.array(['b', 'g', 'y', 'c', 'm', 'y', 'r', 'g', 'b', 'c', 'm', 'y',
                    'r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'k', 'k', 'k', 'k', 'k'])
    plt.scatter(x,y, s=10, c=colormap[clusters])
    plt.scatter(centroidx,centroidy, s=50)
    plt.ylabel('')
    plt.show()

#run k-means and return the season that is most likely
def k_means(k, truthValues, labels = None, plot = False):

    clustersDefinitive = None
    total_distanceDefinitive = math.inf
    centroidsDefinitive = None
    #randomize starting points 10 times
    print("k-means with k:", k)
    for _ in range(10):
        centroids = get_centroids(k, truthValues)
        clusters = None
        total_distance = math.inf
        #iterate 10 times
        for i in range(10):
            #get every clusternumber that belongs to the truthvalues
            clusters = get_clusters(truthValues, centroids)
            #count the total distance
            total_distance = total_len_to_centroids(truthValues, clusters, centroids)
            #move the centroids towards the mean of the cluster
            centroids = get_new_centroids(truthValues, clusters,k)
        if total_distance < total_distanceDefinitive:
            #if the distance is shorter than an existing cluster, make this the new definitive
            total_distanceDefinitive = total_distance
            clustersDefinitive = clusters.copy()
            centroidsDefinitive = centroids.copy()
        print("local optimum:", total_distance)
    if plot:
        plot_truthValues_and_centroids(truthValues, centroidsDefinitive, clustersDefinitive)
        

    if labels != None:
        #calulate what the most used labels are in every cluster
        clustersWithLabels = dict()
        for i in range(max(clustersDefinitive)+1):
            clustersWithLabels[i] = []
        for i in range(len(clustersDefinitive)):
            clustersWithLabels[clustersDefinitive[i]].append(labels[i])
        clusterLabels = list()
        for i in range(len(clustersWithLabels)):
            clusterLabels.append(most_frequent(clustersWithLabels[i]))
        return total_distanceDefinitive, clusterLabels
    else:
        return total_distanceDefinitive
    

#get distance between 2 np arrays
def distance(entry1, entry2):
    distance = 0
    for i in range(entry1.shape[0]):
        distance+=(entry1[i]-entry2[i])**2
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
    minMax = np.full((entriesAmount, 2), math.inf, float)
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







dates = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[0])
labels = []
for label in dates:
  if label < 20000301:
    labels.append('winter')
  elif 20000301 <= label < 20000601:
    labels.append('lente')
  elif 20000601 <= label < 20000901:
    labels.append('zomer')
  elif 20000901 <= label < 20001201:
    labels.append('herfst')
  else: # from 01-12 to end of year
    labels.append('winter')


#visualise the clustering
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
minMax = maxMinValues(data)
scaledArray = scaleData(data, minMax)
k_means(4, scaledArray, plot=True)

#visualise distance compared to k
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
minMax = maxMinValues(data)
scaledArray = scaleData(data, minMax)
results = np.full([15], 0, float)
for i in range(0,15):
    #count unique seasons
    results[i] = k_means(i+1, scaledArray)
plt.plot(results)
plt.xlabel('k')
plt.ylabel('total distance')
plt.show()

#visualise how many different seasons are found with every k value
data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
minMax = maxMinValues(data)
scaledArray = scaleData(data, minMax)
results = np.full([15], 0, float)
for i in range(0,15):
    _, seasons = k_means(i+1, scaledArray, labels=labels)
    #count unique seasons
    results[i] = len(set(seasons))
plt.plot(results)
plt.xlabel('k')
plt.ylabel('total unique seasons')
plt.show()

