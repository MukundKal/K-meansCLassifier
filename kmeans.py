
# coding: utf-8

# # K-Means Clustering

# Let's make some fake data that includes people clustered by income and age, randomly:

import numpy as np
from numpy import random, array


#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    
    return X


# We'll use k-means to rediscover these clusters using unsupervised learning:

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float

data = createClusteredData(100, 5)

model = KMeans(n_clusters=4)   #number of clusters you want kmeans to initialise and create

# Note I'm scaling the data to normalize it, required to avoid over-crediting certain parameters
model = model.fit(scale(data))

# We can look at the clusters each data point was assigned to
model.labels_.reshape(100,1) #cluster assignment of each of the hundred people

# And we'll visualize it:

plt.figure(figsize=(10, 10))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))  #color will vary for each cluster
plt.show()




