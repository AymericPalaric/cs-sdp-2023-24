import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm 
from sklearn.cluster import KMeans
from data import Dataloader
import os

X = np.load("../data/dataset_10/X.npy")
Y = np.load("../data/dataset_10/Y.npy")

Z = (X >= Y).astype(int)

print (Z.shape)

kmeans = KMeans(n_clusters=3)
kmeans.fit(Z)

Z_means = kmeans.labels_
#trier par cluster les numéro de ligne
k0=np.where(kmeans.labels_ == 0)[0]
k1=np.where(kmeans.labels_ == 1)[0]
k2=np.where(kmeans.labels_ == 2)[0]

Z = np.load("../data/dataset_10/Z.npy")
max = 0

#faire une rotation de la numérotaion des cluster pour avoir le meilleur score
for i in range(2):
    for i in range(len(Z_means)):
        Z_means[i] = (Z_means[i]+1)%3
    score = sum(np.array_equal(Z[i], Z_means[i]) for i in range(len(Z))) / len(Z)

    if score > max:
        max = score
        print(max)
