import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm 
from sklearn.cluster import KMeans

# Load X et Y dataset
X=np.loadtxt('../data/dataset_10/X.npy')
Y=np.loadtxt('../data/dataset_10/Y.npy')

# Create KMeans model
model = KMeans(n_clusters=3)
model.fit(X)







