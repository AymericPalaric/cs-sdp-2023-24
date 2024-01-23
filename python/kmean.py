import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm 
from sklearn.cluster import KMeans
from data import Dataloader
import os

X = np.loadtxt("../data/dataset_4/X.npy",delimiter=';',dtype='float',header=True)
Y = np.loadtxt("../data/dataset_4/Y.npy",delimiter=';',dtype='float',header=True)

print(X)






