import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# create data set from sklearn
X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=10)

# convert the dataset to torch tensor
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
