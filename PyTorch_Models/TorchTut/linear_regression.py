import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# create dataset from sklearn and split the data into train and test
X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=10)

# convert the dataset to torch tensor
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
# reshare Y dataset
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

def forward(X):
    input_size, output_size = X.shape
    return nn.Linear(input_size, output_size)

def loss(Y_pred, Y):
    l = nn.MSELoss()
    return l(Y_pred, Y)


def optimization(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr)

def traning(epoch, lr, X, Y):
    print("Training the model")
    for epoch in range(epoch):
        # forward pass
        model = forward(X)
        Y_pred = model(X)
        # calculate loss
        los = loss(Y_pred, Y)
        # perform tbe backward pass, calculate the gradient
        los.backward()
        # update the parameters and empty the gradients
        optimizer = optimization(model.parameters(), lr)
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            w, b = model.parameters()
            print("epoch: {}, loss: {:.8f}, w: {}".format(epoch, los, w[0][0].item()))

def ploter(model):
    


if __name__ == "__main__":
    pass