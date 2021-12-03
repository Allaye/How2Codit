import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt


# pytorch pipline
# Design model, design input, output layer
# Construct loss function and optimizer
# Train the model
   # 3.1 forward pass (forward propagation): compute the model prediction
   # 3.2 backward pass (backward propagation): compute the gradient of the loss function with respect to the model parameters
   # 3.3 update the model parameters
# create the dataset and prediction visualization function
# provide the model with the input and compute the prediction



# create dataset from sklearn and split the data into train and test
X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=10)

# convert the dataset to torch tensor
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
# reshare Y dataset
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

def forward(X: torch.Tensor):
    output_size, input_size = X.shape
    return nn.Linear(input_size, 1)

def loss(Y_pred, Y):
    l = nn.MSELoss()
    return l(Y_pred, Y)


def optimization(w, lr):
    return torch.optim.SGD(w, lr=lr)

def traning(epoch, lr: float, X: torch.Tensor, Y: torch.Tensor):
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
    ploter(model, X, Y)
    

def ploter(model, X, Y):
    Y_pred = model(X).detach().numpy()
    plt.scatter(X, Y, label="Original data")
    plt.scatter(X, Y_pred, label="Predicted data")
    plt.show()

if __name__ == "__main__":
    traning(epoch=100, lr=0.01, X=X, Y=Y)
    