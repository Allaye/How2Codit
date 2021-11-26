import math
import torch


def backpropagation(x, y, w):
    # calculate Y hat and compute the loss value
    y_hat = w * x
    loss = (y_hat - y) ** 2

    print(loss)

    # perform backpropagation
    loss.backward()
    print(w.grad)
    # update the weights


