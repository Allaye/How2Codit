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


if __name__ == "__main__":
    # create respective tensor with the weight have grad func activated
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)
    w = torch.tensor(1.0, requires_grad=True)

    backpropagation(x, y, w)
