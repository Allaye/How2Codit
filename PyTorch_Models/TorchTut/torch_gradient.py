from numpy import dtype
import torch
import torch.nn as nn


# pytorch nn pipeline
# 1. define the model, design the input, output layer
# 2. construct the loss function and the optimizer
# 3. train the model
     # 3.1 forward pass (forward propagation): compute the modelprediction
     # 3.2 backward pass (backward propagation): compute the gradient of the loss function
     # 3.3 update the parameters (update the model parameters like weight and bias)



X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

test = torch.tensor([5], dtype=torch.float32)


def forward(X):
    # _, n_features = X.shape
    #input_size = n_features
    #output_size = n_features
    return nn.Linear(1, 1)

def loss(Y, Y_predicted):
    # calculate loss MSE
    l = nn.MSELoss()
    return l(Y, Y_predicted)

def gradient(l):
    # calculate = backwardpass
    # pytorch calculates the local derivative of the loss function
    return l.backward()

def optimize(w, lr=0.01):
    # define the optimizer
    # update the weight by moving in the negative direction of the gradient

    return torch.optim.SGD(w, lr=0.01)

def training(lr, iters, X, Y):
    m = forward(X)
    print(f"The predicted Y before training for value X: 5 is {m(test).item():.2f}")
    for epoch in range(iters):
        # forward pass, calculate the predicted value
        model = forward(X)
        Y_pred = model(X)
        # calculate the loss
        los = loss(Y, Y_pred)
        # calculate the gradient
        gradient(los)
        # update the weight by moving in the negetive direction of the gradient
        optimizer = optimize(model.parameters(), lr)
        optimizer.step()
        # with torch.no_grad():
        #     w -= lr * w.grad
        optimizer.zero_grad()
        if epoch % 10 == 0:
            w, b = model.parameters()
            print("epoch: {}, loss: {:.8f}, w: {}".format(epoch, los, w[0][0].item()))
    print(f"The predicted Y After training for value X: 5 is {model(test).item():.2f}")


if __name__ == "__main__":
    print(type(X))
    print(X.shape)
    training(lr=0.01, iters=100, X=X, Y=Y)


