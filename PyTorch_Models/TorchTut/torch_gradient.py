import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(X, w=torch.tensor(0.0 , dtype=torch.float32, requires_grad=True)):
    # calculate the forward function to calculate for the predicted value
    return X * w

def loss(Y, Y_predicted):
    # loss MSE 
    # MSE = 1/N * (w*x -y)**2
    # where w*x is the value of the forward function
    return ((Y_predicted - Y) ** 2).mean()

def gradient(l):
    # calculate = backwardpass 
    # pytorch calculates the local derivative of the loss function
    return l.backward()

def training(lr, iters, X, Y, w):
    print(f"The predicted Y before training for value X: 5 is {forward(5, w):.2f}")
    for epoch in range(iters):
        # forward pass, calculate the predicted value
        Y_pred = forward(X, w)
        # calculate the loss
        los = loss(Y, Y_pred)
        # calculate the gradient
        gradient(los)
        # update the weight by moving in the negetive direction of the gradient
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        if epoch % 10 == 0:
            print("epoch: {}, loss: {:.8f}, w: {}".format(epoch, los, w))
    print(f"The predicted Y After training for value X: 5 is {forward(5, w):.2f}")


if __name__ == "__main__":
    training(lr=0.01, iters=100, X=X, Y=Y, w=w)


