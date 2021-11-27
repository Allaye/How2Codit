import numpy as  np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)


w = 0.0

def forward(X, w=0.0):
    # calculate the forward function to calculate for the predicted value
    return X * w

def loss(Y, Y_predicted):
    # loss MSE 
    # MSE = 1/N * (w*x -y)**2
    # where w*x is the value of the forward function
    return ((Y_predicted - Y) ** 2).mean()

def gradient(X, Y, Y_predicted):
    # calculate the gradient of the loss function inrespect to the weight
    # MSE = 1/N * (w*x -y)**2
    # dMSE/dw = 2/N * (w*x -y) * x 
    # or 
    # dMSE/dw = 1/N  2x (w*x -y)
    return np.dot(2*X, Y_predicted - Y).mean()

def training(lr, iters, X, Y, w):
    print(f"The predicted Y before training for value X: 5 is {forward(5, w):.2f}")
    for epoch in range(iters):
        # forward pass, calculate the predicted value
        Y_pred = forward(X, w)
        # calculate the loss
        los = loss(Y, Y_pred)
        # calculate the gradient
        dw = gradient(X, Y, Y_pred)
        # update the weight by moving in the negetive direction of the gradient
        #w = w - lr * dw
        w -= lr * dw
        if epoch % 1 == 0:
            print("epoch: {}, loss: {:.8f}, w: {}".format(epoch, los, w))
    print(f"The predicted Y After training for value X: 5 is {forward(5, w):.2f}")


if __name__ == "__main__":
    training(lr=0.01, iters=20, X=X, Y=Y, w=w)


