import numpy as  np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)


w = 0.0

def forward(X):
    # calculate the forward function to calculate for the predicted value
    return X * w

def loss(Y, Y_predicted):
    # loss MSE 
    # MSE = 1/N * (w*x -y)**2
    # where w*x is the value of the forward function
    return ((Y_predicted - Y) ** 2).mean()

def gradient():
    pass
    