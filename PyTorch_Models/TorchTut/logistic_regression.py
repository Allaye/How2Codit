import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# pytorch pipline
# Design model, design input, output layer
# Construct loss function and optimizer
# Train the model
   # 3.1 forward pass (forward propagation): compute the model prediction
   # 3.2 backward pass (backward propagation): compute the gradient of the loss function with respect to the model parameters
   # 3.3 update the model parameters
# create the dataset and prediction visualization function
# provide the model with the input and compute the prediction

dataset = load_breast_cancer()
data = dataset.data
target = dataset.target


def prepare_data():
   # load the dataset
   dataset = load_breast_cancer()
   # split the dataset into features and target
   features, target = dataset.data, dataset.target
   # split the dataset into training and testing sets
   x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
   # standardize the features
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)
   x_test = scaler.transform(x_test)
   x_train = torch.from_numpy(x_train).float()
   x_test = torch.from_numpy(x_test).float()
   # convert the target to a tensor and reshape  to a column vector (n, 1)
   y_train = torch.from_numpy(y_train).float().view(y_train.shape[0], 1)
   y_test = torch.from_numpy(y_test).float().view(y_test.shape[0], 1)
   return x_train, x_test, y_train, y_test
   

class LogisticRegression(nn.Module):

   def __init__(self, n_features):
      super().__init__()
      self.linear = nn.Linear(n_features, 1)

   def forward(self, x):
      y_pred = torch.sigmoid(self.linear(x))
      return y_pred

   def fit(self, x, y, epochs=100, lr=0.01):
      # define the loss function and the optimizer
      criterion = nn.BCELoss()
      optimizer = torch.optim.SGD(self.parameters(), lr)
      # train the model
      for epoch in range(epochs):
         # 1. forward pass
         y_pred = self.forward(x)
         # 2. compute the loss
         loss = criterion(y_pred, y)
         # 3. backward pass
         loss.backward()
         # 4. update the parameters
         optimizer.step()
         # 5. reset the gradients to zero
         optimizer.zero_grad()
         # print out the training progress and details
         if epoch % 10 == 0:
            w, b = self.parameters()
            print(f'epoch {epoch+1}/{epochs} loss: {loss.item()} w: {w.item()} b: {b.item()}')
            # print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')
            self.evaluate(x, y)
      return -1
   
   def evaluate(self, x, y):
      # stop possible gradient flow
      with torch.no_grad():
         # compute the model prediction
         y_pred = self.forward(x)
         # compute the model accuracy
         y_pred = (y_pred > 0.5).float()
         accuracy = (y_pred == y).float().mean()
         print(f'accuracy: {accuracy.item():.4f}')
         return accuracy



if __name__ == '__main__':
   x_train, x_test, y_train, y_test = prepare_data()
   model = LogisticRegression(x_train.shape[1])
   model.fit(x_train, y_train)
   model.evaluate(x_test, y_test)