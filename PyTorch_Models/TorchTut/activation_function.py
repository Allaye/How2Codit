import torch
import torch.nn as nn



# first way of using activation function
class ActivationFunction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        return x

# second way of using activation function
class ActivationFunction2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x 