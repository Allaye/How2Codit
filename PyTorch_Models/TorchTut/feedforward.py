import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt



# pipeline
# 1. load data and process data
# 2. build model 
# 3. calculate loss and optimizer
# 4. perform training using batch
# 5. evelauate model
# 6. save model

# Device configuration
def configure_device():
    '''
    check if the working device has CUDA support
    if yes use the GPU otherwise use the CPU
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device

# define hyper parameters
def hyper_parameters():
    '''
    define hyper parameters used in the model
    '''
    learning_rate = 0.001
    input_size = 784
    hidden_size = 500
    num_classes = 10
    epochs = 10
    batch_size = 100
    return learning_rate, input_size, hidden_size, num_classes, epochs, batch_size

# papare dataset
def prepare_dataset(batch_size):
    '''
    get the mnist dataset, transform it to tensor and normalize it
    '''
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./',train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class FeedForward(nn.Module):
    '''
    define the feed forward neural network, with all neccesary working part
    this class is used to design the network
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        '''
        initialize the feed forward network'''
        super().__init__()
        self.f1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        '''
        forward pass of the feed forward network
        '''
        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        return out

    def loss_optimizer(self, lr):
        '''
        define loss and optimizer
        '''
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return loss_fn, optimizer


