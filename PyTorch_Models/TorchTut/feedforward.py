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

    def loss_optimizer(self, lr=0.001):
        '''
        define loss and optimizer
        '''
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return loss_fn, optimizer


def train_model(model, train_loader, loss_fn, optimizer, epochs, device):
    '''
    perform training on the model, update hyper parameters
    '''
    n_total_steps = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # prepare the images by flattening them
            images = images.reshape(-1, 28*28)
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            # calculate loss
            loss = loss_fn(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training statistics and information
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, n_total_steps, loss.item()))





if __name__ == "__main__":
    # configure the device
    device = configure_device()
    # define hyper parameters
    learning_rate, input_size, hidden_size, num_classes, epochs, batch_size = hyper_parameters()
    # prepare dataset
    train_loader, test_loader = prepare_dataset(batch_size)
    # define feed forward network
    model = FeedForward(input_size, hidden_size, num_classes)
    # define loss and optimizer
    loss_fn, optimizer = model.loss_optimizer(lr=learning_rate)
    # train the model
    train_model(model, train_loader, loss_fn, optimizer, epochs, device)
    # evaluate the model