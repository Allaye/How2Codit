import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


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
    train_dataset = torchvision.datasets.CIFAR10(root='./',train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


# define the model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss_optimizer(self, lr=0.001):
        '''
        define loss and optimizer
        '''
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return loss_fn, optimizer


def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, device, batch_size):
    '''
    perform training on the model, update hyper parameters
    '''
    n_total_steps = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # prepare the images by flattening them
            # images = images.reshape(-1, 28*28)
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)

            # calculate loss
            loss = loss_fn(outputs, labels)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training statistics and information
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, n_total_steps, loss.item()))
    eval_model(model, test_loader, device, batch_size)


def eval_model(model, test_loader, device, batch_size):
    '''
    evaluate the model
    '''
    # loop over test data
    with torch.no_grad():
        total_correct = 0
        total_sample = 0
        n_class_correct = [0 for i in range(10)]
        n_class_sample = [0 for i in range(10)]
        for images, labels in test_loader:
            # move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            # images = images.reshape(-1, 28*28)
            outputs = model(images)
            
            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_sample += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_sample[label] += 1
        accuracy = 100.0 * total_correct / total_sample
        print('Accuracy of the network on the 10000 test images: {} %'.format(accuracy))

        for i in range(10):
            accuracy = 100.0 * n_class_correct[i] / n_class_sample[i]
            print('Accuracy of {} class: {} %'.format(i, accuracy))


if __name__ == '__main__':
    # load hyper parameters
    learning_rate, input_size, hidden_size, num_classes, epochs, batch_size = hyper_parameters()
    # load dataset
    train_loader, test_loader, classes = prepare_dataset(batch_size)
    # configure device
    device = configure_device()
    # define model
    model = CNN().to(device)
    # define loss and optimizer
    loss_fn, optimizer = model.loss_optimizer(lr=learning_rate)
    # train model
    train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, device, batch_size)