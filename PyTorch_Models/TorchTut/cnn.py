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

