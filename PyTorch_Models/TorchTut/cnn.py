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
