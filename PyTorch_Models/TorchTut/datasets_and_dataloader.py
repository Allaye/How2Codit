import math
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader



'''
# epoch = a complete forward/backward pass of all training dataset
# batch_size = number of training sample in a forward/backward pass
# number of iteration = 100 data sample, batch_size of 20 = 100/20 = 5 iteration for i epoch

'''



class CustomDatasetLoader(Dataset):
    """
    A class to create a custom dataset, which is a subclass of torch.utils.data.Dataset
    and a data loader, which is a subclass of torch.utils.data.DataLoader

    
    """
    def __init__(self, data_path, delimiter=',', dtype=np.float32, skiprows=0):
        self.dataset = np.loadtxt(data_path, delimiter=delimiter, dtype=dtype, skiprows=skiprows)
        self.features = torch.from_numpy(self.dataset[:, 1:])
        self.labels = torch.from_numpy(self.dataset[:, 0])
        self.n_samples = self.dataset.shape[0]

    
    def __getitem__(self, index):
        """
        override the __getitem__ method to return the data sample
        """
        return self.features[index], self.labels[index]

    def __len__(self):
        """
        override the __len__ method to return the number of data samples
        """
        return self.n_samples

    def loader(self, batch_size=5, shuffle=True, num_workers=0):
        """
        create a customizable data loader
        """
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



data = CustomDatasetLoader(data_path='./wine.csv', skiprows=1)
first_data = data[0]
features, labels = first_data
print(features, labels, ';;;;;')

dataloader = data.loader()
next_batch = next(iter(dataloader))
print(next_batch)

