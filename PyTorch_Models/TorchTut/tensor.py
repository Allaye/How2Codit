import torch
import numpy as np


# create a 3 by 3 matrix 
data = [[1,2,3],[4,5,6],[7,8,9]]
#convert the metrix to tensor
d_t = torch.tensor(data)

print(d_t)
print(d_t.shape)
print(d_t.size())

# create a tensor from numpy
np_data = np.array([[2,4,3],[1,2,3],[4,5,6]])
t_np = torch.from_numpy(np_data)

# create a new tensor from tensor data
one_t = torch.ones_like(t_np)

# create a new random tensor from tensor data
rand_t = torch.rand_like(t_np)
