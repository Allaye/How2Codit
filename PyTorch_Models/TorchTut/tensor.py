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

# create a new random tensor from tensor data, and change the data type to float
rand_t = torch.rand_like(t_np, dtype=torch.float32)

#indexing and slicing
tensor = torch.ones(3,3)
print("first row", tensor[0])
print("first column", tensor[:,0])

# reshape tensor
x = tensor.view(9)
y = tensor.view(-1)


# arithmatic operations on tensor

y1 = tensor @ tensor
y2 = tensor.matmul(tensor)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)