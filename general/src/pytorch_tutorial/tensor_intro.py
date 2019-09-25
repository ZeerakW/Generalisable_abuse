import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Set seed.
torch.manual_seed(32)

# Create dummy data as a vector
vector_data = [1., 2., 3.]
V = torch.tensor(vector_data)
print(V)

# Create dummy data as a matrix
matrix_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.tensor(matrix_data)
print(M)


# Create dummy data as a tensor
tensor_data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
T = torch.tensor(matrix_data)
print(T)

# Adding the flag requires_grad = True on tensors allows for backpropagation because a pointer
# network is maintained of the actions taken to obtain the current stage.

# User generated tensors default to not having autograd set to true.
x = torch.randn(2, 2)
x = x.requires_grad_(True)  # Modify to add gradient.
y = torch.randn(2, 2, requires_grad = True)

z = x + y  # If any input to an operation has "requires_grad = True then so will the output.

# We can detatch z from the computational history - this also means that backprop is not available for it.
new_z= z.detatch()

# To prevent tracking hisotry on tensors use with torch.no_grad:

print(x.requires_grad)  # Requires_grad returns a boolean indicating whether the requires_grad flag is on.
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


