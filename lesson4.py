# mathematics operations.
import torch
import numpy as np
my_device = torch.device("mps")

tensor_a = torch.tensor([1, 2, 3, 4], device=my_device)
tensor_b = torch.tensor([5, 6, 7, 8], device=my_device)
# Addition
print(tensor_a + tensor_b)

# Addition longhand
print(torch.add(tensor_b, tensor_a))
print(torch.cuda.is_available())

# Subtraction
print(tensor_b - tensor_a)
print(torch.subtract(tensor_b, tensor_a))

# Multiply
print(tensor_a * tensor_b)
print(torch.mul(tensor_b, tensor_a))

# Division
print(tensor_a/tensor_b)
print(torch.div(tensor_a, tensor_b))

# Remainder
print(tensor_a % tensor_b)
print(torch.remainder(tensor_a, tensor_b))

# Exponential
print(torch.pow(tensor_b, tensor_a))

# Reassignment
# tensor_a = tensor_a + tensor_b
print(tensor_a)
tensor_a.add_(tensor_b)
print(tensor_a)

# Test case:
# a = torch.rand(1, 4, device=my_device)
# b = a
# a = a + tensor_a
# print(a)
# print(b)


