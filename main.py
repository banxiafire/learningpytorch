import torch
import numpy as np
mps_device = torch.device("mps")
x = torch.rand(2, 5, device=mps_device)
y = np.random.rand(3, 4)
torch_3d = torch.zeros(2, 3, 4, device=mps_device)  # Matrix with same structure
# but all zero
my_tensor = torch.tensor(y, dtype=torch.float, device=mps_device)
print(x)
print(torch_3d.dtype)
print(my_tensor.device)


# lesson2
les2 = torch.arange(10, device=mps_device)
print(les2)
les2 = les2.reshape(2, 5)
print(les2)
les2 = les2.reshape(10)
les22 = les2
print(les2)
les2 = les2.reshape(2, -1)
print(les2.size())
les2[1, 1] = 901
print(les2)
print(les22)  # the mem will change, reshape only changes it's view but won't
# change it's actual inside, and all change of num inside will reflect
print(les2[:, 1:])  # extract certain row or col
