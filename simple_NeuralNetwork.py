import torch
import torch.nn as nn
import torch.nn.functional as F
my_device = torch.device("mps")

# creat a Model class that inherits nn.Module

class Model(nn.Module):
    # Input Layer(4 feature of a flower to) -> hidden layer1 -> h2(n)
    # -> output 3 layers
    def __init__(self, in_feature=4, h1=8, h2=9, output_features=3, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_feature, h1, device=my_device)
        self.fc2 = nn.Linear(h1, h2, device=my_device)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

torch.manual_seed(41)

model = Model()
